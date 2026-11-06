"""Shared Crew attribution for people the Kiro relay hands over.

Driven over a real loopback listener with the relay ingress, the client-sent strip
and token auth chained as both gateway chains order them; the relay client is the
seam it calls, ``RelayIngress.stamp_headers``. A relayed teammate is published as
their own ``kiro`` principal, keyed by their Kiro user ID, and the relayed owner of a
personal Crew as the owner at the machine, with ``kiro_relay`` provenance. The
second half walks every owner check that compares principal ids only (listed in the
module's own table below) and pins what each answers for the relayed owner.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from shared_crew_host_isolation import host_isolation  # noqa: F401
from token_auth_helpers import clear_nonces  # noqa: F401 -- the autouse fixture registers by import

from kiro_crew.dashboard.kiro_relay.caller import (
    CrewKind,
    HostBinding,
    RelayCaller,
    RelayCallerKind,
)
from kiro_crew.dashboard.kiro_relay.ingress import (
    RELAY_INGRESS_APP_KEY,
    RelayIngress,
    relay_ingress_middleware,
)
from kiro_crew.dashboard.relay_headers import relay_header_strip_middleware
from kiro_crew.dashboard.server_internal_paths import _STRICT_INTERNAL_API_PATHS
from kiro_crew.dashboard.token_auth import generate_token, token_auth_middleware
from kiro_crew.shared_crew import identity, owner, people
from kiro_crew.shared_crew.contracts import (
    PrincipalKind,
    Provenance,
    Transport,
    principal_id_for,
)
from kiro_crew.shared_crew.identity import (
    actor_of,
    local_operator_principal,
    tailnet_principal,
)


def relay_principal(caller: RelayCaller, owner_subject: str):
    """Looked up at call time, so each test fails on its own before the branch."""
    return identity.relay_principal(caller, owner_subject)


pytestmark = pytest.mark.usefixtures("host_isolation")

OWNER = "kiro-user.owner-0001"
ANA = "kiro-user.teammate-ana-0002"
BEA = "kiro-user.teammate-bea-0003"
NAME = "Ren\u00e9e \u00d8"
EMAIL = "renee@example.com"


@pytest.fixture(autouse=True)
def _fresh(_floor_monkeypatch: pytest.MonkeyPatch):
    _floor_monkeypatch.setenv("KIROCREW_SHARED_CREW_PREVIEW", "1")
    people.reset_for_tests()
    yield
    people.reset_for_tests()


async def _whoami(request: web.Request) -> web.Response:
    from kiro_crew.shared_crew import routes_invites
    from kiro_crew.shared_crew.admission_gate import actor_budget
    from kiro_crew.shared_crew.attribution import with_request_author
    from kiro_crew.shared_crew.collaborators import is_collaborator_actor

    actor = actor_of(request)
    refusal = await routes_invites._owner_refusal_decision(request)
    return web.json_response(
        {
            "principal": actor.principal.to_dict(),
            "transport": actor.transport.value,
            "budget": actor_budget(request),
            "author": (with_request_author({}) or {}).get("author"),
            "collaborator": is_collaborator_actor(actor),
            "owner_page": refusal.status if refusal is not None else 200,
            "owner_page_code": json.loads(refusal.text)["code"] if refusal is not None else "",
        }
    )


def _app(ingress: RelayIngress, *, owner_id: str = "") -> web.Application:
    app = web.Application(
        middlewares=[
            relay_ingress_middleware(strict_internal_paths=_STRICT_INTERNAL_API_PATHS),
            relay_header_strip_middleware,
            token_auth_middleware(port=5476),
        ]
    )
    app["state"] = SimpleNamespace(owner_id=owner_id)
    app["allowed_origins"] = set()
    app[RELAY_INGRESS_APP_KEY] = ingress
    app.router.add_get("/api/whoami", _whoami)
    return app


def _ingress(kind: CrewKind) -> RelayIngress:
    ingress = RelayIngress(enabled=True)
    owner_id = OWNER if kind is CrewKind.PERSONAL else ""
    ingress.bind_host(HostBinding(crew_kind=kind, owner_user_id=owner_id))
    return ingress


async def _as(client: TestClient, user: str, **labels: str) -> dict[str, Any]:
    ingress = client.server.app[RELAY_INGRESS_APP_KEY]
    headers = ingress.stamp_headers([], relay_user=user, **labels)
    resp = await client.get("/api/whoami", headers=headers)
    assert resp.status == 200, await resp.text()
    return await resp.json()


# ---------------------------------------------------------------------------
# Teammates on a team Crew
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_relayed_teammate_is_their_own_kiro_principal() -> None:
    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        body = await _as(client, ANA, relay_email=EMAIL, relay_name=NAME)
    principal = body["principal"]
    assert principal["kind"] == PrincipalKind.HUMAN.value
    assert principal["provider"] == "kiro"
    assert principal["issuer"] == "kiro-relay"
    assert principal["subject"] == ANA
    assert principal["provenance"] == Provenance.KIRO_RELAY.value
    assert principal["id"] == principal_id_for("kiro", "kiro-relay", ANA)
    assert principal["display"]["name"] == NAME
    assert principal["display"]["login"] == EMAIL
    assert body["transport"] == Transport.REMOTE.value
    # Their messages carry their own principal, not Unknown.
    assert body["author"]["id"] == principal["id"]
    # People lists them.
    assert [obs.principal.id for obs in people.registry().snapshot()] == [principal["id"]]


@pytest.mark.asyncio
async def test_the_id_is_the_kiro_user_id_alone_never_the_labels() -> None:
    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        named = await _as(client, ANA, relay_email=EMAIL, relay_name=NAME)
        renamed = await _as(client, ANA, relay_email="other@example.com", relay_name="Other")
        bare = await _as(client, ANA)
        # Exact: no case folding, so another ID is another person.
        upper = await _as(client, ANA.upper())
    ids = {b["principal"]["id"] for b in (named, renamed, bare)}
    assert len(ids) == 1
    assert upper["principal"]["id"] not in ids


@pytest.mark.asyncio
async def test_two_teammates_are_two_people_with_two_budgets_and_one_typing_lease() -> None:
    from kiro_crew.shared_crew.collab import Activity
    from kiro_crew.shared_crew.contracts import Principal
    from kiro_crew.shared_crew.presence import PresenceRegistry
    from kiro_crew.shared_crew.typing_lease import TypingLeaseRegistry, can_hold

    ACTIVITY_TYPING = Activity.TYPING

    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        ana = await _as(client, ANA, relay_name="Ana")
        bea = await _as(client, BEA, relay_name="Bea")
    assert ana["principal"]["id"] != bea["principal"]["id"]
    assert ana["budget"] == ana["principal"]["id"]
    assert bea["budget"] == bea["principal"]["id"]
    a = Principal.from_dict(ana["principal"])
    b = Principal.from_dict(bea["principal"])

    presence = PresenceRegistry()
    presence.report("conn-a", a, "s1", ACTIVITY_TYPING)
    presence.report("conn-b", b, "s1", ACTIVITY_TYPING)
    assert {e.principal.id for e in presence.entries("s1")} == {a.id, b.id}

    leases = TypingLeaseRegistry()
    assert can_hold(a) and can_hold(b)
    leases.typed("s1", "conn-a", a)
    assert leases.holder("s1") == a
    # Bea's send is refused while Ana holds it (409 composer_locked at the route).
    assert leases.blocks("s1", b) == a
    assert leases.blocks("s1", a) is None
    # And Bea may take it over.
    leases.take_over("s1", "conn-b", b)
    assert leases.holder("s1") == b


@pytest.mark.asyncio
async def test_a_teammate_with_no_name_or_email_gets_the_fallback_label() -> None:
    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        bare = await _as(client, ANA)
        email_only = await _as(client, BEA, relay_email=EMAIL)
    assert bare["principal"]["display"]["name"] == "Kiro user ...a-0002"
    assert bare["principal"]["display"]["login"] == ""
    assert email_only["principal"]["display"]["name"] == EMAIL


@pytest.mark.asyncio
async def test_a_relayed_teammate_is_never_a_collaborator_or_the_owner() -> None:
    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        body = await _as(client, ANA, relay_name=NAME)
    assert body["collaborator"] is False
    assert body["owner_page"] == 403
    pid = body["principal"]["id"]
    assert pid not in owner.owner_principal_ids()


# ---------------------------------------------------------------------------
# The relayed owner of a personal Crew
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_id", ["", "U0OWNER"])
async def test_the_relayed_owner_is_the_owner_at_the_machine_and_owner_pages_stay_refused(
    owner_id: str,
) -> None:
    owner.set_owner_subject(owner_id)
    async with TestClient(TestServer(_app(_ingress(CrewKind.PERSONAL), owner_id=owner_id))) as c:
        relayed = await _as(c, OWNER, relay_name=NAME)
        token = generate_token(owner_id or "local-app")
        resp = await c.get("/api/whoami", params={"token": token})
        local = await resp.json()
    machine = local_operator_principal(owner_id or "local-app")
    assert local["principal"]["id"] == machine.id
    # One person: the same id from the machine and over the relay.
    assert relayed["principal"]["id"] == machine.id
    assert relayed["budget"] == local["budget"]
    assert relayed["principal"]["provenance"] == Provenance.KIRO_RELAY.value
    assert relayed["author"]["id"] == machine.id
    # Shared Crew's owner pages need the local operator, so they stay refused.
    assert local["owner_page"] == 200
    assert relayed["owner_page"] == 403
    assert relayed["owner_page_code"] == "owner_only"
    # People recognizes them as the viewer and the owner, in one row.
    view = people.people_view(
        relayed_principal := relay_principal(
            RelayCaller(kiro_user_id=OWNER, kind=RelayCallerKind.OWNER), owner_id or "local-app"
        ),
        [],
        owner_id=owner_id,
    )
    rows = [r for r in view["people"] if r["principal"]["id"] == relayed_principal.id]
    assert len(rows) == 1
    assert rows[0]["is_viewer"] is True and rows[0]["is_owner"] is True


@pytest.mark.asyncio
async def test_a_tailnet_peer_keeps_its_tailscale_provenance() -> None:
    """Guard: the relay branch never runs without a published relay caller."""
    peer = tailnet_principal("alice@example.com", "alice-laptop", Provenance.TAILSCALE_SERVE)
    assert peer.provider == "tailscale" and peer.provenance is Provenance.TAILSCALE_SERVE
    async with TestClient(TestServer(_app(_ingress(CrewKind.TEAM)))) as client:
        token = generate_token("local-app")
        resp = await client.get(
            "/api/whoami",
            params={"token": token},
            headers={"X-Kiro-Relay-User": ANA, "X-Kiro-Relay-User-Name": "UTF-8''Forged"},
        )
        body = await resp.json()
    assert body["principal"]["provenance"] == Provenance.LOCAL_OPERATOR.value
    assert body["principal"]["provider"] == "local"


# ---------------------------------------------------------------------------
# Every owner check that compares principal ids only, for the relayed owner
#
# The relayed owner of a personal Crew shares the owner's principal id, so each
# check that compares ids alone now answers "owner" for them. Each one below is a
# label, a list filter, a fold of the owner's own ids or the typing lease, the
# same answers the owner gets over Tailscale. The checks that keep something
# refused remotely read the local-operator provenance, which a relayed principal
# never has, and stay refused.
# ---------------------------------------------------------------------------


def _relayed_owner(owner_subject: str = "local-app"):
    caller = RelayCaller(kiro_user_id=OWNER, kind=RelayCallerKind.OWNER, display_name=NAME)
    return relay_principal(caller, owner_subject)


def test_is_owner_principal_and_viewer_ids_name_the_relayed_owner() -> None:
    relayed = _relayed_owner()
    assert relayed.provenance is Provenance.KIRO_RELAY
    assert owner.is_owner_principal(relayed)
    assert owner.viewer_ids(relayed) == owner.owner_principal_ids()


def test_the_mine_filter_finds_the_owners_machine_sessions_for_the_relayed_owner() -> None:
    from kiro_crew.shared_crew.contracts import SessionQuery

    relayed = _relayed_owner()
    machine = local_operator_principal("local-app")
    mine = SessionQuery.from_query({"creator": "mine"})
    assert mine.matches(machine, relayed, viewer_ids=owner.viewer_ids(relayed))
    folded = SessionQuery.from_query({"creator": machine.id})
    assert folded.matches(machine, relayed, local_owner_ids=owner.local_owner_principal_ids())


def test_people_marks_the_relayed_owner_as_owner_and_lists_sharees_as_over_tailscale() -> None:
    from kiro_crew.shared_crew.tailscale_sharees import Sharee

    relayed = _relayed_owner()
    sharee_principal = tailnet_principal("bob@example.com", "bob", Provenance.TAILSCALE_SERVE)
    sharee = Sharee(
        user=SimpleNamespace(user_id="u-bob", profile_pic_url=""),  # type: ignore[arg-type]
        principal=sharee_principal,
    )
    view = people.people_view(relayed, [], sharees=(sharee,))
    ids = {r["principal"]["id"]: r for r in view["people"]}
    assert ids[relayed.id]["is_owner"] is True and ids[relayed.id]["is_viewer"] is True
    # The owner, from anywhere, sees who the machine is shared with.
    assert sharee_principal.id in ids
    # A relayed teammate does not.
    teammate = relay_principal(RelayCaller(kiro_user_id=ANA, kind=RelayCallerKind.TEAMMATE), "")
    other = people.people_view(teammate, [], sharees=(sharee,))
    assert sharee_principal.id not in {r["principal"]["id"] for r in other["people"]}


def test_the_prompt_reads_the_relayed_owners_message_as_the_owners() -> None:
    from kiro_crew.shared_crew.attribution import author_meta, sender_line
    from kiro_crew.shared_crew.contracts import ActorContext

    relayed = _relayed_owner()
    meta = author_meta(
        ActorContext(
            principal=relayed, request_id="r", transport=Transport.REMOTE, correlation_id="r"
        )
    )
    row = {"role": "user", "meta": meta, "content": "hi"}
    slot = SimpleNamespace(messages=[row])
    assert sender_line(slot, row) == ""


def test_the_typing_lease_is_one_holder_for_the_owners_two_tabs() -> None:
    from kiro_crew.shared_crew.typing_lease import TypingLeaseRegistry, same_person

    relayed = _relayed_owner()
    machine = local_operator_principal("local-app")
    assert same_person(relayed, machine)
    leases = TypingLeaseRegistry()
    leases.typed("s1", "machine-tab", machine)
    assert leases.blocks("s1", relayed) is None


def test_the_operator_id_expansion_needs_the_local_operator() -> None:
    """The local-operator fold of every bootstrap subject is not opened by the relay."""
    from kiro_crew.shared_crew.crew_mode import operator_ids

    relayed = _relayed_owner()
    assert operator_ids(relayed) == frozenset({relayed.id})
    assert len(operator_ids(local_operator_principal("local-app"))) > 1


def test_a_relayed_owner_counts_as_a_network_path() -> None:
    """Provenance gates read the relay as remote: stricter, never wider."""
    from kiro_crew.shared_crew.crew_mode import others_observed
    from kiro_crew.shared_crew.memory_changes import crew_is_shared

    relayed = _relayed_owner()
    assert others_observed(local_operator_principal("local-app"), [relayed]) is True
    assert crew_is_shared(SimpleNamespace(), relayed) is True
