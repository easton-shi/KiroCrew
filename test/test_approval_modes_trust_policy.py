"""A ``permission_mode`` deny of ``trust`` / ``trust_reads`` holds at every entry point.

RFC crew-enterprise-edition D8. The verdict is pushed at ceiling install
(``kiro_crew.approval_mode_policy``); these cases install a real ceiling and check each
place a session's Trust is set or honoured: the dashboard mode API, the approval card,
the picker's disabled list, the slot's own flags, the stored ``"auto"`` policy, the
messaging Trust set, the crewmate default, a cron job's ``auto`` and a spawn's ``auto``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kiro_crew import approval_mode_policy as amp
from kiro_crew import safety_override as so_mod
from kiro_crew.dashboard.state import DashboardState
from kiro_crew.history import ConversationLog
from kiro_crew.platform import context as ctx_mod
from kiro_crew.platform import governance_profiles as gp
from kiro_crew.platform.bootstrap import build_default_context
from kiro_crew.platform.governance import parse_policy

#: The owner's rule, after translation: Trust is not offered.
OWNER_RULE = {"mode": "allow", "allow": ["normal", "trust_reads"]}
ALL_THREE = {"mode": "allow", "allow": ["normal", "trust_reads", "trust"]}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, _floor_monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    _floor_monkeypatch.setattr(gp, "_PROFILES_DIR", d)
    _floor_monkeypatch.setattr("kiro_crew.dashboard.state.config_dir", lambda: tmp_path)
    gp.reset_store()
    ctx_mod.reset_context()
    so_mod.reset_singleton()
    amp.reset_state()
    yield
    so_mod.reset_singleton()
    gp.reset_store()
    ctx_mod.reset_context()
    amp.reset_state()


def _install(rule: dict | None) -> None:
    """Install *rule* as the ``approval_modes`` ceiling (``None``: no policy)."""
    from kiro_crew.config.loader import KiroCrewConfig

    base = build_default_context(KiroCrewConfig.load())
    ceiling = (
        None
        if rule is None
        else parse_policy({"version": 1, "boot": {"fail_closed": True}, "approval_modes": rule})
    )
    ctx_mod.set_context(dataclasses.replace(base, governance=ceiling))


def _make_state(tmp_path) -> DashboardState:
    sessions = MagicMock(count=0)
    sessions.get_pid = MagicMock(return_value=None)
    sessions.remove = AsyncMock()
    return DashboardState(
        sessions=sessions,
        crons=MagicMock(list_jobs=MagicMock(return_value=[]), status=MagicMock(return_value={})),
        lessons=MagicMock(load_all=MagicMock(return_value=[])),
        start_time=0.0,
        conversation_log=ConversationLog(base_dir=tmp_path),
    )


def _make_app(state: DashboardState) -> web.Application:
    from kiro_crew.dashboard.chat import api_chat_mode, api_chat_slot_approve

    @web.middleware
    async def _test_auth(request: web.Request, handler):
        if "app" not in request:
            request["app"] = ""
        if "user" not in request:
            request["user"] = "local-app"
        return await handler(request)

    app = web.Application(middlewares=[_test_auth])
    app["state"] = state
    app.router.add_post("/api/chat/slots/{slot}/approve", api_chat_slot_approve)
    app.router.add_post("/api/chat/mode", api_chat_mode)
    return app


async def _set_mode(state: DashboardState, mode: str) -> tuple[int, dict]:
    async with TestClient(TestServer(_make_app(state))) as client:
        resp = await client.post("/api/chat/mode", json={"mode": mode, "slot": "s1"})
        return resp.status, await resp.json()


class TestTheModeApi:
    @pytest.mark.asyncio
    async def test_the_owner_rule_refuses_trust_and_keeps_reads(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
        _install(OWNER_RULE)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")

        status, body = await _set_mode(state, "trust")
        assert (status, body["code"]) == (403, "mode_disabled_by_policy")
        assert slot._trust is False and slot._trust_held is False, "refused before mutation"

        status, _ = await _set_mode(state, "trust_reads")
        assert status == 200
        assert slot._trust_reads is True

    @pytest.mark.asyncio
    async def test_a_rule_allowing_all_three_keeps_trust(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
        _install(ALL_THREE)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")

        status, _ = await _set_mode(state, "trust")
        assert status == 200
        assert slot._trust is True

    @pytest.mark.asyncio
    async def test_no_rule_keeps_trust(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
        _install(None)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")

        status, _ = await _set_mode(state, "trust")
        assert status == 200
        assert slot._trust is True

    @pytest.mark.asyncio
    async def test_an_unknown_mode_id_is_refused(self, tmp_path):
        _install(None)
        state = _make_state(tmp_path)
        state.get_or_create_slot("s1")

        for mode in ("trust_all", "TRUST", ""):
            status, body = await _set_mode(state, mode)
            assert (status, body["code"]) == (400, "unknown_approval_mode"), (mode, body)

    @pytest.mark.asyncio
    async def test_the_card_trust_press_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kiro_crew.dashboard.chat_handlers.sel", lambda: MagicMock())
        _install(OWNER_RULE)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        slot._approval_futures["req-1"] = fut

        async with TestClient(TestServer(_make_app(state))) as client:
            resp = await client.post(
                "/api/chat/slots/s1/approve", json={"action": "trust", "request_id": "req-1"}
            )
            body = await resp.json()

        assert (resp.status, body["code"]) == (403, "mode_disabled_by_policy")
        assert slot._trust_held is False
        assert not fut.done(), "the card stays open for Approve or Reject"


class TestThePickerList:
    def test_the_owner_rule_hides_trust(self):
        _install(OWNER_RULE)
        assert so_mod.cached_disabled_approval_modes() == ["trust", "yolo"]

    def test_all_three_hides_only_yolo(self):
        _install(ALL_THREE)
        assert so_mod.cached_disabled_approval_modes() == ["yolo"]

    def test_no_rule_hides_nothing(self):
        _install(None)
        assert so_mod.cached_disabled_approval_modes() == []


class TestAGrantMadeBeforeTheDeny:
    def test_the_slot_flags_are_masked_then_dropped(self, tmp_path):
        # Through the facade, whose names the function runs on.
        from kiro_crew.dashboard.server import _drop_policy_denied_trust

        _install(None)
        state = _make_state(tmp_path)
        slot = state.get_or_create_slot("s1")
        slot._trust = True
        assert slot._trust is True

        _install(OWNER_RULE)
        assert slot._trust is False, "a deny is honoured on the next read"
        assert slot._trust_held is True, "masked until the teardown runs"

        _drop_policy_denied_trust(state, frozenset({"trust"}))
        assert slot._trust_held is False
        state.sessions.set_approval_policy.assert_called_with("dashboard:s1", "")

        _install(None)
        assert slot._trust is False, "a later allow brings back no grant nobody re-made"

    def test_a_deny_runs_the_registered_teardown(self):
        seen: list[frozenset[str]] = []
        amp.register_deny_hook("test", seen.append)
        try:
            _install(OWNER_RULE)
        finally:
            amp._deny_hooks.pop("test")
        assert seen == [frozenset({"trust"})]

    def test_a_stored_auto_policy_is_not_honoured(self):
        _install(None)
        assert amp.stored_auto_permitted() is True
        _install(OWNER_RULE)
        assert amp.stored_auto_permitted() is False, "only a live YOLO grant keeps it"

    def test_the_session_store_reads_a_denied_auto_as_empty(self):
        from types import SimpleNamespace

        from kiro_crew.session_allocation import SessionAllocationService

        boundary = SimpleNamespace(
            _sessions={"k": SimpleNamespace(approval_policy="auto")},
            _owner=SimpleNamespace(_fold_key=lambda key: key),
        )
        _install(None)
        assert SessionAllocationService.get_approval_policy(boundary, "k") == "auto"
        _install(OWNER_RULE)
        assert SessionAllocationService.get_approval_policy(boundary, "k") == ""

    @pytest.mark.asyncio
    async def test_a_running_subagents_inherited_auto_ends_with_the_deny(self):
        from kiro_crew.tool_permission import ParentPolicyAuto

        _install(None)
        grant = ParentPolicyAuto("auto")
        assert await grant.offer(MagicMock()) is not None
        _install(OWNER_RULE)
        assert await grant.offer(MagicMock()) is None
        # The operator's ``auto_approve_subagent_tools`` grant is not a Trust.
        assert await ParentPolicyAuto("auto", trust_gated=False).offer(MagicMock()) is not None

    @pytest.mark.asyncio
    async def test_a_running_auto_cron_asks_after_the_deny(self):
        from kiro_crew.cron_service.fields import auto_job_approver

        asked: list[object] = []

        async def _ask(event):
            asked.append(event)
            return False

        approve = auto_job_approver(_ask)
        _install(None)
        assert await approve("e1") is True and asked == []
        _install(OWNER_RULE)
        assert await approve("e2") is False and asked == ["e2"]

    def test_an_unrelated_install_under_yolo_keeps_standing_trust(self, tmp_path):
        """The install mask must not make the YOLO suspend take standing Trust."""
        from kiro_crew.dashboard.server import (
            _clear_override_derived_trust,
            _suspend_override_derived_trust,
        )

        _install(None)
        state = _make_state(tmp_path)
        policies = {"dashboard:s1": "auto"}
        state.sessions.get_approval_policy = lambda key: policies.get(key, "")
        state.sessions.set_approval_policy = lambda key, value: policies.__setitem__(key, value)
        slot = state.get_or_create_slot("s1")
        slot._trust = True
        so_mod.safety_override().on_policy_suspend = lambda: _suspend_override_derived_trust(state)
        so_mod.safety_override().on_policy_revoked = lambda src: _clear_override_derived_trust(
            state, src
        )
        so_mod.safety_override().activate("dashboard")

        _install(None)
        assert slot._trust is True
        assert policies["dashboard:s1"] == "auto"

        # A ceiling that denies YOLO but permits Trust revokes the grant; the
        # standing Trust keeps its stored ``"auto"``.
        _install({"mode": "allow", "allow": ["normal", "trust_reads", "trust"]})
        assert so_mod.safety_override().has_grant() is False
        assert slot._trust is True
        assert policies["dashboard:s1"] == "auto"

    def test_the_messaging_trust_set_refuses_and_masks(self):
        from kiro_crew.messaging import session_trust as st

        _install(None)
        st.add_trusted_session("slack:C1:1")
        assert st.is_session_trusted("slack:C1:1") is True

        _install(OWNER_RULE)
        assert st.is_session_trusted("slack:C1:1") is False
        st.add_trusted_session("slack:C2:1")
        assert "slack:C2:1" not in st._trusted_sessions
        with pytest.raises(PermissionError):
            st.add_trusted_session("slack:C3:1", MagicMock(), strict=True)
        st.clear_trusted_sessions()


class TestOtherEntryPoints:
    def test_a_crewmate_default_lands_on_the_highest_allowed_mode(self):
        _install(OWNER_RULE)
        assert amp.clamp_session_mode("trust") == "trust_reads"
        _install({"mode": "allow", "allow": ["normal"]})
        assert amp.clamp_session_mode("trust") == "normal"
        assert amp.clamp_session_mode("trust_reads") == "normal"
        _install(None)
        assert amp.clamp_session_mode("trust") == "trust"

    def test_a_cron_jobs_auto_is_refused_and_not_honoured(self):
        from kiro_crew.cron_service.fields import (
            effective_job_approval_mode,
            refuse_policy_denied_auto,
        )

        _install(OWNER_RULE)
        with pytest.raises(ValueError, match="organization's policy"):
            refuse_policy_denied_auto("auto")
        refuse_policy_denied_auto("")
        assert effective_job_approval_mode("auto") == ""

        _install(ALL_THREE)
        assert effective_job_approval_mode("auto") == "auto"

    @pytest.mark.asyncio
    async def test_a_typed_trust_reply_approves_once_and_says_so(self):
        from kiro_crew.messaging import approval as ap

        ap.reset_for_tests()
        _install(OWNER_RULE)
        entry = ap.open_approval("whatsapp:1", "r1")
        assert ap.deliver_verdict("whatsapp:1", ap.TRUST) == ap.RECEIPT_APPROVED
        assert await entry.future == ap.APPROVE
        ap.reset_for_tests()

    def test_editing_a_saved_auto_job_still_saves(self):
        from kiro_crew.cron_service.fields import apply_job_update
        from kiro_crew.cron_service.model import CronJob

        _install(OWNER_RULE)
        saved = CronJob(id="j1", name="j", message="m", approval_mode="auto")
        apply_job_update(saved, {"approval_mode": "auto", "name": "renamed"}, None)
        assert saved.name == "renamed"
        fresh = CronJob(id="j2", name="j", message="m")
        with pytest.raises(ValueError, match="organization's policy"):
            apply_job_update(fresh, {"approval_mode": "auto"}, None)

    def test_a_spawns_auto_is_not_inherited_by_its_child(self, monkeypatch):
        from types import SimpleNamespace

        import kiro_crew.subagent as sa_facade
        from kiro_crew.subagent_manager.turn_session import resolve_parent_policy

        monkeypatch.setattr(sa_facade, "sel", lambda: MagicMock())
        manager = SimpleNamespace(
            _sessions=SimpleNamespace(get_approval_policy=lambda key: ""),
            _is_yolo=None,
            _global_approval_mode="auto",
            _ctx_builder=None,
        )
        coordinator = SimpleNamespace(_manager=manager)
        info = SimpleNamespace(approval_mode="auto", parent_session_key="", id="sub-1")

        _install(OWNER_RULE)
        assert resolve_parent_policy(coordinator, info) == ""
        _install(None)
        assert resolve_parent_policy(coordinator, info) == "auto"

    @pytest.mark.asyncio
    async def test_the_messaging_driver_falls_back_to_asking(self):
        from types import SimpleNamespace

        from kiro_crew.messaging.driver import APPROVAL_TRUST, APPROVAL_TRUST_READS, TurnDriver

        asked: list[object] = []

        async def _decider(event):
            asked.append(event)
            return False

        event = SimpleNamespace(tool_kind="read")
        _install(OWNER_RULE)
        trust = SimpleNamespace(approval_mode=APPROVAL_TRUST, decider=_decider)
        assert await TurnDriver._approve(trust, event) is False
        assert asked == [event], "a denied Trust asks the person instead"
        reads = SimpleNamespace(approval_mode=APPROVAL_TRUST_READS, decider=_decider)
        assert await TurnDriver._approve(reads, event) is True

        _install(None)
        assert await TurnDriver._approve(trust, event) is True


def test_the_trust_hooks_run_after_the_yolo_hooks():
    """The YOLO suspend reads stored policies the trust mask also answers."""
    from kiro_crew import safety_override_grant

    invalidate = list(ctx_mod._CEILING_INVALIDATE_HOOKS)
    install = list(ctx_mod._CEILING_INSTALL_HOOKS)
    assert invalidate.index(safety_override_grant._on_ceiling_invalidating) < invalidate.index(
        amp._on_ceiling_invalidating
    )
    assert install.index(so_mod._on_ceiling_installed) < install.index(amp._on_ceiling_installed)
