"""Move a local-only execution grant to the repository a registry update installs.

An app installed from a local folder holds a LOCAL grant (``agent.apps_trusted``
plus ``agent.apps_trusted_local``). That grant never covers repository code, so a
registry update of the app is refused with ``app_trust_local_only``, and the
consent dialog cannot help: the grant endpoint binds to the INSTALLED source,
which is the folder.

``POST /api/apps/{name}/update`` with ``source: "registry:<name>"`` and the
refusal's ``trustRepository`` as consent proof performs the switch instead. This
module holds its three steps, which the route runs under the app's lifecycle lock
through :func:`switch_to_repository` (steps 1 and 2) and :func:`revert_switch`
(step 3), each audited under the grant route's SEL operation:

1. :func:`check_switch` verifies the proof against the row
   ``install_from_registry`` will act on, and that the grant is exactly the local
   one the refusal named. It writes nothing.
2. :func:`bind_repository` records the repository-bound grant the consent dialog
   records for a fresh install, removing the local marker, before any repository
   bytes are fetched. It checks the local grant again inside the locked write.
3. :func:`restore_local_grant` puts the local grant back when the update fails,
   before the route restores the old app, so the old backend starts admitted.

The grant shapes match ``security_trusted_apps.api_trusted_app_grant`` and
``default_installs._grant`` / ``_revoke``, and every write goes through the one
trust writer (``agent_config_writer.mutate_trust_settings``).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Refused when the proof does not match the row the update would install.
MISMATCH_CODE = "app_trust_repository_mismatch"
#: Refused when the app's grant is not the local-only grant this switch replaces.
NOT_APPLICABLE_CODE = "app_trust_switch_not_applicable"
_NOT_APPLICABLE_ERROR = (
    "this app's trust is not the local-only grant that trusting its repository "
    "replaces; refresh the app page and update it again"
)


@dataclass(frozen=True)
class SwitchRefusal:
    """Why the switch cannot run. Nothing was written."""

    status: int
    code: str
    error: str


def _grant_is_local_only(name: str) -> bool:
    """Whether *name* holds exactly a local-source grant over local installed code."""
    from kiro_crew.apps.execution import (
        builtin_app_names,
        trusted_app_names,
        trusted_app_repository,
        trusted_local_app_names,
    )
    from kiro_crew.apps.registry import _trust_grant_would_stay_local

    if name in builtin_app_names():
        return False
    if name not in trusted_app_names() or name not in trusted_local_app_names():
        return False
    if trusted_app_repository(name):
        return False
    return _trust_grant_would_stay_local(name)


def _install_repository(name: str) -> str:
    """The normalized clone target ``install_from_registry(name)`` resolves, or ``""``."""
    from kiro_crew.apps import registry as reg

    entry, refusal = reg._resolve_install_entry(name)
    if refusal or not entry:
        return ""
    url = reg._entry_git_url(entry)
    if not url or reg._git_target_is_unsupported(url):
        return ""
    return reg._normalize_git_target(url)


def _check_switch_sync(name: str, consent: str) -> str | SwitchRefusal:
    from kiro_crew.apps import registry as reg

    if reg._git_target_is_unsupported(consent):
        return SwitchRefusal(
            400,
            "invalid_repository_consent",
            "repository consent proof contains an unsupported query or fragment "
            "or an ambiguous Git transport identity",
        )
    if not _grant_is_local_only(name):
        return SwitchRefusal(409, NOT_APPLICABLE_CODE, _NOT_APPLICABLE_ERROR)
    repository = _install_repository(name)
    # The proof binds the click to what the dialog showed. The registry row can
    # change between the refusal and the click; a stale proof is refused the way
    # the grant endpoint refuses one, and nothing is written. The coordinates are
    # comparison inputs only, never response or audit text.
    if not repository or repository != reg._normalize_git_target(consent):
        return SwitchRefusal(
            409,
            MISMATCH_CODE,
            "app repository changed since the consent dialog was opened; refresh and review it",
        )
    return repository


async def check_switch(name: str, consent: str) -> str | SwitchRefusal:
    """Return the repository to bind *name* to, or why the switch is refused.

    Blocking reads (config, installed metadata, the catalog) run off the loop.
    """
    return await asyncio.to_thread(_check_switch_sync, name, consent)


class GrantChanged(Exception):
    """The local-only grant was gone by the time the config lock was held."""


async def bind_repository(name: str, repository: str) -> None:
    """Record *name*'s grant as bound to *repository*, dropping its local marker.

    The grant is checked again inside the locked write, against the file itself:
    :func:`check_switch` read it before the lock, and another process (the CLI,
    an uninstall) can revoke or rebind it in between. Moving a grant that is no
    longer there would put back a grant the operator removed, so that raises
    :class:`GrantChanged` and writes nothing. Also raises the trust writer's
    ``ConfigCorruptError`` / ``TrustSettingOverlayOwned``.
    """
    from kiro_crew.apps.agent_config_writer import mutate_trust_settings
    from kiro_crew.dashboard.handlers.security_trusted_apps import (
        _trusted_list_raw,
        _trusted_local_list_raw,
        _trusted_repositories_raw,
    )

    def _mutate(agent_raw: dict) -> None:
        local = _trusted_local_list_raw(agent_raw)
        raw_bindings = agent_raw.get("apps_trusted_repositories")
        bound = isinstance(raw_bindings, dict) and bool(raw_bindings.get(name))
        if name not in _trusted_list_raw(agent_raw) or name not in local or bound:
            raise GrantChanged(name)
        repositories = _trusted_repositories_raw(agent_raw)
        repositories[name] = repository
        agent_raw["apps_trusted_repositories"] = repositories
        agent_raw["apps_trusted_local"] = [a for a in local if a != name]

    await mutate_trust_settings(_mutate)


async def restore_local_grant(name: str, repository: str) -> bool:
    """Undo :func:`bind_repository` after a failed update. Returns whether it did.

    Only a binding that still names *repository* is undone: a grant the operator
    changed while the update ran is theirs and is left as it is. The name stays in
    ``apps_trusted`` (it was there before the switch), and a marker without that
    name is inert, so an uninstall that ran meanwhile is not undone either.
    A failed write is logged and leaves the repository grant, which does not
    cover the restored local code: the app is then refused, never widened.
    """
    from kiro_crew.apps.agent_config_writer import mutate_trust_settings
    from kiro_crew.dashboard.handlers.security_trusted_apps import (
        _trusted_local_list_raw,
        _trusted_repositories_raw,
    )

    restored = False

    def _mutate(agent_raw: dict) -> None:
        nonlocal restored
        repositories = _trusted_repositories_raw(agent_raw)
        if repositories.get(name) != repository:
            return
        repositories.pop(name, None)
        agent_raw["apps_trusted_repositories"] = repositories
        local = _trusted_local_list_raw(agent_raw)
        if name not in local:
            local.append(name)
        agent_raw["apps_trusted_local"] = local
        restored = True

    try:
        await mutate_trust_settings(_mutate)
    except Exception:  # noqa: BLE001 - the caller must still restore the old app
        # Any failure, a full disk included: the route restores the old app
        # after this returns, and an exception here would skip that and leave
        # it stopped. The repository grant that stays fails closed.
        logger.warning(
            "could not restore the local grant after a failed update of %r", name, exc_info=True
        )
        return False
    return restored


#: Every grant write is audited under the trusted-apps grant route's operation.
_GRANT_OPERATION = "security.trusted_apps.grant"


def _audit(caller: str, outcome: str, resources: str) -> None:
    """Best-effort SEL row; losing it never changes the outcome."""
    try:
        from kiro_crew.sel import sel

        sel().log_api_access(
            caller=caller,
            operation=_GRANT_OPERATION,
            outcome=outcome,
            source="dashboard",
            resources=resources,
        )
    except Exception:  # noqa: BLE001 - an audit failure must not mask the result
        logger.warning("SEL logging failed for %s", _GRANT_OPERATION, exc_info=True)


async def switch_to_repository(name: str, consent: str, *, caller: str) -> str | SwitchRefusal:
    """Check the consent and bind *name*'s grant to the repository. Steps 1 and 2.

    Returns the bound repository, or the refusal to send; a refusal wrote nothing.
    The caller holds ``app_lifecycle_lock(name)``.
    """
    from kiro_crew.dashboard.handlers.security_audit import ConfigCorruptError
    from kiro_crew.dashboard.handlers.security_trusted_apps import TrustSettingOverlayOwned

    checked = await check_switch(name, consent)
    if isinstance(checked, SwitchRefusal):
        _audit(caller, "denied", f"{name}={checked.code}")
        return checked
    try:
        await bind_repository(name, checked)
    except GrantChanged:
        _audit(caller, "denied", f"{name}={NOT_APPLICABLE_CODE}")
        return SwitchRefusal(409, NOT_APPLICABLE_CODE, _NOT_APPLICABLE_ERROR)
    except ConfigCorruptError as exc:
        _audit(caller, "denied", f"{name}=config_corrupt")
        return SwitchRefusal(409, "config_corrupt", str(exc))
    except TrustSettingOverlayOwned as exc:
        _audit(caller, "denied", f"{name}=overlay_owned")
        return SwitchRefusal(409, "trust_setting_overlay_owned", str(exc))
    _audit(caller, "ok", f"{name}=registry_update")
    return checked


async def revert_switch(name: str, repository: str, *, caller: str) -> None:
    """Step 3 after a failed update: put the local grant back, and audit it."""
    restored = await restore_local_grant(name, repository)
    _audit(caller, "reverted" if restored else "revert_skipped", f"{name}=registry_update")


async def switch_cancel_safe(name: str, consent: str, *, caller: str) -> str | SwitchRefusal:
    """:func:`switch_to_repository`, undone if the request is cancelled mid-write.

    The grant write runs in a worker thread, so a cancelled request can return
    before the write lands while the write still lands afterwards, leaving the
    repository grant over local code with nobody to revert it. The switch runs
    as its own task: on cancellation this waits for it to settle and reverts a
    grant it wrote, then lets the cancellation through.
    """
    task = asyncio.ensure_future(switch_to_repository(name, consent, caller=caller))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:

        async def _undo() -> None:
            try:
                result = await task
                if not isinstance(result, SwitchRefusal):
                    await revert_switch(name, result, caller=caller)
            except Exception:  # noqa: BLE001 - the cancellation is what propagates
                logger.warning(
                    "could not settle a cancelled trust switch for %r", name, exc_info=True
                )

        await asyncio.shield(_undo())
        raise
