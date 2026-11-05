"""Whether the policy lets a session hold ``trust`` or ``trust_reads``. From memory.

The ``approval_modes`` scope (KGL ``permission_mode``) may deny ``trust`` and
``trust_reads``; only ``normal`` is the floor. A deny has to hold where a grant is
HONOURED, not only where it is set: a slot that was trusted before the policy landed,
a session whose stored ``approval_policy`` is ``"auto"``, a messaging session in the
Trust set. Those reads happen per tool call and on the 5s status push, so the answer
is resolved once per ceiling install and read from a module flag, exactly like the
``yolo`` verdict in ``safety_override_grant``.

Fail closed: before a verdict is resolved, while a new ceiling is being installed, and
when the policy cannot be read, both modes are denied (a person is asked instead).
With no policy (``oss``) the scope is ungoverned and both are permitted.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from kiro_crew.platform.context import (
    register_ceiling_install_hook,
    register_ceiling_invalidate_hook,
)

logger = logging.getLogger(__name__)

#: Approval modes a single session may hold. ``yolo`` is process-wide and has its
#: own pushed verdict; ``normal`` is the floor and never denied.
SESSION_TRUST_MODES = ("trust_reads", "trust")
#: Every approval-mode id Crew knows. Anything else is refused, not passed on.
KNOWN_APPROVAL_MODES = ("normal", "trust_reads", "trust", "yolo")

_denied: frozenset[str] = frozenset(SESSION_TRUST_MODES)
_resolved = False
#: Called with the denied modes after an install resolves a deny. Each one drops the
#: grants it holds, so a later policy that allows the mode again does not bring a
#: grant back that nobody re-made. The read paths mask on their own; these make the
#: mask and the state agree.
#: Keyed by owner, so a re-registration (a second ``start_dashboard`` in one
#: process) replaces the old callback instead of keeping its state alive.
_deny_hooks: dict[str, Callable[[frozenset[str]], None]] = {}


def register_deny_hook(owner: str, cb: Callable[[frozenset[str]], None]) -> None:
    """Call ``cb(denied)`` after each install that resolves a deny; one per *owner*."""
    _deny_hooks[owner] = cb


def _run_deny_hooks(denied: frozenset[str]) -> None:
    for cb in list(_deny_hooks.values()):
        try:
            cb(denied)
        except Exception:
            logger.warning("a trust-revoke hook raised; its grants are masked only", exc_info=True)


def _resolve() -> frozenset[str]:
    from kiro_crew.safety_override_permits import approval_mode_permitted

    denied = set()
    for mode in SESSION_TRUST_MODES:
        try:
            if not approval_mode_permitted(mode):
                denied.add(mode)
        except Exception:
            logger.warning(
                "could not resolve the approval_modes policy for %s; denying it",
                mode,
                exc_info=True,
            )
            denied.add(mode)
    return frozenset(denied)


def _on_ceiling_invalidating() -> None:
    """Deny both modes before a new ceiling becomes visible. Pure memory."""
    global _denied, _resolved
    _denied = frozenset(SESSION_TRUST_MODES)
    _resolved = True


def _on_ceiling_installed(ctx: object) -> None:
    global _denied, _resolved
    if ctx is None:
        reset_state()
        return
    _denied = _resolve()
    _resolved = True
    if _denied:
        _run_deny_hooks(_denied)


def _bootstrap() -> None:
    """First read with no install pushed yet (a late import, a CLI, a test)."""
    global _denied, _resolved
    try:
        from kiro_crew.platform.context import current_context

        current_context()
    except Exception:
        logger.debug("no installed ceiling for the trust verdict; denying", exc_info=True)
        _denied, _resolved = frozenset(SESSION_TRUST_MODES), True
        return
    if not _resolved:
        _denied, _resolved = _resolve(), True


def session_mode_permitted(mode: str) -> bool:
    """True when policy lets a session hold *mode*. ``normal`` always; unknown never."""
    if mode == "normal":
        return True
    if mode not in SESSION_TRUST_MODES:
        return False
    if not _resolved:
        _bootstrap()
    return mode not in _denied


def clamp_session_mode(mode: str) -> str:
    """*mode* if the policy permits it, else the highest permitted mode below it.

    For a default nobody chose this session (a crewmate's unset Profile opens in
    ``trust``): it lands on what the policy allows rather than being refused.
    """
    order = ("normal", *SESSION_TRUST_MODES)
    if mode not in order:
        return "normal"
    for candidate in reversed(order[: order.index(mode) + 1]):
        if session_mode_permitted(candidate):
            return candidate
    return "normal"


def stored_auto_permitted() -> bool:
    """Whether an ``approval_policy="auto"`` grant may still be honoured. From memory.

    ``"auto"`` is written by a session Trust (slot trust, a messaging Trust press, a
    crewmate Profile, a spawn's ``approval_mode``) or by a session-wide YOLO grant, and
    its readers (the spawn gate, a subagent's inherited policy, a messaging session's
    own approval) trust it as is. With ``trust`` denied it holds only while a
    session-wide YOLO grant is live; YOLO revocation already clears what it wrote.
    """
    if session_mode_permitted("trust"):
        return True
    from kiro_crew.safety_override import safety_override

    return bool(safety_override().is_active())


def disabled_session_modes() -> list[str]:
    """The session modes the policy forbids, in picker order."""
    return [m for m in SESSION_TRUST_MODES if not session_mode_permitted(m)]


def reset_state() -> None:
    """Forget the verdict so the next read resolves again. Tests and ``reset_context``."""
    global _denied, _resolved
    _denied, _resolved = frozenset(SESSION_TRUST_MODES), False


def register_hooks() -> None:
    """Register the mask and the push, AFTER the YOLO verdict's pair.

    Called by :mod:`kiro_crew.safety_override` right after it registers its own.
    The order is load-bearing: the YOLO suspend (``on_policy_suspend``) reads each
    slot's stored ``approval_policy`` through ``get_approval_policy``, which this
    module's mask also answers. Masked first, the suspend would read ``""``, skip a
    YOLO-inherited ``"auto"``, and this push would unmask it again before the YOLO
    verdict resolved. Registration is idempotent per callable.
    """
    register_ceiling_invalidate_hook(_on_ceiling_invalidating)
    register_ceiling_install_hook(_on_ceiling_installed)


# Importing the facade registers both pairs in order, whichever module loads first.
from kiro_crew import safety_override as _safety_override  # noqa: E402,F401
