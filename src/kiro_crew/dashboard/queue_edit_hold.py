"""Hold a queued message while its card's editor is open.

Without the hold, a turn that ends while the person is still typing in a
queued card's editor drains the entry as it was, and the edit lands on an
entry that is already gone: the agent gets the old words. Opening the editor
marks the entry; the drain treats a marked entry like the sub-agent hold
treats user messages (``chat_turn.queue_drain``): user entries wait, system
injections still flow. Saving the edit (the PATCH), closing the editor, or the
deadline releases it, and a release on an idle slot starts the drain the hold
deferred.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aiohttp import web

from kiro_crew.dashboard.handlers._shared import read_bounded_json
from kiro_crew.dashboard.slot_ownership import deny_app_slot_access, slot_not_found

logger = logging.getLogger(__name__)

#: In-memory only: underscore keys are not in the durable queue keys, so a
#: restart never restores a hold nobody is still editing under.
EDIT_HOLD_KEY = "_edit_hold_until"
# shortcut: a fixed deadline with no renewal, so an edit left open longer than
# this drains the entry as it was; add a client keep-alive if that shows up.
EDIT_HOLD_SECONDS = 300.0
# The hold's clock; tests replace it.
_clock = time.monotonic
#: A released hold drains only after this pause. Moving straight from one
#: card's editor to another's releases the first (its blur) before the second's
#: hold leaves the browser (its click); a drain started at once would send the
#: second card's old words.
RELEASE_GRACE_SECONDS = 2.0
# The grace timer's clock; tests replace it with a gate they release.
_grace_sleep = asyncio.sleep


def edit_held(item: dict[str, Any]) -> bool:
    """Whether *item*'s editor is open and its deadline has not passed."""
    until = item.get(EDIT_HOLD_KEY)
    return isinstance(until, float) and until > _clock()


def queue_edit_held(slot: Any) -> bool:
    """Whether any entry of *slot*'s queue is held for an open editor."""
    return any(edit_held(item) for item in getattr(slot, "_queue", ()))


def release_edit_hold(item: dict[str, Any]) -> bool:
    """Drop *item*'s hold; True when it had one."""
    return item.pop(EDIT_HOLD_KEY, None) is not None


async def drain_if_idle(state: Any, slot: Any) -> None:
    """Start the drain a hold deferred, if the slot is idle and nothing is held."""
    if not slot._queue or queue_edit_held(slot) or slot.executor == "remote":
        return
    # circular import: chat_runner imports the dashboard package at module level.
    from kiro_crew.dashboard.chat_runner import _start_next_queued_turn, idle_slot_admits_drain

    started = False
    try:
        async with slot._lock:
            # The same admission every idle-slot drain reads, sign-in hold
            # included, so a release never runs a turn into a signed-out CLI.
            if not slot._queue or not idle_slot_admits_drain(state, slot) or queue_edit_held(slot):
                return
            slot._turn_admission_reserved = True
            try:
                started = await _start_next_queued_turn(state, slot)
            finally:
                slot._turn_admission_reserved = False
    except Exception:
        logger.warning("Could not drain slot %s after a queue edit hold", slot.key, exc_info=True)
        return
    if started:
        state.push_slots_update()


async def _drain_after_grace(state: Any, slot: Any) -> None:
    await _grace_sleep(RELEASE_GRACE_SECONDS)
    await drain_if_idle(state, slot)


def _spawn_drain(state: Any, slot: Any, *, grace: bool = False) -> None:
    work = _drain_after_grace(state, slot) if grace else drain_if_idle(state, slot)
    task = asyncio.get_running_loop().create_task(work)
    state._background_tasks.add(task)
    task.add_done_callback(state._background_tasks.discard)


def _arm_expiry(state: Any, slot: Any, item: dict[str, Any], until: float) -> None:
    """Drain at the deadline, unless the hold was released or renewed first."""

    def _expire() -> None:
        if item.get(EDIT_HOLD_KEY) != until:
            return
        item.pop(EDIT_HOLD_KEY, None)
        _spawn_drain(state, slot)

    asyncio.get_running_loop().call_later(EDIT_HOLD_SECONDS, _expire)


def release_and_drain(state: Any, slot: Any, item: dict[str, Any] | None) -> None:
    """Release *item*'s hold and start the deferred drain if it had one."""
    if item is not None and release_edit_hold(item):
        _spawn_drain(state, slot, grace=True)


async def api_chat_slot_queue_editing(request: web.Request) -> web.Response:
    """POST /api/chat/slots/{slot}/queue/{queue_id}/editing — open or close a hold.

    ``{"editing": true}`` holds the entry while its card's editor is open;
    ``{"editing": false}`` releases it without an edit. Saving through the
    PATCH releases it too. 404 when the entry has already left the queue.
    """
    state = request.app["state"]
    name = request.match_info["slot"]
    queue_id = request.match_info["queue_id"]
    slot = state._slots.get(name)
    if not slot:
        return slot_not_found()
    denied = deny_app_slot_access(request.get("app", ""), slot, name, "slot_queue_edit")
    if denied is not None:
        return denied
    body, body_err = await read_bounded_json(request)
    if body_err is not None:
        return body_err
    assert body is not None  # read_bounded_json returns (dict, None) on success
    editing = body.get("editing")
    if not isinstance(editing, bool):
        return web.json_response(
            {"error": "editing must be a boolean", "code": "invalid_field_type"}, status=400
        )
    item = next((i for i in slot._queue if i.get("id") == queue_id), None)
    if item is None:
        return web.json_response(
            {"error": "queue item not found", "code": "queue_entry_not_found"}, status=404
        )
    if editing:
        until = _clock() + EDIT_HOLD_SECONDS
        item[EDIT_HOLD_KEY] = until
        _arm_expiry(state, slot, item, until)
    else:
        release_and_drain(state, slot, item)
    return web.json_response({"ok": True})
