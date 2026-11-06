"""An agent spec's own hook reads kiro-cli's stdin field names when Crew runs it.

Crew runs on kiro-cli's V2 engine, which tells a spec hook the event in camelCase,
the tool by its full name (``shell``, ``@server/tool``) with its ``tool_tags``, and
a Stop's text as ``assistant_response``. On KAS Crew runs the same hook, so it must
say the same; a ``~/.kiro/hooks`` file hook was written for KAS and keeps KAS's names.
"""

from __future__ import annotations

import asyncio
import dataclasses

import kiro_crew.hooks as hooks_mod
from kiro_crew.agent_sdk import spec_hooks
from kiro_crew.hook_runtime.hook_dispatch import pre_tool_match_names
from kiro_crew.hooks import (
    HOOK_EVENT_PRE_TOOL_USE,
    HOOK_EVENT_STOP,
    ScriptHook,
    ScriptHookResult,
    ScriptHookStore,
)


def _spec_hook(event_key: str, origin: str = "user") -> ScriptHook:
    (hook,) = spec_hooks.spec_script_hooks(
        "a1", {"hooks": {event_key: [{"matcher": "*", "command": "guard.sh"}]}}
    )
    if event_key == "stop":
        hook = dataclasses.replace(hook, matcher="")
    return dataclasses.replace(hook, spec_origin=origin)


def _record(monkeypatch) -> dict:
    seen: dict = {}

    async def fake_run(hook, context="", hook_event=None, **_kw):
        seen[hook.id] = hook_event
        return ScriptHookResult(hook_id=hook.id, hook_name=hook.name, event=hook.event)

    monkeypatch.setattr(hooks_mod, "run_script_hook", fake_run)
    return seen


def _fire_pre_tool(store, hooks, **names):
    _all, spec = pre_tool_match_names("touch /tmp/x", **names)
    return asyncio.run(
        store.fire(
            HOOK_EVENT_PRE_TOOL_USE,
            tool_name="touch /tmp/x",
            extra_hooks=hooks,
            extra_hooks_tool_names=spec,
        )
    )


def test_a_spec_hook_on_a_kas_shell_call_reads_kiro_cli_names(tmp_path, monkeypatch):
    seen = _record(monkeypatch)
    store = ScriptHookStore(tmp_path)
    page = ScriptHook(id="page", event=HOOK_EVENT_PRE_TOOL_USE, matcher="*", command="c")
    store._hooks[page.id] = page
    spec = _spec_hook("preToolUse")
    _fire_pre_tool(store, [spec], harness_tool_id="run_command")
    assert seen[spec.id]["hook_event_name"] == "preToolUse"
    assert seen[spec.id]["tool_name"] == "shell"
    assert seen[spec.id]["tool_tags"] == ["shell", "@builtin"]
    # A Hooks-page hook in the same fire keeps Crew's own names.
    assert seen["page"]["hook_event_name"] == HOOK_EVENT_PRE_TOOL_USE
    assert seen["page"]["tool_name"] == "touch /tmp/x"
    assert "tool_tags" not in seen["page"]


def test_a_spec_hook_on_an_mcp_call_is_told_the_server_tool_name(tmp_path, monkeypatch):
    seen = _record(monkeypatch)
    spec = _spec_hook("preToolUse", origin="project")
    _fire_pre_tool(
        ScriptHookStore(tmp_path),
        [spec],
        tool_identity="create_issue",
        mcp_server="github",
        harness_tool_id="github/create_issue",
    )
    assert seen[spec.id]["tool_name"] == "@github/create_issue"
    assert seen[spec.id]["tool_tags"] == ["@github/create_issue", "@github", "@mcp"]


def test_a_spec_stop_hook_reads_assistant_response(tmp_path, monkeypatch):
    seen = _record(monkeypatch)
    spec = _spec_hook("stop")
    asyncio.run(ScriptHookStore(tmp_path).fire(HOOK_EVENT_STOP, "done [x]", extra_hooks=[spec]))
    assert seen[spec.id]["hook_event_name"] == "stop"
    assert seen[spec.id]["assistant_response"] == "done [x]"
    # Crew's own key stays for a script that already reads it.
    assert seen[spec.id]["assistant_text"] == "done [x]"


def test_a_user_hook_file_hook_keeps_kas_names(tmp_path, monkeypatch):
    seen = _record(monkeypatch)
    spec = _spec_hook("preToolUse", origin="user_file")
    _fire_pre_tool(ScriptHookStore(tmp_path), [spec], harness_tool_id="run_command")
    assert seen[spec.id]["hook_event_name"] == HOOK_EVENT_PRE_TOOL_USE
    assert seen[spec.id]["tool_name"] == "execute_bash"
    assert "tool_tags" not in seen[spec.id]


def test_governance_is_asked_about_the_store_event_for_a_camelcase_payload(monkeypatch):
    asked: list = []

    def refuse(_sk, event):
        asked.append(event)
        return "refused"

    monkeypatch.setattr(hooks_mod, "_script_hooks_capability_denied", lambda _sk: None)
    monkeypatch.setattr("kiro_crew.hook_event_governance.event_refusal", refuse)
    hook = _spec_hook("preToolUse")
    result = asyncio.run(
        hooks_mod.run_script_hook(hook, "", {"hook_event_name": "preToolUse", "cwd": "/"})
    )
    assert asked == [HOOK_EVENT_PRE_TOOL_USE]
    assert result.refused


def test_opencode_apply_patch_is_told_write_not_its_mcp_split():
    # The fused title splits as ``@apply/patch`` too, but the call is opencode's own
    # write tool, so a guard on ``tool_name == "write"`` must still see it.
    from kiro_crew.acp.harness_tool_names import qualified_harness_tool_id

    tool_id = qualified_harness_tool_id("opencode", "apply_patch", ("apply",))
    _all, spec = pre_tool_match_names("apply_patch", harness_tool_id=tool_id)
    payload = spec_hooks.kiro_cli_hook_payload({}, HOOK_EVENT_PRE_TOOL_USE, spec)
    assert payload["tool_name"] == "write"
    assert payload["tool_tags"] == ["write", "@builtin"]
