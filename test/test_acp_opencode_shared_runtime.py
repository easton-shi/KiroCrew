"""OpenCode on the shared ``AcpRuntime``: the seams that no session reaches yet.

The class is registered in ``_HARNESSES`` before it is routed
(``ACP_BACKENDS_ACP_RUNTIME`` does not name it), so these tests drive the seams
directly: the process-level seed, the process-scoped spawn plan, the
per-session read-back and the routing dispatch that calls it. No real
``opencode`` binary is started; the child runs are the per-session launch's own
functions, which their tests already cover.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kiro_crew import acp_tool_gate
from kiro_crew.acp import launch as launch_mod
from kiro_crew.acp import session_handle_permissions
from kiro_crew.acp.harness import _HARNESSES
from kiro_crew.acp.harness import opencode as opencode_mod
from kiro_crew.acp.harness.base import SpawnContext
from kiro_crew.acp.harness.opencode_harness import OpencodeHarness
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
from kiro_crew.acp_backends import ACP_BACKEND_OPENCODE, ACP_BACKENDS_ACP_RUNTIME

_ENV_CONFIG = "OPENCODE_CONFIG_CONTENT"


def _ctx(tmp_path) -> SpawnContext:
    return SpawnContext(
        agent="",
        work_dir=tmp_path,
        model=None,
        environ={},
        home=tmp_path,
        sandbox_mode="strict",
    )


class TestDormantUntilRoutingLands:
    def test_the_harness_is_importable(self) -> None:
        assert OpencodeHarness.backend == ACP_BACKEND_OPENCODE

    def test_registration_and_routing_land_together(self) -> None:
        """The contract tests pin ``_HARNESSES`` to ``ACP_BACKENDS_ACP_RUNTIME``,
        so a harness is not registered before a session may be routed to it. The
        live ``session/close`` capture that earns ``SESSION_EVICTION`` is still
        missing, so neither half is on today."""
        assert ACP_BACKEND_OPENCODE not in _HARNESSES
        assert ACP_BACKEND_OPENCODE not in ACP_BACKENDS_ACP_RUNTIME


class TestTheProcessSeed:
    def test_apply_spawn_env_writes_an_asking_seed_and_captures_it(self) -> None:
        harness = OpencodeHarness()
        env = {_ENV_CONFIG: json.dumps({"model": "ollama/qwen3:8b"})}

        harness.apply_spawn_env(env)

        seed = json.loads(env[_ENV_CONFIG])
        assert seed["permission"] == "ask", "Crew's setting must win the merge"
        assert seed["model"] == "ollama/qwen3:8b", "the operator's keys are preserved"
        assert harness.process_config_content() == env[_ENV_CONFIG]

    def test_the_process_seed_carries_no_per_tool_deny(self) -> None:
        """Per-tool denies are per session and cannot be scoped to one exec."""
        harness = OpencodeHarness()
        env: dict[str, str] = {}

        harness.apply_spawn_env(env)

        assert "deny" not in env[_ENV_CONFIG]

    def test_process_config_content_rebuilds_the_same_document_without_a_spawn(self) -> None:
        harness = OpencodeHarness()
        assert json.loads(harness.process_config_content())["permission"] == "ask"


class TestTheSpawnPlan:
    @pytest.mark.asyncio
    async def test_resolve_spawn_returns_the_process_scoped_half(self, tmp_path, monkeypatch):
        async def _launch(_backend: str):
            return "/opt/opencode", ["/opt/opencode", "acp"], "opencode", "opencode"

        async def _preflight(_fn, _backend: str, _sandbox_mode: str):
            return ("/home/user/.aws",)

        monkeypatch.setattr(launch_mod, "resolve_self_served_launch", _launch)
        monkeypatch.setattr(launch_mod, "_run_preflight_bounded", _preflight)
        monkeypatch.setattr(acp_tool_gate, "adapter_expose_files", lambda _b, _h: ())

        plan = await OpencodeHarness().resolve_spawn(_ctx(tmp_path))

        assert plan.argv == ["/opt/opencode", "acp"]
        assert plan.spawn_label == "opencode"
        assert plan.stderr_label == "opencode"
        assert plan.extra_hidden_dirs == ("/home/user/.aws",)


class TestThePerSessionReadBack:
    @pytest.mark.asyncio
    async def test_a_per_tool_deny_refuses_the_session(self) -> None:
        handle = SimpleNamespace(harness_deny_rules=("bash",))

        with pytest.raises(AcpToolGateUnroutable, match="per-tool deny"):
            await OpencodeHarness().verify_session_routing(handle)

    @pytest.mark.asyncio
    async def test_a_verified_session_records_its_readback_state(self, monkeypatch):
        async def _version(_view, _bin, **_kw):
            return "", ""

        def _verify(view, _backend, _argv, _config):
            view._opencode_config_mcp_servers = ("docs.server",)
            view._opencode_denies_unenforced = frozenset({"bash"})
            view._opencode_native_mounts = frozenset({"docs.server"})
            return "", ""

        async def _launch(_backend: str):
            return "/opt/opencode", ["/opt/opencode", "acp"], "opencode", "opencode"

        async def _wrap(argv, **_kw):
            return argv, None

        monkeypatch.setattr(opencode_mod, "_opencode_version_refusal", _version)
        monkeypatch.setattr(opencode_mod, "_verify_opencode_routing", _verify)
        monkeypatch.setattr(launch_mod, "resolve_self_served_launch", _launch)
        monkeypatch.setattr("kiro_crew.sandbox.wrap_argv_async", _wrap)
        monkeypatch.setattr(acp_tool_gate, "adapter_expose_files", lambda _b, _h: ())

        runtime = SimpleNamespace(_extra_env={}, sandbox_mode="strict", sandbox_hidden_dirs=())
        handle = SimpleNamespace(
            _runtime=runtime,
            _bound_cwd="/work",
            harness_deny_rules=(),
            session_mcp_unhonoured=frozenset(),
        )

        await OpencodeHarness().verify_session_routing(handle)

        assert handle.opencode_config_mcp_servers == ("docs.server",)
        assert handle.opencode_denies_unenforced == frozenset({"bash"})
        assert handle.opencode_native_mounts == frozenset({"docs.server"})

    @pytest.mark.asyncio
    async def test_a_routing_issue_refuses_the_session(self, monkeypatch):
        async def _version(_view, _bin, **_kw):
            return "", ""

        def _verify(_view, _backend, _argv, _config):
            return "the resolved setting is not ask", "remove the source that outranks it"

        async def _launch(_backend: str):
            return "/opt/opencode", ["/opt/opencode", "acp"], "opencode", "opencode"

        async def _wrap(argv, **_kw):
            return argv, None

        monkeypatch.setattr(opencode_mod, "_opencode_version_refusal", _version)
        monkeypatch.setattr(opencode_mod, "_verify_opencode_routing", _verify)
        monkeypatch.setattr(launch_mod, "resolve_self_served_launch", _launch)
        monkeypatch.setattr("kiro_crew.sandbox.wrap_argv_async", _wrap)
        monkeypatch.setattr(acp_tool_gate, "adapter_expose_files", lambda _b, _h: ())

        runtime = SimpleNamespace(_extra_env={}, sandbox_mode="strict", sandbox_hidden_dirs=())
        handle = SimpleNamespace(
            _runtime=runtime,
            _bound_cwd="/work",
            harness_deny_rules=(),
            session_mcp_unhonoured=frozenset(),
        )

        with pytest.raises(AcpToolGateUnroutable):
            await OpencodeHarness().verify_session_routing(handle)


class TestTheRoutingDispatch:
    @pytest.mark.asyncio
    async def test_verified_seeded_settings_calls_the_harness_seam(self) -> None:
        class _Harness:
            def __init__(self) -> None:
                self.handles: list[Any] = []

            async def verify_session_routing(self, handle: Any) -> None:
                self.handles.append(handle)

        harness = _Harness()
        handle = SimpleNamespace(
            _runtime=SimpleNamespace(acp_backend=ACP_BACKEND_OPENCODE, _harness=harness)
        )

        await session_handle_permissions.apply_session_permission_routing(handle)

        assert harness.handles == [handle]
