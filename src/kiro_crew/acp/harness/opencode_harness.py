"""OpenCode on the shared ``AcpRuntime``: one ``opencode acp`` process, N sessions.

This is the SHARED-RUNTIME harness for OpenCode, the counterpart to the
per-session :class:`~kiro_crew.acp.harness.opencode.OpencodeLaunch`. Where that
one is a :class:`ProcessAdapter` that ``AcpClient`` starts once per session, this
is a :class:`~kiro_crew.acp.harness._common.MembershipHarness` that ``AcpRuntime``
drives for many sessions on one process, the way :class:`CodexHarness` does.

Why OpenCode can run here at all: ``opencode acp`` serves ACP from its own binary
and its server multiplexes. Measured on opencode 1.19.5, two ``session/new`` calls
on one stdio connection return two distinct, independent session ids, and its
``initialize`` advertises ``sessionCapabilities: {close, fork, list, resume}`` with
``loadSession: true``. So the limit that kept it on the per-session path was Crew's
``AcpRuntime`` being kiro-shaped, not the harness -- the same situation the
``ACP_BACKENDS_SESSION_SHARING`` comment records for deepseek, resolved for OpenCode
by giving it a harness that fills the runtime's seams.

WHAT IS STILL OPEN (do not read this module as finished):

* The capability-set membership (``ACP_BACKENDS_ACP_RUNTIME`` / ``SESSION_EVICTION``
  / ``SESSION_SHARING``) is NOT flipped yet, and the class is not registered in
  ``_HARNESSES`` either: the contract tests pin registration to routing, so both
  land together when a live ``session/close`` capture exists. Until then the class
  is importable on its own and its seams are tested directly.
* The per-session read-back runs AFTER ``session/new``, because the runtime has no
  per-session spawn to run it in. A refusal therefore terminates a session that
  already exists in the shared process rather than refusing before the process
  starts; it still runs before the first prompt, and the create/load error path
  already calls ``terminate_session``.
* The permission seed is UNIFORM: ``OPENCODE_CONFIG_CONTENT`` is read once at exec,
  so every session on this process verifies the same document. A session whose
  projection asks for per-tool deny rules cannot be enforced on a shared process
  and is refused by :meth:`OpencodeHarness.verify_session_routing` rather than run
  with a switched-off tool live.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from kiro_crew.acp.harness._common import MembershipHarness
from kiro_crew.acp.harness.base import (
    NotificationAliases,
    SessionExtras,
    SpawnContext,
    SpawnPlan,
    TeardownPolicy,
)
from kiro_crew.acp.types import (
    ACP_BACKEND_OPENCODE,
    ACP_CLIENT_CAPABILITIES,
    METHOD_SESSION_CLOSE,
    METHOD_SESSION_UPDATE,
)

__all__ = ["PROTOCOL_VERSION_OPENCODE", "OpencodeHarness"]

logger = logging.getLogger(__name__)


class _HandleReadbackView:
    """A read-only view of a shared-runtime handle, shaped like the session the
    opencode read-back functions were written against.

    The read-back and the version probe take the few things they read off a session
    -- the work dir, the env overlays, the scrub, the projected deny inputs -- and write
    their findings back as ``_opencode_*`` attributes. This supplies exactly those,
    from the live handle and its runtime, and collects the findings so the caller
    copies them onto the handle. It exists so the audited child-spawn sites stay in
    one place rather than being duplicated for the shared path.
    """

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._extra_env: dict[str, str] = dict(handle._runtime._extra_env or {})
        self._spawn_work_dir: str = handle._bound_cwd
        self._session_harness_deny_rules: tuple[str, ...] = tuple(handle.harness_deny_rules)
        self._session_mcp_unhonoured: frozenset[str] = frozenset(handle.session_mcp_unhonoured)
        self._opencode_config_mcp_servers: tuple[str, ...] = ()
        self._opencode_denies_unenforced: frozenset[str] = frozenset()
        self._opencode_native_mounts: frozenset[str] = frozenset()

    def _agent_env_for_spawn(self) -> dict[str, str]:
        from kiro_crew.acp.agent_env import agent_env_overlay

        return agent_env_overlay()

    def _scrub_observed(self, value: object) -> object:
        from kiro_crew.acp.client_permissions import _scrub_observed

        return _scrub_observed(value)


#: OpenCode answers ``initialize`` with an integer ``protocolVersion`` of 1 (the
#: SPEC dialect), measured on opencode 1.19.5. Its OWN literal even though the int
#: matches codex's and claude's today, for the harness-parity H10 reason: a future
#: divergence should be a one-line edit here, not a silent downgrade of whichever
#: harness moved first. The per-session launch reads the same value from the
#: ``ACP_BACKEND_LAUNCH`` row; both are 1, and they are kept as separate reads so
#: the client and runtime arms can never drift on the wire version.
PROTOCOL_VERSION_OPENCODE = 1


class OpencodeHarness(MembershipHarness):
    """OpenCode on the shared runtime: a self-served binary, N sessions per process."""

    backend = ACP_BACKEND_OPENCODE

    #: The exact ``OPENCODE_CONFIG_CONTENT`` this process was spawned with, captured
    #: by :meth:`apply_spawn_env`. The per-session read-back verifies this value, not
    #: a rebuild, so the spawn and the verification cannot seed different documents.
    #: Empty until a spawn runs one; :meth:`process_config_content` then rebuilds it.
    _config_content: str = ""

    # ── Seam 1: spawn ──
    #
    # OpenCode's asking posture has two halves, and only one is process-scoped:
    #   (1) the self-served binary + argv and the credential-mask preflight --
    #       ``ctx`` alone answers these, so they run here at spawn;
    #   (2) the version gate and the permission read-back -- coupled to a session's
    #       env/workdir, so they run per session on ``verify_session_routing``.
    # The permission seed itself is process-level (``OPENCODE_CONFIG_CONTENT`` is
    # read once at exec), so ``apply_spawn_env`` writes one uniform seed here; a
    # session that needs per-tool deny rules is refused there rather than seeded
    # differently.

    async def resolve_spawn(self, ctx: SpawnContext) -> SpawnPlan:
        """Resolve the PROCESS-SCOPED half of the launch: the self-served binary and
        argv, and the credential mask an enforced host is confined by.

        This runs on the shared runtime, where ``ctx.session is None`` and the
        process is shared by every session. Only the parts that depend on
        ``ctx`` alone belong here:

        * the binary/argv off the self-served ladder (``resolve_self_served_launch``),
          which keys on the backend id, not a session;
        * the credential-mask preflight (``_run_preflight_bounded(_sandbox_preflight)``),
          which keys on backend + ``ctx.sandbox_mode``. opencode is ENFORCED, so this
          mask is the compensating control for the passive reads ACP v1 cannot make
          it ask about; an empty mask here is a missing control, not a simplification.

        The version gate and the permission read-back need a session
        (``session._agent_env_for_spawn()``, ``session._extra_env``,
        ``session._spawn_work_dir``), so they run per session in
        :meth:`verify_session_routing`, before the first prompt.
        """
        # Deferred imports: kiro_crew.acp.launch and kiro_crew.acp_tool_gate reach
        # back into the harness registry this module is part of, so both resolve at
        # call time rather than at module scope (the same reason codex defers its
        # client import).
        from kiro_crew import acp_tool_gate
        from kiro_crew.acp import launch as launch_mod

        _bin, argv, spawn_label, stderr_label = await launch_mod.resolve_self_served_launch(
            self.backend
        )
        # The credential mask, keyed on backend + tier. opencode is enforced, so a
        # tier where the mask cannot be applied refuses inside this call.
        hidden = await launch_mod._run_preflight_bounded(
            launch_mod._sandbox_preflight, self.backend, ctx.sandbox_mode
        )
        expose = acp_tool_gate.adapter_expose_files(self.backend, hidden)
        return SpawnPlan(
            argv=argv,
            spawn_label=spawn_label,
            stderr_label=stderr_label,
            extra_hidden_dirs=hidden,
            extra_expose_files=expose,
        )

    def apply_spawn_env(
        self,
        env: dict[str, str],
        *,
        spawned_binary: str | None = None,
        cli_owned_auth: bool = False,
    ) -> None:
        """Write the process-level permission seed into the child's environment.

        ``OPENCODE_CONFIG_CONTENT`` is read once at exec, so every session on this
        shared process verifies the same document. The seed therefore carries only
        the required asking setting and no per-tool deny rules; a session whose
        projection asks for those is refused per session instead of being seeded
        differently (a per-tool deny cannot be scoped to one session's exec).

        The ambient value is read from the environment this child is about to
        receive, not from the gateway's own, so the merge sees exactly the source
        the child will. It is captured on the harness for
        :meth:`verify_session_routing`, which reads back the same document instead
        of rebuilding it from a possibly-changed environment.

        opencode brings its own sign-in, so there is no kiro-cli API key to strip
        the way codex does.
        """
        from kiro_crew.acp.harness import opencode as opencode_mod

        ambient = env.get(opencode_mod._ENV_OPENCODE_CONFIG_CONTENT, "")
        self._config_content = opencode_mod._opencode_routing_config(
            self.backend, (), ambient=ambient
        )
        if self._config_content:
            env[opencode_mod._ENV_OPENCODE_CONFIG_CONTENT] = self._config_content

    @property
    def verifies_agent_activation(self) -> bool:
        """No -- opencode takes no ``--agent`` flag, so nothing is selected at spawn
        for a later ``set_mode`` check to confirm. Same answer as codex."""
        return False

    # ── Per-session routing verification ──

    def process_config_content(self) -> str:
        """The ``OPENCODE_CONFIG_CONTENT`` seed of the SHARED process.

        Process-level, so it carries no per-tool deny rules (those are per session
        and the env is fixed at exec). Returns the value captured at spawn when
        there was one, so the read-back verifies exactly the document the child
        received rather than rebuilding it from an environment that may have moved.
        A harness with no captured seed (a unit test, or a verification reached
        before ``apply_spawn_env``) rebuilds the same uniform document.
        """
        if self._config_content:
            return self._config_content
        from kiro_crew.acp.harness import opencode as opencode_mod

        return opencode_mod._opencode_routing_config(self.backend, ())

    async def verify_session_routing(self, handle: Any) -> None:
        """Run opencode's own ``debug config`` read-back against THIS session.

        The seed is process-level but the project config opencode resolves depends
        on the session's work dir, so the verification that the asking posture is in
        force happens here, per session, before its first prompt. Refusal raises
        :class:`AcpToolGateUnroutable`, which ends the session instead of running it
        with an unverified posture.

        The judgement and both child runs are the per-session launch's own functions
        (:mod:`kiro_crew.acp.harness.opencode`), reached through a read-only view of
        the handle so the audited spawn sites exist once.
        """
        from kiro_crew import acp_tool_gate
        from kiro_crew.acp import launch as launch_mod
        from kiro_crew.acp.harness import opencode as opencode_mod
        from kiro_crew.acp.transport_errors import AcpToolGateUnroutable
        from kiro_crew.sandbox import wrap_argv, wrap_argv_async

        if handle.harness_deny_rules:
            # Per-tool deny rules live in the process-level config, which is shared by
            # every session on this process, and session/new has already mounted the
            # servers they narrow. Nothing can enforce them for this session alone, so
            # the session is refused rather than run with a switched-off tool live.
            raise AcpToolGateUnroutable(
                f"{self.backend} launch refused: this session's agent switches off "
                f"{len(handle.harness_deny_rules)} individual tool(s), and a shared "
                "opencode process cannot enforce a per-tool deny for one session. "
                "Remove the per-tool restriction from the agent, or use a per-session "
                "opencode process."
            )

        runtime = handle._runtime
        sandbox_mode = runtime.sandbox_mode
        hidden = tuple(runtime.sandbox_hidden_dirs)
        expose = acp_tool_gate.adapter_expose_files(self.backend, hidden)
        opencode_bin = (await launch_mod.resolve_self_served_launch(self.backend))[0]
        view = _HandleReadbackView(handle)

        version_issue, version_remedy = await opencode_mod._opencode_version_refusal(
            view,  # type: ignore[arg-type]
            opencode_bin,
            sandbox_mode=sandbox_mode,
            hidden=hidden,
            expose=expose,
        )
        if version_issue:
            raise AcpToolGateUnroutable(
                f"{self.backend} launch refused: {version_issue}. {version_remedy}"
            )

        config_content = self.process_config_content()
        readback_argv, readback_cleanup = await wrap_argv_async(
            [opencode_bin, *opencode_mod._OPENCODE_CONFIG_READBACK_ARGS],
            mode=sandbox_mode,
            strip_python_env=True,
            extra_hidden_dirs=hidden,
            extra_expose_files=expose,
            _prepare=wrap_argv,
        )
        try:
            issue, remedy = await asyncio.to_thread(
                opencode_mod._verify_opencode_routing,
                view,  # type: ignore[arg-type]
                self.backend,
                readback_argv,
                config_content,
            )
        finally:
            if readback_cleanup:
                await asyncio.to_thread(launch_mod._unlink_readback_launcher, readback_cleanup)
        # Recorded whatever the verdict: the turn-time hook matcher reads them even
        # for an admitted session.
        handle.opencode_config_mcp_servers = tuple(view._opencode_config_mcp_servers)
        handle.opencode_denies_unenforced = frozenset(view._opencode_denies_unenforced)
        handle.opencode_native_mounts = frozenset(view._opencode_native_mounts)
        if issue:
            try:
                acp_tool_gate.enforce_runtime_routing(self.backend, issue, remedy=remedy)
            except acp_tool_gate.ToolGateUnroutable as exc:
                raise AcpToolGateUnroutable(str(exc)) from None

    # ── Seam 2: initialize ──

    @property
    def protocol_version(self) -> Any:
        return PROTOCOL_VERSION_OPENCODE

    @property
    def client_capabilities(self) -> dict[str, Any]:
        # TODO(opencode-shared-runtime): confirm against a live opencode handshake
        # capture that the shared ACP_CLIENT_CAPABILITIES is accepted unchanged (codex
        # uses it as-is). Start from the shared object; narrow only if a capture shows
        # opencode needs a variant.
        return ACP_CLIENT_CAPABILITIES

    # ── Seam 3: session/new and session/load extras ──

    async def session_extras(
        self,
        agent: str,
        *,
        work_dir: str | Path | None,
        mcp_gateway_overlay: Any = None,
        member_dispatch: bool = False,
        crew_panel: bool = False,
        session_key: str = "",
        worker_stub_names: frozenset[str] | None = None,
        engine_features: frozenset[str] = frozenset(),
    ) -> SessionExtras:
        """Empty for now. opencode has no ``--agent`` custom-agent channel to register
        on, so like codex this carries no ``custom_agents``.

        The per-session permission read-back is NOT here: it runs on the
        ``verify_session_routing`` seam, which the runtime calls with the live
        ``AcpSessionHandle`` after ``session/new`` and before the first prompt.
        """
        return SessionExtras()

    def session_mcp_servers(
        self,
        requested: list[dict[str, Any]],
        *,
        agent_capabilities: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Narrow the array to the transports THIS session's handshake advertised,
        the way codex does.

        TODO(opencode-shared-runtime): confirm opencode's ``mcpCapabilities`` shape
        from a live handshake capture and reuse the same ``drop_unadvertised_transports``
        narrowing codex uses. An UNKNOWN handshake (absent / non-mapping
        ``mcpCapabilities``) must pass the array through untouched, never narrow to
        nothing. Pass-through placeholder until that capture is in.
        """
        return list(requested)

    @property
    def wants_session_file_on_load(self) -> bool:
        """No -- opencode keeps its own session records (it is in
        ``ACP_BACKENDS_HARNESS_OWNED_SESSIONS``) and resolves a ``session/load`` from
        the id alone, so there is no Crew-side transcript path to send."""
        return False

    # ── Seam 4: inbound requests the host answers ──

    @property
    def host_answered_methods(self) -> tuple[str, ...]:
        """None at the connection level. opencode brings its own sign-in and asks Crew
        for no credential callback, so like codex this is empty (not absent: a reader
        can tell "needs no callback" from "nobody checked")."""
        return ()

    async def answer_request(self, method: str) -> dict[str, Any]:
        """Never called: :attr:`host_answered_methods` is empty. Raises rather than
        returning ``{}`` so a frame this harness never claimed is not answered as if
        it had."""
        raise NotImplementedError(
            f"opencode harness answers no inbound request, including {method!r}"
        )

    # ── Seam 5: notification aliases ──

    @property
    def notification_aliases(self) -> NotificationAliases:
        """Plain ACP. opencode is not kiro-family, so it sends no ``_kiro.dev`` alias,
        announces no subagent roster, and stages no MCP-init frame.

        TODO(opencode-shared-runtime): confirm against a live ``session/update``
        capture that opencode uses only the standard method and no forked spelling."""
        return NotificationAliases(session_update=(METHOD_SESSION_UPDATE,))

    # ── Seam 6: teardown ──

    @property
    def teardown(self) -> TeardownPolicy:
        """The standard ACP ``session/close``, awaited (a REQUEST, not a notification),
        the same shape codex uses.

        opencode 1.19.5 advertises ``sessionCapabilities.close`` and owns its session
        records (``loadSession: true``), so a close evicts the session from the process
        while the record survives for a later ``session/load`` -- evict, not delete,
        which is the ``ACP_BACKENDS_SESSION_SHARING`` precondition.

        TODO(opencode-shared-runtime): this is DECLARED from the handshake, not yet
        PROVEN. ``ACP_BACKENDS_SESSION_EVICTION`` membership requires a live capture
        (``test/fixtures/acp_frames/opencode/session-evict-live.jsonl``): open two
        sessions on one connection, ``session/close`` one, prove its id stops answering
        while the sibling lives. Do not add opencode to SESSION_EVICTION until that
        capture exists. ``notification=False`` because a close sent as a notification
        would be ignored and every session would leak.
        """
        return TeardownPolicy(method=METHOD_SESSION_CLOSE, notification=False)

    # ── Seam 9: reclaim ──
    #
    # Inherited from MembershipHarness: pass the runtime's configured thresholds
    # straight through. codex only overrode this because of its per-session MCP
    # fleet; opencode has no measured leak profile yet, so the operator's
    # configuration is the whole answer until one is measured.
    # TODO(opencode-shared-runtime): if a shared opencode process is measured to grow
    # with sessions, override reclaim_policy with a core-only scope the way codex does.
