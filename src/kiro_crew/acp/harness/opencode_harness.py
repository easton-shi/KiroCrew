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

* Seam 1 (spawn) and the per-session permission read-back. OpenCode's asking
  posture is established by seeding ``OPENCODE_CONFIG_CONTENT`` into the child and
  reading the harness's own resolved config back from a short-lived child. On the
  per-session adapter that runs inside ``resolve_spawn`` against a ``ProcessSession``.
  On ``AcpRuntime`` the spawn has no session (``SpawnContext.session is None``) and
  the process is shared, so the read-back is a PER-SESSION step that has to move onto
  a per-session seam, reading the live ``AcpSessionHandle`` rather than a
  ``ProcessSession``. Until that move lands, :meth:`resolve_spawn` here resolves only
  the process-scoped half (binary, version gate, credential mask) and raises on the
  session-scoped half so a half-wired session can never start ungated.
* The capability-set membership (``ACP_BACKENDS_ACP_RUNTIME`` / ``SESSION_EVICTION``
  / ``SESSION_SHARING``) is NOT flipped yet. This module exists first so the
  registry can reach it and its seams can be tested before a session is routed here,
  exactly as ``__init__.py`` documents for the registration-vs-routing split.
"""

from __future__ import annotations

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

    # ── Seam 1: spawn ──
    #
    # TODO(opencode-shared-runtime): this is the hard seam and it is NOT finished.
    #
    # The per-session OpencodeLaunch.resolve_spawn does five things, and only the
    # first two are process-scoped and safe to run here:
    #   (1) resolve the self-served binary + argv (process-scoped) -- move here;
    #   (2) the credential-mask preflight + version gate (process-scoped, reads only
    #       ctx.sandbox_mode/env and the resolved binary) -- move here;
    #   (3) warm THIS session's MCP array (session._prepare_session_mcp)       -- per-session;
    #   (4) seed OPENCODE_CONFIG_CONTENT and READ BACK the resolved config, which
    #       mutates ProcessSession fields (_opencode_config_mcp_servers,
    #       _opencode_denies_unenforced, _opencode_native_mounts, ...)          -- per-session;
    #   (5) settle_opencode_denies / refuse_native_mount_of_unhonoured          -- per-session.
    #
    # On AcpRuntime SpawnContext.session is None and the process is shared, so (3)-(5)
    # cannot run in resolve_spawn. They have to move onto a per-session seam that the
    # runtime already calls with the live AcpSessionHandle (Seam 3 session_extras /
    # the session-create projection hook), NOT an ``if backend == opencode`` branch on
    # the shared path (harness-parity H13). Reuse the PURE helpers already in
    # kiro_crew.acp.harness.opencode (_opencode_version_refusal,
    # _opencode_routing_config, _verify_opencode_routing, the mask preflight) by
    # importing them -- do not copy.

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

        Deliberately NOT here (see the Seam 1 TODO above): the version gate and the
        permission read-back. Both are coupled to a session's env/workdir
        (``session._agent_env_for_spawn()``, ``session._extra_env``,
        ``session._spawn_work_dir``) and the read-back mutates per-session state, so
        they cannot run on a process-scoped spawn. Until they are moved onto a
        per-session seam reading the live ``AcpSessionHandle``, this method REFUSES
        rather than start a session whose asking posture was never established --
        a half-wired opencode session would execute tool calls that never reach
        Crew's gate.
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

        # TODO(opencode-shared-runtime): the version gate and the permission
        # read-back run next on the per-session launch and both need a session.
        # Move them to a per-session seam, then return the SpawnPlan below instead
        # of raising. The mask is already resolved so it is carried on the plan when
        # that lands:
        #
        #   return SpawnPlan(
        #       argv=argv,
        #       spawn_label=spawn_label,
        #       stderr_label=stderr_label,
        #       extra_hidden_dirs=hidden,
        #       extra_expose_files=expose,
        #   )
        raise NotImplementedError(
            "OpencodeHarness.resolve_spawn resolved the process-scoped half (binary, "
            "argv, credential mask) but the per-session permission read-back is not "
            "wired yet: an opencode session must not start on the shared runtime until "
            "its asking posture is established on a per-session seam. "
            f"(resolved argv={argv!r}, mask={len(hidden)} hidden dirs, {len(expose)} exposed)"
        )

    def apply_spawn_env(
        self,
        env: dict[str, str],
        *,
        spawned_binary: str | None = None,
        cli_owned_auth: bool = False,
    ) -> None:
        # TODO(opencode-shared-runtime): the per-session launch writes
        # OPENCODE_CONFIG_CONTENT here from the read-back it verified. On the shared
        # runtime that seed is per-session, not per-process, so this process-level
        # hook is NOT where it belongs. Left as a no-op until the per-session seam
        # carries the routing seed; opencode brings its own sign-in, so there is no
        # kiro-cli API key to strip the way codex does.
        raise NotImplementedError(
            "OpencodeHarness.apply_spawn_env is not implemented yet: the routing seed "
            "is per-session on the shared runtime and must travel on a session seam."
        )

    @property
    def verifies_agent_activation(self) -> bool:
        """No -- opencode takes no ``--agent`` flag, so nothing is selected at spawn
        for a later ``set_mode`` check to confirm. Same answer as codex."""
        return False

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
    ) -> SessionExtras:
        """Empty for now. opencode has no ``--agent`` custom-agent channel to register
        on, so like codex this carries no ``custom_agents``.

        TODO(opencode-shared-runtime): the per-session permission read-back that
        OpencodeLaunch runs in resolve_spawn belongs on THIS seam (or the
        session-create projection hook the runtime calls), because it needs the live
        session, not a process. Verify the exact hook against runtime_session_create
        before wiring it; until then this returns an empty payload and the
        unimplemented spawn above keeps a session from starting ungated.
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
