"""``dashboard.new_chat_crew_mode``: a New chat born in Ultra Mode.

A dashboard create that asks for ``crew_mode_default`` (the person's own New chat,
sent only while the dashboard's Ultra Mode preview is on) starts on the conductor
when the setting is on, and only when nothing else chose an agent: the caller's
pick, an explicit folder's pin and the default folder's pin all win over it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state
from dashboard_owner_helpers import as_owner

from kiro_crew import crew_mode
from kiro_crew.config.loader import KiroCrewAgentConfig, KiroCrewConfig
from kiro_crew.dashboard import chat_handlers


def _folder(fid: str, **extra: Any) -> dict[str, Any]:
    return {"id": fid, "name": fid, "parent_id": None, "collapsed": False, **extra}


@pytest.fixture
def cfg(tmp_path, monkeypatch) -> KiroCrewConfig:
    cfg = KiroCrewConfig()
    cfg.agents = {"default": KiroCrewAgentConfig(kiro_agent="kirocrew")}
    cfg.default_agent = "default"
    cfg.dashboard.new_chat_crew_mode = True
    monkeypatch.setattr(chat_handlers, "KiroCrewConfig", SimpleNamespace(load=lambda: cfg))
    monkeypatch.setattr(chat_handlers, "schedule_eager_spawn", lambda *a, **k: None)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(chat_handlers, "default_project_dir", lambda _ws: str(project))
    # The conductor template is installed: the shipped catalog is the external
    # boundary here, as in chat_agent_selection_helpers.
    from kiro_crew.config import loader

    conductor = crew_mode.crew_mode_agent(cfg)
    installed = loader._materialized_kiro_agent
    monkeypatch.setattr(
        loader,
        "_materialized_kiro_agent",
        lambda name, project_dir=None: name if name == conductor else installed(name, project_dir),
    )
    return cfg


@pytest.fixture
def state(tmp_path, cfg) -> Any:
    state = _make_state(tmp_path)
    state._folders = [_folder("inbox"), _folder("pinned", default_agent="reviewer")]
    return state


async def _agent_of(state: Any, payload: dict[str, Any], *, owner: bool = True) -> str:
    app = web.Application()
    app["state"] = state
    app.router.add_post("/api/chat/slots", chat_handlers.api_chat_slot_create)
    async with TestClient(TestServer(as_owner(app) if owner else app)) as client:
        resp = await client.post("/api/chat/slots", json=payload)
        data = await resp.json()
        assert resp.status == 200, data
        return state._slots[data["key"]].agent


@pytest.mark.asyncio
async def test_an_asked_new_chat_starts_on_the_conductor(state, cfg) -> None:
    agent = await _agent_of(state, {"crew_mode_default": True, "default_folder": True})
    assert agent == crew_mode.crew_mode_agent(cfg)


@pytest.mark.asyncio
async def test_a_default_folder_pin_wins(state, cfg) -> None:
    cfg.dashboard.default_session_folder = "pinned"
    assert await _agent_of(state, {"crew_mode_default": True, "default_folder": True}) != (
        crew_mode.crew_mode_agent(cfg)
    )


@pytest.mark.asyncio
async def test_an_unpinned_default_folder_still_starts_in_ultra_mode(state, cfg) -> None:
    cfg.dashboard.default_session_folder = "inbox"
    agent = await _agent_of(state, {"crew_mode_default": True, "default_folder": True})
    assert agent == crew_mode.crew_mode_agent(cfg)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="not-asked"),
        pytest.param({"crew_mode_default": "yes"}, id="not-true"),
        pytest.param({"crew_mode_default": True, "agent": "kirocrew"}, id="caller-picked"),
        pytest.param({"crew_mode_default": True, "memory_mode": "incognito"}, id="incognito"),
        pytest.param({"crew_mode_default": True, "name": "chat-saved"}, id="named"),
    ],
)
@pytest.mark.asyncio
async def test_other_creates_keep_their_agent(state, cfg, payload) -> None:
    assert await _agent_of(state, payload) != crew_mode.crew_mode_agent(cfg)


@pytest.mark.asyncio
async def test_the_setting_off_keeps_a_plain_chat(state, cfg) -> None:
    cfg.dashboard.new_chat_crew_mode = False
    assert await _agent_of(state, {"crew_mode_default": True}) != crew_mode.crew_mode_agent(cfg)


@pytest.mark.asyncio
async def test_a_non_owner_caller_keeps_a_plain_chat(state, cfg) -> None:
    agent = await _agent_of(state, {"crew_mode_default": True}, owner=False)
    assert agent != crew_mode.crew_mode_agent(cfg)


@pytest.mark.asyncio
async def test_a_conductor_default_agent_is_left_alone(state, cfg, monkeypatch) -> None:
    stamped = chat_handlers.default_template(cfg)
    monkeypatch.setattr(crew_mode, "conductor_agents", lambda _cfg=None: frozenset({stamped}))
    assert await _agent_of(state, {"crew_mode_default": True}) != crew_mode.crew_mode_agent(cfg)


@pytest.mark.asyncio
async def test_a_conductor_that_does_not_resolve_keeps_a_plain_chat(
    state, cfg, monkeypatch
) -> None:
    """A missing conductor must not refuse the New chat (agent_choice_unavailable)."""
    from kiro_crew.config import loader

    monkeypatch.setattr(loader, "_materialized_kiro_agent", lambda name, project_dir=None: "")
    assert await _agent_of(state, {"crew_mode_default": True}) != crew_mode.crew_mode_agent(cfg)
