"""``kirocrew learn add`` refuses an over-limit rule like MCP and REST do.

``POST /api/lessons`` and the ``learn_add`` MCP tool both validate against
``LEARN_ADD_SCHEMA``, which caps ``rule`` and ``negative`` at
``MAX_SHORT_STRING`` (500). The CLI wrote through ``write_lesson`` directly and
never ran that check, so it saved an over-cap rule the other two surfaces
refuse. The CLI now runs the same validator, refusing with the same message,
exit 2, and no store touched; a rule exactly at the cap still saves.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from kiro_crew import cli_commands
from kiro_crew.constants import MAX_SHORT_STRING
from kiro_crew.vector_memory import VectorMemoryStore


def _add_args(rule: str, negative: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        learn_action="add", rule=rule, category="knowledge", negative=negative
    )


def _store(tmp_path) -> VectorMemoryStore:
    store = VectorMemoryStore(db_path=tmp_path / "m.db", embedding_dim=4)
    store.init()
    return store


@pytest.mark.parametrize("extra", [1, 120])
def test_cli_add_refuses_an_over_limit_rule_before_any_store(capsys, extra) -> None:
    length = MAX_SHORT_STRING + extra
    vs_cls = MagicMock()
    args = _add_args("x" * length)
    with (
        patch.object(cli_commands, "VectorMemoryStore", vs_cls),
        patch.object(cli_commands, "LessonStore", return_value=MagicMock()),
        patch.object(cli_commands.KiroCrewConfig, "load", return_value=MagicMock()),
        pytest.raises(SystemExit) as exc,
    ):
        cli_commands._learn(args)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    # The exact message POST /api/lessons and the learn_add MCP tool return.
    assert f"rule: exceeds max length {MAX_SHORT_STRING} (got {length}, trim {extra} chars)" in err
    vs_cls.assert_not_called()


def test_cli_add_refuses_an_over_limit_negative_clause(capsys) -> None:
    vs_cls = MagicMock()
    args = _add_args("stay concise", negative="n" * (MAX_SHORT_STRING + 5))
    with (
        patch.object(cli_commands, "VectorMemoryStore", vs_cls),
        patch.object(cli_commands, "LessonStore", return_value=MagicMock()),
        patch.object(cli_commands.KiroCrewConfig, "load", return_value=MagicMock()),
        pytest.raises(SystemExit) as exc,
    ):
        cli_commands._learn(args)
    assert exc.value.code == 2
    assert f"negative: exceeds max length {MAX_SHORT_STRING}" in capsys.readouterr().err
    vs_cls.assert_not_called()


def test_cli_add_saves_a_rule_exactly_at_the_cap(tmp_path, capsys) -> None:
    rule = "y" * MAX_SHORT_STRING
    args = _add_args(rule)
    # _learn opens its own store and closes it in a finally, so read the result
    # back through a fresh store on the same db rather than the closed handle.
    store = _store(tmp_path)
    with (
        patch.object(cli_commands, "VectorMemoryStore", return_value=store),
        patch.object(cli_commands, "LessonStore", return_value=MagicMock()),
        patch.object(cli_commands.KiroCrewConfig, "load", return_value=MagicMock()),
    ):
        cli_commands._learn(args)
    assert "Saved:" in capsys.readouterr().out
    reader = _store(tmp_path)
    try:
        stored = reader.get_lessons()
        assert len(stored) == 1
    finally:
        reader.close()
