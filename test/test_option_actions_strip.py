"""Non-dashboard surfaces never show a raw ``[OPTION-ACTIONS: ...]`` marker."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from kiro_crew import option_action_marker
from kiro_crew.constants import (
    split_trailing_protocol_suffix,
    strip_control_comments,
)
from kiro_crew.dashboard.state_parts.slot_transcript import _strip_options as slot_preview
from kiro_crew.messaging.egress import channel_egress_parts
from kiro_crew.messaging.renderer import render_options_as_text, split_options_trailer
from kiro_crew.messaging.tables import TABLE_POLICY_CARDS, render_tables
from kiro_crew.messaging.transport import TransportCapabilities
from kiro_crew.option_action_marker import strip_option_action_markers
from kiro_crew.preview_text import strip_markdown_preview
from kiro_crew.slack.format import extract_options
from kiro_crew.slack.handler import _filter_options_brackets, _resolve_comment_hold
from kiro_crew.voice_reply import strip_markdown
from kiro_crew.whatsapp.turn_renderer import _strip_options as whatsapp_strip

_CAPS = TransportCapabilities()
ACTION = "[OPTION-ACTIONS: close=Nothing else, close this session]"


def _egress(text: str) -> str:
    return "\n".join(asyncio.run(channel_egress_parts(text, _CAPS)))


def _slack_stream(text: str) -> str:
    """What Slack's append-only stream shows, fed one character at a time, then settled."""
    hold, streamed = "", ""
    for n in range(1, len(text) + 1):
        hold, streamed = _filter_options_brackets(text[n - 1], hold, streamed, text[:n])
    hold, released = _resolve_comment_hold(hold, text)
    return streamed + released


SURFACES: dict[str, Callable[[str], str]] = {
    "discord-telegram-teams-wecom-webex": lambda t: split_options_trailer(t)[0],
    "streaming-frame": lambda t: split_options_trailer(t, hide_partial=True)[0],
    "weixin-imessage-feishu-mirror": lambda t: render_options_as_text(t, _CAPS),
    "slack-final": lambda t: extract_options(t)[0],
    "slack-live-stream": _slack_stream,
    "voice": strip_markdown,
    "whatsapp": whatsapp_strip,
    "unattended-egress": _egress,
    "notification-preview": strip_markdown_preview,
    "session-card-preview": slot_preview,
}

NESTED_ACTION_REPLY = "Pick one.\n[OPTIONS: Add [OPTION-ACTIONS: close=Done] | Skip]"
NESTED_ACTION_MAIN_RESULTS = {
    "discord-telegram-teams-wecom-webex": "Pick one.",
    "streaming-frame": "Pick one.",
    "weixin-imessage-feishu-mirror": ("Pick one.\n\n1. Add [OPTION-ACTIONS: close=Done]\n2. Skip"),
    "slack-final": "Pick one.",
    "slack-live-stream": "Pick one.\n | Skip]",
    "voice": "Pick one.\n | Skip]",
    "whatsapp": "Pick one.",
    "unattended-egress": NESTED_ACTION_REPLY,
    "notification-preview": "Pick one.",
    "session-card-preview": "Pick one.",
}


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_an_action_marker_inside_an_options_label_is_left_as_main_parses_it(
    surface: str,
) -> None:
    assert SURFACES[surface](NESTED_ACTION_REPLY) == NESTED_ACTION_MAIN_RESULTS[surface]
    assert split_options_trailer(NESTED_ACTION_REPLY) == (
        "Pick one.",
        ["Add [OPTION-ACTIONS: close=Done]", "Skip"],
    )


def test_an_action_marker_inside_an_open_options_head_is_left_as_main_parses_it() -> None:
    partial = "Pick.\n[OPTIONS: Add [OPTION-ACTIONS: clo"
    complete = "Pick.\n[OPTIONS: Add [OPTION-ACTIONS: close=Done]"

    assert split_trailing_protocol_suffix(partial) == (
        "Pick.\n",
        "[OPTIONS: Add [OPTION-ACTIONS: clo",
    )
    assert split_trailing_protocol_suffix(complete) == (complete, "")
    assert split_options_trailer(complete) == (complete, [])
    assert split_options_trailer(complete, hide_partial=True) == (complete, [])
    assert split_options_trailer(partial) == (partial, [])
    assert split_options_trailer(partial, hide_partial=True) == ("Pick.", [])
    assert strip_option_action_markers(partial) == partial
    assert strip_option_action_markers(complete) == complete


@pytest.mark.parametrize("surface", sorted(SURFACES))
@pytest.mark.parametrize(
    "text",
    [
        f"Done.\n{ACTION}",
        "Done.\n[option-actions: close=Close it]",
        f"Done.\n{ACTION}\nMore prose after it.",
    ],
)
def test_every_surface_strips_a_complete_action_marker(surface: str, text: str) -> None:
    shown = SURFACES[surface](text)

    assert "OPTION-ACTIONS" not in shown.upper()
    assert "Done." in shown


@pytest.mark.parametrize(
    "text",
    [
        f"Pick one.\n[OPTIONS: Ship | Hold]\n{ACTION}",
        f"Pick one.\n{ACTION}\n[OPTIONS: Ship | Hold]",
        f"Pick one.\n[OPTIONS: Ship | Hold] {ACTION}",
        f"Pick one.\n{ACTION} [OPTIONS: Ship | Hold]",
    ],
)
def test_an_action_marker_beside_the_trailer_keeps_its_choices(text: str) -> None:
    for parse in (split_options_trailer, extract_options):
        body, choices = parse(text)
        assert choices == ["Ship", "Hold"]
        assert body.strip() == "Pick one."


@pytest.mark.parametrize(
    "reply",
    [
        "Pick one.\n[OPTIONS: Ship | Hold]\n<!-- keep-visible -->\n"
        "[OPTION-ACTIONS: close=Done]\n<!-- plan_task_id:task-1 -->\n"
        "[OPTION-ACTIONS: close=Close]",
        "Pick one.\n[OPTIONS: Ship | Hold]\n<!-- deliver:discord -->\n"
        "[OPTION-ACTIONS: close=Done]\n<!-- keep-visible -->\n"
        "[OPTION-ACTIONS: close=Close]",
        "Pick one.\n[OPTIONS: Ship | Hold]\n[OPTION-ACTIONS: close=Done]\n"
        "<!-- keep-visible -->\n[OPTION-ACTIONS: close=Close]\n"
        "<!-- plan_task_id:task-1 -->",
    ],
    ids=[
        "trigger",
        "options-control-action-control-action",
        "options-action-control-action-control",
    ],
)
def test_interleaved_action_markers_and_control_tags_keep_the_trailer_choices(
    reply: str,
) -> None:
    body, choices = split_options_trailer(reply)

    assert choices == ["Ship", "Hold"]
    assert "[OPTIONS" not in body
    assert "OPTION-ACTIONS" not in body


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_a_crlf_reply_strips_the_action_marker_and_keeps_the_choices(surface: str) -> None:
    without_action = "Pick.\r\n[OPTIONS: Ship | Hold]\r\n"
    reply = without_action + "[OPTION-ACTIONS: close=Done]\r\n"
    shown = SURFACES[surface](reply)
    choice_surfaces = {
        "discord-telegram-teams-wecom-webex": lambda t: split_options_trailer(t)[1],
        "streaming-frame": lambda t: split_options_trailer(t, hide_partial=True)[1],
    }

    assert "OPTION-ACTIONS" not in shown.upper()
    assert shown.rstrip() == SURFACES[surface](without_action).rstrip()
    if surface in choice_surfaces:
        assert choice_surfaces[surface](reply) == ["Ship", "Hold"]


@pytest.mark.parametrize(
    "reply",
    [
        "Pick.\r\n[OPTION-ACTIONS: close=Done] [OPTIONS: Ship | Hold]\r\n",
        "Pick.\r\n[OPTIONS: Ship | Hold] [OPTION-ACTIONS: close=Done]\r\n",
    ],
)
def test_crlf_action_and_options_siblings_keep_the_choices(reply: str) -> None:
    body, choices = split_options_trailer(reply)

    assert body == "Pick."
    assert choices == ["Ship", "Hold"]


@pytest.mark.parametrize(
    "text",
    [
        f"```\n{ACTION}\n```",
        f"~~~md\n```\n{ACTION}\n~~~",
        f"````\n```\n{ACTION}\n````",
        f"```\n{ACTION}",
        f'<mcwidget title="Syntax">\n{ACTION}\n</mcwidget>',
        f"<MCWIDGET>{ACTION}",
    ],
)
def test_a_marker_shown_verbatim_on_the_dashboard_stays_as_written(text: str) -> None:
    assert strip_option_action_markers(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "[OPTION-ACTIONS: close=Close after checking arr[0]",
        "[OPTION-ACTIONS: close=Close 【x] now]",
        f"{ACTION} in the docs",
        "[OPTION-ACTIONS: close=A] [OPTION-ACTIONS: close=B arr[0]",
        "[OPTION-ACTIONS: close=A] [OPTIONS: B] then prose",
    ],
)
def test_a_candidate_that_is_not_a_marker_stays_visible(text: str) -> None:
    assert strip_option_action_markers(text) == text
    assert split_options_trailer(text)[0] == text


CONTROL_TAG_PATHS: dict[str, Callable[[str], str]] = {
    "discord-webex-final": lambda t: split_options_trailer(strip_control_comments(t))[0],
    "unattended-egress": _egress,
    "notification-preview": strip_markdown_preview,
}


@pytest.mark.parametrize("path", sorted(CONTROL_TAG_PATHS))
def test_a_control_tag_above_the_action_marker_is_stripped_too(path: str) -> None:
    shown = CONTROL_TAG_PATHS[path](f"Done.\n<!-- keep-visible -->\n{ACTION}")

    assert "keep-visible" not in shown
    assert "OPTION-ACTIONS" not in shown


def test_an_options_marker_between_two_action_markers_keeps_the_line() -> None:
    text = "Pick.\n[OPTION-ACTIONS: close=X] [OPTIONS: Ship | Hold] [OPTION-ACTIONS: close=Y]"

    for parse in (split_options_trailer, extract_options):
        body, choices = parse(text)
        assert choices == ["Ship", "Hold"]
        assert body.strip() == "Pick."


@pytest.mark.parametrize("partial", ["[OPTION-ACTIONS: clo", "[option-actions: clo"])
def test_a_partial_action_after_the_trailer_still_offers_its_choices(partial: str) -> None:
    body, choices = split_options_trailer(
        f"Pick.\n[OPTIONS: Ship | Hold] {partial}", hide_partial=True
    )

    assert (body, choices) == ("Pick.", ["Ship", "Hold"])


@pytest.mark.parametrize(
    "text", [f"```\n{ACTION}\n```", f"<mcwidget>{ACTION}</mcwidget>", f"```\n{ACTION}"]
)
def test_the_slack_live_stream_keeps_a_verbatim_action_marker(text: str) -> None:
    assert _slack_stream(text) == text


@pytest.mark.parametrize("marker", [ACTION, "[OPTIONS: Yes | No]"])
def test_the_slack_live_stream_leaves_a_marker_after_a_fenced_open_bracket_as_main_does(
    marker: str,
) -> None:
    text = f'```python\nprint("[")\n```\n{marker}'

    assert _slack_stream(text) == text


def test_the_slack_live_stream_keeps_an_options_example_in_a_multiline_bracket() -> None:
    text = '```python\nexample = [\n    "[OPTIONS: Yes | No]",\n]\n```\n'

    assert _slack_stream(text) == text


@pytest.mark.parametrize("tail", ["\n", ""])
def test_the_slack_live_stream_keeps_options_nested_in_a_rejected_action(tail: str) -> None:
    text = f"[OPTION-ACTIONS: close=example [OPTIONS: Yes | No]{tail}"

    assert _slack_stream(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "See [the [OPTIONS: Yes | No] syntax](https://example.org) for details.\n",
        "See [the [OPTIONS: Yes | No] syntax](https://example.org) for details.",
    ],
)
def test_the_slack_live_stream_keeps_an_options_tag_nested_in_another_bracket(text: str) -> None:
    assert _slack_stream(text) == text


def test_the_slack_live_stream_still_drops_an_options_tag_after_a_closed_bracket() -> None:
    assert _slack_stream("See [1] then\n[OPTIONS: Yes | No]") == "See [1] then\n"


@pytest.mark.parametrize("marker", [ACTION, "[OPTIONS: Yes | No]"])
def test_the_slack_live_stream_leaves_a_marker_after_a_prose_open_bracket_as_main_does(
    marker: str,
) -> None:
    text = f"see [1\n{marker}"

    assert _slack_stream(text) == text


def test_the_slack_live_stream_judges_a_fence_opened_in_an_earlier_chunk() -> None:
    reply = f"```\n{ACTION}"
    hold, streamed = _filter_options_brackets(ACTION, "", "", reply)
    _, released = _resolve_comment_hold(hold, reply)

    assert streamed + released == ACTION


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("Done.\n[OPTION-ACTIONS: close=Close [x] now]\nAfter.", "Done.\n\nAfter."),
        ("Use [OPTION-ACTIONS: close=A] in docs\n", "Use [OPTION-ACTIONS: close=A] in docs\n"),
        (
            "[OPTION-ACTIONS: close=Close after arr[0]\n",
            "[OPTION-ACTIONS: close=Close after arr[0]\n",
        ),
    ],
)
def test_the_slack_live_stream_judges_a_whole_line(text: str, shown: str) -> None:
    assert _slack_stream(text) == shown


@pytest.mark.parametrize("tail", ["\n", ""])
def test_the_slack_live_stream_drops_an_options_marker_after_an_action_marker(tail: str) -> None:
    shown = _slack_stream(f"Pick one.\n{ACTION} [OPTIONS: Ship | Hold]{tail}")

    assert "OPTIONS" not in shown
    assert shown.startswith("Pick one.")


def test_the_slack_live_stream_judges_a_line_whose_reply_was_reset() -> None:
    hold, _ = _filter_options_brackets("[OPTION-ACT", "", "", "Done.\n[OPTION-ACT")
    hold, streamed = _filter_options_brackets(
        "IONS: close=A] tail\n", hold, "", "IONS: close=A] tail\n"
    )

    assert streamed == "[OPTION-ACTIONS: close=A] tail\n"


@pytest.mark.parametrize(
    "partial", ["[OPTION-ACTIONS: close=Close [x] now", "[OPTION-ACTIONS: close=Cl 【x】"]
)
def test_a_partial_action_with_a_matched_pair_is_still_held(partial: str) -> None:
    assert split_options_trailer(f"Body {partial}", hide_partial=True) == ("Body", [])
    assert split_trailing_protocol_suffix(f"Body {partial}") == ("Body ", partial)


@pytest.mark.parametrize(
    "text",
    [
        "Body [OPTION-ACTIONS: close=x] see note [1",
        "Body [OPTION-ACTIONS: close=x] then 【重要 note",
        "Body [OPTION-ACTIONS: close=x】 then prose",
        "Body [OPTION-ACTIONS: close=x], then prose",
        "Body [OPTION-ACTIONS: close=x] | and more",
    ],
)
def test_prose_after_a_closed_action_head_is_not_held(text: str) -> None:
    assert split_options_trailer(text, hide_partial=True) == (text, [])
    assert split_trailing_protocol_suffix(text) == (text, "")


def test_a_partial_action_head_on_an_earlier_line_is_prose() -> None:
    text = "Intro [OPTION-ACTIONS: illustrative\nFinal result is ready."

    assert split_options_trailer(text, hide_partial=True) == (text, [])
    assert split_trailing_protocol_suffix(text) == (text, "")


def test_a_lowercase_action_line_under_a_table_is_not_a_table_row() -> None:
    marker = "[option-actions: close=X | close=Y]"
    out = render_tables(f"| a | b |\n| --- | --- |\n| 1 | 2 |\n{marker}", policy=TABLE_POLICY_CARDS)

    assert out.endswith(f"\n{marker}")


@pytest.mark.parametrize("partial", ["[OPTION-ACTIONS: close=Cl", "[option-actions: close=Cl"])
def test_the_length_split_detaches_a_partial_action_marker_in_any_case(partial: str) -> None:
    assert split_trailing_protocol_suffix(f"Done.\n{partial}") == ("Done.\n", partial)


@pytest.mark.parametrize(
    "text", ["```text\n[OPTION-ACTIONS: close=X\n```", "<mcwidget>\n[OPTION-ACTIONS: close=X"]
)
def test_a_verbatim_partial_action_marker_is_not_cut(text: str) -> None:
    assert split_options_trailer(text, hide_partial=True) == (text, [])
    assert split_trailing_protocol_suffix(text) == (text, "")


def test_only_the_last_of_many_unclosed_action_heads_is_held() -> None:
    text = "Hi " + "[OPTION-ACTIONS:" * 20000

    assert split_options_trailer(text, hide_partial=True) == (text[: -len("[OPTION-ACTIONS:")], [])


@pytest.mark.parametrize(
    "earlier",
    [
        "```text\n[OPTION-ACTIONS: close=X\n```\n",
        "```\n[OPTION-ACTIONS:\n```\n",
        "<mcwidget>[OPTION-ACTIONS:x</mcwidget> ",
    ],
)
def test_a_verbatim_head_earlier_does_not_shield_a_live_partial(earlier: str) -> None:
    text = f"{earlier}Body [OPTION-ACTIONS: close=Y"

    assert split_options_trailer(text, hide_partial=True)[0] == f"{earlier}Body"
    assert split_trailing_protocol_suffix(text)[1] == "[OPTION-ACTIONS: close=Y"


def test_a_body_with_a_matched_pair_is_still_a_marker() -> None:
    assert strip_option_action_markers("Done.\n[OPTION-ACTIONS: close=Close [x] now]") == "Done.\n"


def test_a_fence_that_closed_does_not_protect_a_later_marker() -> None:
    text = f"```\n{ACTION}\n```\nDone.\n{ACTION}"

    assert strip_option_action_markers(text) == f"```\n{ACTION}\n```\nDone.\n"


def test_a_still_arriving_action_marker_is_held_from_a_streaming_frame() -> None:
    text = "Done.\n[OPTION-ACTIONS: close=Close th"

    assert split_options_trailer(text, hide_partial=True)[0] == "Done."
    assert split_options_trailer(text)[0] == text


def test_a_long_sibling_chain_checks_the_options_tail_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    real = option_action_marker._ends_with_options_marker

    def spy(text: str, pos: int, line_end: int) -> bool:
        calls.append(pos)
        return real(text, pos, line_end)

    monkeypatch.setattr(option_action_marker, "_ends_with_options_marker", spy)
    text = "[OPTION-ACTIONS: close=X] " * 2000 + "[OPTIONS: Ship | Hold]"

    assert strip_option_action_markers(text) == "[OPTIONS: Ship | Hold]"
    assert len(calls) == 1


def test_the_length_split_keeps_the_trailer_with_a_following_action_marker() -> None:
    visible, suffix = split_trailing_protocol_suffix(f"Pick one.\n[OPTIONS: Ship | Hold]\n{ACTION}")

    assert visible == "Pick one.\n"
    assert suffix.startswith("[OPTIONS: Ship | Hold]")
    assert split_options_trailer(suffix) == ("", ["Ship", "Hold"])


def test_many_nested_options_heads_keep_the_parent_head_strip_result() -> None:
    text = "[OPTIONS:" * 200 + "]" * 200 + "[OPTION-ACTIONS: x]" * 200
    expected = "[OPTIONS:" * 200 + "]" * 200

    assert strip_option_action_markers(text) == expected
