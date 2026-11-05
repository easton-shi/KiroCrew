"""``[OPTION-ACTIONS: close=<label>]``, the dashboard's zero-turn action chip.

Every other surface has nothing to draw, so each ``[OPTIONS:]`` strip removes this
sibling first. The decisions mirror the dashboard's web grammar.
"""

from __future__ import annotations

import bisect
import re
from typing import Callable

from kiro_crew.constants import (
    _MARKER_BRACKETS,
    _MARKER_CLOSE_CLASS,
    _MARKER_STRAY_TIC,
    _RAW_OPTIONS_RE_LINE,
    MARKER_CLOSERS,
    MARKER_OPENERS,
    OPTIONS_RE_LINE,
    _FenceWalk,
    _in_open_fence,
    _marker_body,
    _marker_labels_have_unmatched_opener,
)

_OPTIONS_HEAD = "[OPTIONS:"
_OPTION_ACTIONS_HEAD = "[OPTION-ACTIONS:"
_MARKER_PAIR = dict(zip(MARKER_OPENERS, MARKER_CLOSERS, strict=True))
_ANY_MARKER_HEAD = r"(?i:OPTIONS?:|OPTION-ACTIONS:)"
_ACTION_BODY_LINE = _marker_body(rf"(?!{_ANY_MARKER_HEAD})", ordinary=rf"[^{_MARKER_BRACKETS}\n]")
_RAW_OPTION_ACTIONS_RE = re.compile(
    rf"\[(?i:OPTION-ACTIONS:)(?P<labels>{_ACTION_BODY_LINE}){_MARKER_CLOSE_CLASS}"
    rf"{_MARKER_STRAY_TIC}[ \t]*(?:\r?$|(?=\[{_ANY_MARKER_HEAD}))",
    re.MULTILINE,
)
_OPTION_ACTIONS_OPEN_RE = re.compile(r"\[(?i:OPTION-ACTIONS)")
_WIDGET_SPAN_RE = re.compile(
    r"<mcwidget\b[^>]*>.*?(?:</mcwidget\s*>|\Z)", re.IGNORECASE | re.DOTALL
)


def in_verbatim_region(text: str, idx: int) -> bool:
    """Whether *idx* sits in a code fence or ``<mcwidget>``, which render as written."""
    if _in_open_fence(text, idx):
        return True
    return any(m.start() <= idx < m.end() for m in _WIDGET_SPAN_RE.finditer(text, 0, idx + 1))


class _OpenOptionsHeadWalk:
    """One left-to-right walk that reports open ``[OPTIONS:`` heads."""

    def __init__(self, text: str, line_start: int) -> None:
        self._text = text
        self._pos = line_start
        self._stack: list[tuple[str, bool]] = []
        self._open_heads = 0

    def advance(self, pos: int) -> bool:
        text = self._text
        i = self._pos
        while i < pos:
            ch = text[i]
            if ch in MARKER_OPENERS:
                is_head = ch == "[" and text.startswith(_OPTIONS_HEAD, i)
                if is_head:
                    if i + len(_OPTIONS_HEAD) > pos:
                        break
                    self._stack.append((ch, True))
                    self._open_heads += 1
                    i += len(_OPTIONS_HEAD)
                    continue
                self._stack.append((ch, False))
            elif ch in MARKER_CLOSERS and self._stack:
                opener, is_head = self._stack[-1]
                if is_head:
                    self._stack[-1] = (opener, False)
                    self._open_heads -= 1
                if _MARKER_PAIR[opener] == ch:
                    self._stack.pop()
            i += 1
        self._pos = i
        return self._open_heads > 0


def unfinished_option_action_prefix_start(text: str) -> int:
    """Start of a trailing action-head prefix outside verbatim text, else ``-1``."""
    line_start = max(0, text.rfind("\n") + 1)
    start = text.rfind("[", line_start)
    if start == -1:
        return -1
    suffix = text[start:]
    if not _OPTION_ACTIONS_HEAD.casefold().startswith(suffix.casefold()):
        return -1
    return -1 if in_verbatim_region(text, start) else start


def unfinished_option_action_start(text: str) -> int:
    """Start of a still-arriving ``[OPTION-ACTIONS`` tail outside verbatim text, else ``-1``."""
    # Only the last head can still be arriving: a body never crosses a later head.
    line_start = max(0, text.rfind("\n") + 1)
    head = None
    for head in _OPTION_ACTIONS_OPEN_RE.finditer(text, line_start):
        pass
    if head is None or not (head.end() == len(text) or text.startswith(":", head.end())):
        return -1
    if _action_head_closes(text, head.end()):
        return -1
    if _OpenOptionsHeadWalk(text, line_start).advance(head.start()):
        return -1
    return -1 if in_verbatim_region(text, head.start()) else head.start()


def _action_head_closes(text: str, pos: int, end: int | None = None) -> bool:
    """Whether a closer at the head's own depth follows *pos*, as the ``[OPTIONS`` probe judges it."""
    stack: list[str] = []
    for i in range(pos, len(text) if end is None else end):
        ch = text[i]
        if ch in MARKER_OPENERS:
            stack.append(ch)
        elif ch in MARKER_CLOSERS:
            if stack:
                if MARKER_OPENERS.index(stack[-1]) == MARKER_CLOSERS.index(ch):
                    stack.pop()
            else:
                return True
    return False


def _ends_with_options_marker(text: str, pos: int, line_end: int) -> bool:
    """Whether an accepted ``[OPTIONS:]`` marker runs from *pos* to *line_end*."""
    if line_end > pos and text[line_end - 1] == "\r":
        line_end -= 1
    m = _RAW_OPTIONS_RE_LINE.match(text, pos, line_end)
    return m is not None and not _marker_labels_have_unmatched_opener(m.group("labels"))


def _accepted_option_action_spans(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every ``[OPTION-ACTIONS:]`` marker in *text* a strip may remove."""
    candidates = list(_RAW_OPTION_ACTIONS_RE.finditer(text))
    if not candidates:
        return []
    options = [match.span() for match in OPTIONS_RE_LINE.finditer(text)]
    option_starts = [start for start, _ in options]
    newlines = [nl.start() for nl in re.finditer("\n", text)]
    widgets = [m.span() for m in _WIDGET_SPAN_RE.finditer(text)]
    widget_starts = [start for start, _ in widgets]
    walk = _FenceWalk()
    fed_to = 0
    lines: list[int] = []
    viable: list[bool] = []
    options_head_walk: _OpenOptionsHeadWalk | None = None
    walked_line = -1
    for m in candidates:
        line = bisect.bisect_left(newlines, m.start())
        line_start = newlines[line - 1] + 1 if line else 0
        if options_head_walk is None or line != walked_line:
            options_head_walk = _OpenOptionsHeadWalk(text, line_start)
            walked_line = line
        open_options_head = options_head_walk.advance(m.start())
        walk.feed_text(text[fed_to:line_start])
        fed_to = max(fed_to, line_start)
        w = bisect.bisect_right(widget_starts, m.start()) - 1
        option = bisect.bisect_right(option_starts, m.start()) - 1
        lines.append(line)
        viable.append(
            not walk.inside
            and not (w >= 0 and m.start() < widgets[w][1])
            and not (option >= 0 and m.end() <= options[option][1])
            and not open_options_head
            and not _marker_labels_have_unmatched_opener(m.group("labels"))
        )
    accepted: list[tuple[int, int]] = []
    next_ok = False
    for i in range(len(candidates) - 1, -1, -1):
        m, ok = candidates[i], viable[i]
        line_end = newlines[lines[i]] if lines[i] < len(newlines) else len(text)
        if ok and m.end() < line_end:
            # A sibling head follows: the rest of the line must be accepted markers.
            if i + 1 < len(candidates) and lines[i + 1] == lines[i]:
                gap_end = candidates[i + 1].start()
                ok = next_ok and (
                    gap_end == m.end() or _ends_with_options_marker(text, m.end(), gap_end)
                )
            else:
                ok = _ends_with_options_marker(text, m.end(), line_end)
        if ok:
            accepted.append(m.span())
        next_ok = ok
    accepted.reverse()
    return accepted


def strip_option_action_markers_after(text: str, start: int) -> str:
    """``text[start:]`` minus the accepted action markers that begin at or after *start*."""
    out: list[str] = []
    cursor = start
    for span_start, span_end in _accepted_option_action_spans(text):
        if span_start >= start:
            out.append(text[cursor:span_start])
            cursor = span_end
    out.append(text[cursor:])
    return "".join(out)


def strip_option_action_markers(text: str) -> str:
    """*text* minus its accepted ``[OPTION-ACTIONS:]`` markers; call it before an ``[OPTIONS:]`` parse."""
    return strip_option_action_markers_after(text, 0)


def _peel_trailing_option_actions(text: str, end: int) -> int:
    """Where *text[:end]* stops once the action markers ending it are peeled off."""
    for start, stop in reversed(_accepted_option_action_spans(text[:end])):
        if text[stop:end].strip():
            break
        end = start
    return end


def _peel_protocol_tail(text: str, end: int, strip_tags: Callable[[str], str]) -> int:
    """Peel trailing action markers, and the control tags each peel exposes, until neither moves."""
    while (peeled := _peel_trailing_option_actions(text, end)) != end:
        end = len(strip_tags(text[:peeled]))
    return end
