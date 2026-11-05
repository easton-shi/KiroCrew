"""The one-line summary a skill-scope audit record carries.

Imported by ``kas_agents._audit_skill_scope``; ``kas_agents._skill_scope_summary``
stays the import path callers use.
"""

from __future__ import annotations

from typing import Any

#: How many skill names one skill-scope record repeats before it summarises.
_SKILL_SCOPE_AUDIT_NAMES = 40


def _skill_scope_summary(rules: tuple[dict[str, Any], ...]) -> str:
    """What the skill-scope denies keep and take away, in a bounded line."""
    kept: list[str] = []
    denied: list[str] = []
    deny_all = False
    for rule in rules:
        if rule.get("exclude"):
            kept.extend(rule["exclude"])
        elif rule.get("match"):
            denied.extend(rule["match"])
        else:
            deny_all = True

    def _names(names: list[str]) -> str:
        shown = ", ".join(names[:_SKILL_SCOPE_AUDIT_NAMES])
        more = len(names) - _SKILL_SCOPE_AUDIT_NAMES
        return f"{shown}, and {more} more" if more > 0 else shown

    if kept:
        return f"every skill denied except {len(kept)}: {_names(kept)}"
    if denied:
        return f"{len(denied)} default-tree skills denied by name: {_names(denied)}"
    return "every skill denied" if deny_all else "no skill rule"
