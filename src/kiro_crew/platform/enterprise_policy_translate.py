"""Turn the control plane's enterprise-management policy into a Kiro Crew policy document.

The policy arrives as Trust V2 YAML: ``rules: [{capability, effect, match | exclude,
when?}]``, already resolved for the ``CREW`` client surface. Kiro Crew governs only the
capabilities in :data:`MANAGED`; every other capability belongs to another client and
is ignored. A rule on a meta-capability (``all``, ``builtin``, ``filesystem``) reaches
each managed capability beneath it (:func:`_meta_children`).

Per managed capability with a resource the rules reduce to one Crew ruleset, because a
Crew ruleset has one mode (allow-list or deny-list). A capability any rule names is
governed: as the policy language reads it, what no ``allow`` grants is denied.

* ``deny``/``ask`` with neither ``match`` nor ``exclude``: nothing is allowed (``allow []``);
* ``deny``/``ask`` with ``exclude E``: only ``E`` is allowed (several excludes intersect).
  The control plane sends one beside every grant, so it says what is allowed;
* ``allow`` with ``match L`` and no such ``exclude``: ``L`` is allowed;
* ``allow`` with no ``match`` and no such ``exclude``: everything is allowed, so the
  rules become a deny-list of the ``deny match`` entries (empty when there are none);
* ``deny``/``ask`` with ``match L``: ``L`` is taken out of the allowed set;
* a capability rules name but nothing grants: nothing is allowed.

An ``allow`` with ``when`` or with ``exclude`` grants nothing here: the reply is already
resolved for this user, so neither shape is expected, and dropping a grant only
narrows. A ``deny`` with ``when`` applies as if unconditional, which also only narrows.
A pattern that is exactly ``*`` covers every resource.

Each row converts a policy pattern into the patterns its Crew scope's matcher reads
(:attr:`Managed.convert`). A pattern with no exact Crew form is dropped from an allowed
set, which narrows, and widened in a denied set, which also narrows. A denied pattern
that cannot be widened safely refuses the policy (:class:`Unrepresentable`).

A denied glob removes the allowed literals it matches. A denied entry inside an allowed
glob, or a denied glob beside an allowed glob, has no one-mode form. The translation
refuses it with :class:`Unrepresentable` rather than widen or narrow the set, so the
fetch fails. Entries compare as the row's Crew matcher compares them. ``ask`` reads as
``deny``. A managed capability no rule names is left out of the document, so the
enterprise baseline holds (off, or Crew's default for an open control).

A ruleset row may carry a floor: entries its Crew scope never forbids
(``approval_modes`` keeps ``normal``, ``trust`` and ``trust_reads``). They are added to
an allowed set and taken out of a denied one, because the scope refuses a policy that
forbids one.

An on/off capability (one with no resource) reduces to ``{"enabled": bool}``
instead, by :func:`switch`:

* any ``deny``/``ask``: ``enabled: false``;
* an ``allow`` and no deny: ``enabled: true``;
* no rule: unnamed, so the enterprise baseline holds (Crew's default for an open
  control such as ``cron``);
* a rule with ``match``, ``exclude`` or ``when``: ``enabled: false``. The policy language
  refuses ``match`` and ``exclude`` on an on/off capability, and the grant shape the control
  plane sends for one is not confirmed yet, so a shape this module does not know turns the
  capability off rather than guess. Leaving it unnamed would keep an open control on.

A gated row (``subagent``) is a Crew capability switch with an inner list: its ruleset
lands as ``{"enabled": bool, "scopes": {<inner>: ruleset}}`` (:func:`_gate`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import yaml  # type: ignore[import-untyped]

from kiro_crew.platform.governance import MODE_ALLOW, MODE_DENY
from kiro_crew.platform.governance_parts.catalog import SCOPE_CATALOG
from kiro_crew.platform.governance_parts.matching import _MATCHERS, _expand

#: A ruleset row: the rules reduce to one allow-list or deny-list (:func:`ruleset`).
RULESET = "ruleset"
#: An on/off row: the rules reduce to ``{"enabled": bool}`` (:func:`switch`).
SWITCH = "switch"
#: An on/off row with an inner list: a ruleset under ``{"enabled", "scopes"}`` (:func:`_gate`).
GATE = "gate"

#: A policy pattern's place in a reduction: what it grants, or what it takes away.
ALLOW_ROLE = "allow"
DENY_ROLE = "deny"

_EVERYTHING = "*"


class TranslationError(ValueError):
    """The policy cannot be translated."""


class Unrepresentable(TranslationError):
    """The rules for a managed capability have no one-mode Crew form."""


def _single_star(pattern: str, role: str) -> List[str]:
    """A policy pattern whose only wildcard is ``*`` (spec 4.3), for an fnmatch matcher.

    ``?`` and ``[`` are literal in the policy language and special to fnmatch, so each
    is put in a one-character class.
    """
    del role
    return [pattern.replace("[", "[[]").replace("?", "[?]")]


def _drop_or_refuse(pattern: str, role: str, why: str) -> List[str]:
    """No Crew form: an allowed pattern is dropped (narrows); a denied one refuses."""
    if role == ALLOW_ROLE:
        return []
    raise Unrepresentable(f"denied pattern {pattern[:80]!r}: {why}")


def _mcp_patterns(pattern: str, role: str) -> List[str]:
    """A ``server/tool`` policy pattern as Crew ``mcp`` references (``@server[/tool]``).

    Crew's ``mcp`` matcher has no wildcard: ``@server`` covers every tool of that server
    and ``@server/tool`` only itself. ``server/*`` is ``@server``. A tool glob is widened
    to its whole server when denied and dropped when allowed. A server glob has no Crew
    form.
    """
    text = pattern.strip()
    server, sep, tool = text.partition("/")
    if not server or text.startswith("@") or any(c in server for c in "*?["):
        return _drop_or_refuse(pattern, role, "no Crew form for this MCP server name")
    if not sep:
        # Names no ``server/tool`` resource, so an allow grants nothing; a deny is
        # read as the whole server, which only narrows.
        return [] if role == ALLOW_ROLE else [f"@{server}"]
    if tool == _EVERYTHING:
        return [f"@{server}"]
    if not tool or "/" in tool:
        return _drop_or_refuse(pattern, role, "not a server/tool reference")
    if "*" in tool:
        return [] if role == ALLOW_ROLE else [f"@{server}"]
    return [f"@{server}/{tool}"]


def _expand_braces(pattern: str, start: int = 0) -> Optional[List[str]]:
    """``{a,b}`` alternatives as separate patterns, or ``None`` when Crew cannot say it.

    A brace with no comma, such as ``{prod}``, is literal, as in minimatch. A range
    (``{1..3}``) and a nested or unbalanced brace have no Crew form.
    """
    open_at = pattern.find("{", start)
    if open_at < 0:
        return None if "}" in pattern[start:] else [pattern]
    end = pattern.find("}", open_at)
    if end < 0 or "{" in pattern[open_at + 1 : end] or "}" in pattern[start:open_at]:
        return None
    body = pattern[open_at + 1 : end]
    if ".." in body:
        return None
    if "," not in body:
        return _expand_braces(pattern, end + 1)
    out: List[str] = []
    for alt in body.split(","):
        rest = _expand_braces(pattern[:open_at] + alt + pattern[end + 1 :], open_at + len(alt))
        if rest is None:
            return None
        out.extend(rest)
    return out


def _any_case(pattern: str) -> str:
    """*pattern* with each ASCII letter outside a ``[...]`` class matching either case."""
    out: List[str] = []
    in_class = False
    for ch in pattern:
        if in_class:
            in_class = ch != "]"
            out.append(ch)
        elif ch == "[":
            in_class = True
            out.append(ch)
        elif ch.isascii() and ch.isalpha():
            out.append(f"[{ch.lower()}{ch.upper()}]")
        else:
            out.append(ch)
    return "".join(out)


_ABSOLUTE = re.compile(r"^(/|~(/|$)|[A-Za-z]:/)")
#: A ``**`` that is a whole path segment. Any other ``**`` reads as ``*`` (spec 4.3).
_WHOLE_GLOBSTAR = re.compile(r"(?:(?<=/)|^)\*\*(?=/|$)")
#: A whole-segment ``**`` with the separator before it (``/**/`` or a trailing ``/**``).
_WHOLE_GLOBSTAR_SLASH = re.compile(r"/\*\*(?=/|$)")
#: A one-letter class ``_any_case`` writes, such as ``[eE]``.
_CASE_CLASS = re.compile(r"\[([a-z])([A-Z])\]")


def _fs_one(pattern: str, role: str) -> List[str]:
    wild = any(c in pattern for c in "*?[")
    if not _ABSOLUTE.match(pattern):
        if role == ALLOW_ROLE:
            # Relative to the workspace root, which a policy-level ruleset does not
            # have (shortcut: anchor at each session's workspace if the gate learns it).
            return []
        while pattern.startswith(("**/", "./")):
            pattern = pattern[3:] if pattern.startswith("**/") else pattern[2:]
        # In any directory: wider than under the workspace root, so a deny only narrows.
        bases = [f"*/{pattern}"]
    else:
        if role == ALLOW_ROLE and any(c in _WHOLE_GLOBSTAR.sub("", pattern) for c in "*?["):
            # A policy ``*`` stays inside one path segment; Crew's crosses ``/``.
            return []
        bases = [pattern]
    out: List[str] = []
    for base in bases:
        if role == DENY_ROLE:
            # Each whole-segment ``**`` may stand for no directory at all. A ``*`` in its
            # place (``/a/**/b`` -> ``/a*b``) covers every such choice at once, and more,
            # which a deny may.
            # ``~`` is expanded first, because ``~*`` does not name the home directory.
            variants = [_WHOLE_GLOBSTAR_SLASH.sub("*", _expand(base))]
        else:
            # Fewer choices only narrow an allow.
            variants = [base, base.replace("/**/", "/")]
            if base.endswith("/**"):
                variants.append(base[: -len("/**")])
        for v in variants:
            out.append(v)
            if not wild:
                # A pattern with no wildcard names a directory too and covers beneath it.
                out.append(v.rstrip("/") + "/*")
    if role == DENY_ROLE:
        # The policy language matches case-insensitively on a case-insensitive
        # filesystem; folding every deny keeps that and only narrows elsewhere.
        out = [_any_case(v) for v in out]
    return list(dict.fromkeys(out))


def _fs_patterns(pattern: str, role: str) -> List[str]:
    """A policy file-path glob (minimatch, spec 4.3) as Crew ``path`` matcher patterns.

    Exact for an absolute or ``~`` path whose only wildcard is a whole ``**`` segment,
    or with none (then it also covers what is beneath it). A deny is widened where Crew
    cannot be exact: a relative pattern matches in any directory, and letters match
    either case. An allow Crew cannot match exactly is dropped.
    """
    text = pattern.replace("\\", "/")
    if text.startswith("!") or "[[:" in text or "$" in text or re.search(r"[@+!?*]\(", text):
        return _drop_or_refuse(pattern, role, "no Crew form for this path glob")
    expanded = _expand_braces(text.replace("[^", "[!"))
    if expanded is None:
        return _drop_or_refuse(pattern, role, "unbalanced or nested braces")
    out: List[str] = []
    for one in expanded:
        out.extend(_fs_one(one, role))
    return list(dict.fromkeys(out))


@dataclass(frozen=True)
class Managed:
    """Where a policy capability lands in a Crew policy document, and in which form."""

    path: Tuple[str, ...]
    kind: str = RULESET
    #: Entries the Crew scope never forbids, added to a translated allow-list.
    floor: Tuple[str, ...] = ()
    #: The Crew matcher the landed patterns are read with.
    matcher: str = "identifier"
    #: One policy pattern, in an allowed or a denied set, as Crew patterns.
    convert: Callable[[str, str], List[str]] = _single_star
    #: A gated row's inner list (``capabilities.spawn`` -> ``agents``).
    inner: str = ""
    #: Every value of a closed list, so a deny-list can be written as an allow-list.
    values: Tuple[str, ...] = ()


#: Each capability Kiro Crew governs from enterprise management, and where it lands.
MANAGED: Mapping[str, Managed] = MappingProxyType(
    {
        "messaging_channel": Managed(("channels", "members")),
        "messaging": Managed(("capabilities", "messaging"), SWITCH),
        "cron": Managed(("capabilities", "cron"), SWITCH),
        "permission_mode": Managed(
            ("approval_modes",),
            floor=SCOPE_CATALOG["approval_modes"].always_permitted,
            # The policy language's closed ``approvalMode`` list.
            values=SCOPE_CATALOG["approval_modes"].always_permitted + ("yolo",),
        ),
        "shell": Managed(("commands",), matcher=SCOPE_CATALOG["commands"].matcher),
        "fs_read": Managed(
            ("filesystem", "read"),
            matcher=SCOPE_CATALOG["filesystem.read"].matcher,
            convert=_fs_patterns,
        ),
        "fs_write": Managed(
            ("filesystem", "write"),
            matcher=SCOPE_CATALOG["filesystem.write"].matcher,
            convert=_fs_patterns,
        ),
        "network_egress": Managed(
            ("network", "egress"), matcher=SCOPE_CATALOG["network.egress"].matcher
        ),
        "mcp": Managed(("mcp",), matcher=SCOPE_CATALOG["mcp"].matcher, convert=_mcp_patterns),
        "subagent": Managed(
            ("capabilities", "spawn"),
            GATE,
            matcher=SCOPE_CATALOG["capabilities.spawn"].scope_matchers["agents"],
            inner="agents",
        ),
        "tailnet_origin": Managed(("capabilities", "tailnet_origin"), SWITCH),
        "browser_use": Managed(("capabilities", "browse"), SWITCH),
        "theme_install": Managed(("capabilities", "theme_install"), SWITCH),
    }
)

#: How often the running gateway re-reads the policy: five minutes, the Kiro agent's
#: enterprise-management refresh interval. A document with no interval is read at boot
#: only.
REFRESH_INTERVAL_SECS = 300

#: The boot controls of every translated document. The enterprise policy has no rule for
#: them, so they hold the enterprise defaults: the terminal off, a failed policy load
#: refuses to start, and the sandbox requirement left at Crew's own default (RFC
#: appendix, "Minimum sandbox level").
BOOT: Mapping[str, bool] = MappingProxyType(
    {"require_sandbox": False, "allow_terminal": False, "fail_closed": True}
)

_DENY_EFFECTS = ("deny", "ask")
_EFFECTS = ("allow",) + _DENY_EFFECTS
_GLOB_CHARS = frozenset("*?[")
_DESCRIPTION = "Translated from the organisation's enterprise-management policy"
_DEFAULT_ROW = Managed(())


def _folds(row: Managed) -> bool:
    """Whether the row's Crew matcher ignores case and surrounding space."""
    return row.matcher in ("identifier", "host", "mcp")


def _key(pattern: str, row: Managed = _DEFAULT_ROW) -> str:
    return pattern.strip().casefold() if _folds(row) else pattern


def _is_glob(pattern: str, row: Managed = _DEFAULT_ROW) -> bool:
    if row.matcher == "mcp":
        # ``@server`` covers every ``@server/tool``.
        return "/" not in pattern
    return bool(_GLOB_CHARS & set(pattern))


def _matches(item: str, pattern: str, row: Managed = _DEFAULT_ROW) -> bool:
    return _MATCHERS[row.matcher](item, pattern)


def _rules(policy: str) -> List[Mapping[str, Any]]:
    try:
        doc = yaml.safe_load(policy)
    except yaml.YAMLError:
        raise TranslationError("the enterprise policy is not valid YAML") from None
    rules = doc.get("rules") if isinstance(doc, dict) else None
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise TranslationError("the enterprise policy has no list of rules")
    return rules


def _meta_children(capability: str, effect: object) -> Tuple[str, ...]:
    """The managed capabilities a rule on meta-capability *capability* reaches.

    The hierarchy is the one the KAS translation reads
    (``acp.kas_permissions.META_CAPABILITY_EXPANSION``). In the policy language ``all``
    also holds every Crew term, which that table does not list, so a deny on ``all``
    reaches every managed capability and an allow reaches only the table's: the
    stricter reading either way.
    """
    from kiro_crew.acp.kas_permissions import META_CAPABILITY_EXPANSION

    expansion = META_CAPABILITY_EXPANSION.get(capability)
    if expansion is None:
        return ()
    if capability == "all" and effect != "allow":
        return tuple(MANAGED)
    return tuple(c for c in expansion if c in MANAGED)


def _with_meta(rules: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """*rules*, plus a copy of each meta-capability rule for every managed child.

    A meta-capability takes no resource, so a copy carries no ``match`` or ``exclude``:
    a deny copy then denies the whole child (it only narrows), and an allow that named
    resources grants nothing.
    """
    out = list(rules)
    for rule in rules:
        capability = rule.get("capability")
        if not isinstance(capability, str):
            raise TranslationError("a rule's 'capability' is not a string")
        children = _meta_children(capability, rule.get("effect"))
        if not children:
            continue
        scoped = rule.get("match") is not None or rule.get("exclude") is not None
        if scoped and rule.get("effect") == "allow":
            continue
        base = {k: v for k, v in rule.items() if k not in ("capability", "match", "exclude")}
        out.extend({**base, "capability": child} for child in children)
    return out


def _patterns(rule: Mapping[str, Any], key: str, capability: str) -> List[str]:
    raw = rule.get(key)
    if not isinstance(raw, list) or not all(isinstance(p, str) for p in raw):
        raise TranslationError(f"{capability}: '{key}' is not a list of strings")
    return list(raw)


def _covers_everything(patterns: List[str], row: Managed) -> bool:
    """Whether a pattern is ``*`` as the row's matcher reads it (spaces count unless it folds)."""
    return any(_key(p, row) == _EVERYTHING for p in patterns)


def _converted(patterns: List[str], role: str, row: Managed) -> List[str]:
    out: List[str] = []
    for p in patterns:
        out.extend(row.convert(p, role))
    return out


def _admits(patterns: List[str], entry: str, row: Managed = _DEFAULT_ROW) -> bool:
    """Does the allowed set *patterns* admit the literal *entry*?"""
    return any(
        _key(p, row) == _key(entry, row) or (_is_glob(p, row) and _matches(entry, p, row))
        for p in patterns
    )


def _intersect(sets: List[List[str]], capability: str, row: Managed = _DEFAULT_ROW) -> List[str]:
    """The entries every allowed set in *sets* admits, as one list, or raise.

    A literal is kept when every set admits it, by equality or through a glob. A glob is
    kept only when every set names that same glob. Beside a set that names only literals,
    dropping it is exact: what the two share is those literals, which are kept on their
    own. Beside a set holding a different glob, the shared part has no list form, so the
    translation refuses rather than narrow it.
    """
    seen: Dict[str, str] = {}
    for patterns in sets:
        for p in patterns:
            seen.setdefault(_key(p, row), p)
    allowed: List[str] = []
    for p in seen.values():
        if not _is_glob(p, row):
            if all(_admits(patterns, p, row) for patterns in sets):
                allowed.append(p)
            continue
        lacking = [s for s in sets if all(_key(q, row) != _key(p, row) for q in s)]
        if not lacking:
            allowed.append(p)
        elif any(_is_glob(q, row) for s in lacking for q in s):
            raise Unrepresentable(f"{capability}: allowed globs {p!r} and another do not intersect")
    return allowed


def _literal_prefix(pattern: str, row: Managed) -> str:
    """The text every match of *pattern* starts with, compared as the row compares."""
    if row.matcher == "path":
        pattern = _CASE_CLASS.sub(
            lambda m: m.group(1) if m.group(1).upper() == m.group(2) else m.group(0),
            _expand(pattern),
        ).casefold()
    else:
        pattern = _key(pattern, row)
    cut = min((i for i, c in enumerate(pattern) if c in _GLOB_CHARS), default=len(pattern))
    return pattern[:cut]


def _may_meet(a: str, b: str, row: Managed) -> bool:
    """Whether patterns *a* and *b* can match one item: one literal prefix starts the other.

    Errs towards yes, so a disjoint pair it cannot prove is still refused.
    """
    pa, pb = _literal_prefix(a, row), _literal_prefix(b, row)
    return pa.startswith(pb) or pb.startswith(pa)


def _without_denied(
    allowed: List[str], denied: List[str], capability: str, row: Managed
) -> List[str]:
    for entry in denied:
        # Removing the equal entry is not enough: what is left may still admit it.
        allowed = [p for p in allowed if _key(p, row) != _key(entry, row)]
        globs = [p for p in allowed if _is_glob(p, row) and _may_meet(p, entry, row)]
        if any(_matches(entry, p, row) for p in globs):
            raise Unrepresentable(f"{capability}: deny {entry!r} inside an allowed glob")
        if _is_glob(entry, row):
            # A denied glob removes the allowed literals it matches exactly. Beside
            # an allowed glob it can meet, what is left has no list form.
            allowed = [p for p in allowed if _is_glob(p, row) or not _matches(p, entry, row)]
            if globs:
                raise Unrepresentable(f"{capability}: deny {entry!r} beside an allowed glob")
    return allowed


def _sorted_unique(patterns: List[str], row: Managed) -> List[str]:
    unique: Dict[str, str] = {}
    for p in patterns:
        unique.setdefault(_key(p, row), p)
    return sorted(unique.values())


def ruleset(
    rules: List[Mapping[str, Any]], capability: str, row: Optional[Managed] = None
) -> Optional[Dict[str, Any]]:
    """The Crew ruleset for *capability*'s rules, or ``None`` if no rule names it."""
    row = row or MANAGED.get(capability) or _DEFAULT_ROW
    named = deny_all = allow_all = False
    denied: List[str] = []
    excludes: List[List[str]] = []
    granted: List[str] = []
    for rule in rules:
        if rule.get("capability") != capability:
            continue
        named = True
        effect = rule.get("effect")
        if effect not in _EFFECTS:
            raise TranslationError(f"{capability}: unknown effect {str(effect)[:40]!r}")
        match = _patterns(rule, "match", capability) if rule.get("match") is not None else None
        exclude = (
            _patterns(rule, "exclude", capability) if rule.get("exclude") is not None else None
        )
        if effect == "allow":
            if rule.get("when") is not None or exclude is not None:
                continue
            if match is None or _covers_everything(match, row):
                allow_all = True
            else:
                granted.extend(_converted(match, ALLOW_ROLE, row))
            continue
        if match is not None and not _covers_everything(match, row):
            # A deny's own ``exclude`` beside its ``match`` is ignored: the whole
            # ``match`` is denied, which only narrows.
            denied.extend(_converted(match, DENY_ROLE, row))
        elif exclude is not None and match is None:
            if not _covers_everything(exclude, row):
                excludes.append(_converted(exclude, ALLOW_ROLE, row))
        else:
            deny_all = True
    if not named:
        return None
    if deny_all:
        return {"mode": MODE_ALLOW, "allow": []}
    if excludes:
        allowed = _intersect(excludes, capability, row)
    elif allow_all:
        return {"mode": MODE_DENY, "deny": _sorted_unique(denied, row)}
    else:
        allowed = granted
    allowed = _without_denied(allowed, denied, capability, row)
    return {"mode": MODE_ALLOW, "allow": _sorted_unique(allowed, row)}


def switch(rules: List[Mapping[str, Any]], capability: str) -> Optional[Dict[str, bool]]:
    """The ``{"enabled": ...}`` control for on/off *capability*, or ``None`` to keep
    the baseline."""
    granted = denied = unknown = False
    for rule in rules:
        if rule.get("capability") != capability:
            continue
        effect = rule.get("effect")
        if effect not in _EFFECTS:
            raise TranslationError(f"{capability}: unknown effect {str(effect)[:40]!r}")
        if any(rule.get(key) is not None for key in ("match", "exclude", "when")):
            unknown = True
        elif effect == "allow":
            granted = True
        else:
            denied = True
    if unknown or denied:
        # A shape this module does not know turns the capability off, as an
        # unconditional deny does. Leaving it unnamed is not safe: an unnamed
        # open control (``capabilities.cron``, RFC D3) keeps
        # Crew's default, so a policy that restricts it would widen it instead.
        return {"enabled": False}
    if granted:
        return {"enabled": True}
    return None


def _with_floor(rs: Dict[str, Any], row: Managed) -> Dict[str, Any]:
    floor = row.floor
    if rs["mode"] == MODE_DENY:
        if not row.values:
            kept = {_key(e) for e in floor}
            return {"mode": MODE_DENY, "deny": [p for p in rs["deny"] if _key(p) not in kept]}
        # A denied glob may match a floor entry, which the scope refuses to deny, so
        # the closed list is written out as what stays allowed.
        rs = {
            "mode": MODE_ALLOW,
            "allow": [v for v in row.values if not any(_matches(v, p, row) for p in rs["deny"])],
        }
    allowed = {_key(p): p for p in rs["allow"]}
    for entry in floor:
        allowed.setdefault(_key(entry), entry)
    return {"mode": MODE_ALLOW, "allow": sorted(allowed.values())}


def _gate(rs: Optional[Dict[str, Any]], inner: str) -> Optional[Dict[str, Any]]:
    """A ruleset as a capability switch over its inner list."""
    if rs is None:
        return None
    if rs["mode"] == MODE_ALLOW and not rs["allow"]:
        return {"enabled": False}
    if rs["mode"] == MODE_DENY and not rs["deny"]:
        return {"enabled": True}
    return {"enabled": True, "scopes": {inner: rs}}


def _control(
    rules: List[Mapping[str, Any]], capability: str, row: Managed
) -> Optional[Dict[str, Any]]:
    if row.kind == SWITCH:
        return switch(rules, capability)
    rs = ruleset(rules, capability, row)
    if row.kind == GATE:
        return _gate(rs, row.inner)
    if rs is not None and row.floor:
        rs = _with_floor(rs, row)
    return rs


def neutral_document() -> Dict[str, Any]:
    """A document that governs nothing beyond what the build itself locks."""
    return {
        "version": 1,
        "description": _DESCRIPTION,
        "boot": dict(BOOT),
        "distribution": {"refresh_interval_secs": REFRESH_INTERVAL_SECS},
    }


def translate(policy: str) -> Dict[str, Any]:
    """Translate a Trust V2 *policy* into a Crew policy document, or raise."""
    rules = _with_meta(_rules(policy))
    doc = neutral_document()
    for capability, row in MANAGED.items():
        control = _control(rules, capability, row)
        if control is None:
            continue
        node = doc
        for key in row.path[:-1]:
            node = node.setdefault(key, {})
        node[row.path[-1]] = control
    return doc
