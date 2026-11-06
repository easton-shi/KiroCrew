"""Which skills a KAS session may load: Crew's skill scope as ``skill`` deny rules.

kiro-cli (v2) and KAS (v3) build a session's skill catalog differently. On kiro-cli
Crew's native launch turns ``chat.disableInheritingDefaultResources`` on, so the
session loads only the skills its spec maps (or the inherited trees, served by Crew's
``skill_search``). KAS reads no
``cli.json``: it loads every skill under ``~/.kiro/skills`` and the workspace's
``.kiro/skills`` on top of the injected agent's ``resources`` (measured on kiro-cli
2.35.1: a spec mapping 3 skills ran with 45 on KAS and with 3 through Crew's v2 path).

The ``customAgents`` wire has no field that turns that off, but its ``permissions``
does reach the catalog: KAS drops every skill the policy denies under the ``skill``
capability, keyed on the skill's name, from the list its skill tool offers. So the
v2 scope is re-expressed as one deny rule: ``deny skill`` with the in-scope names as
``exclude``. Measured on the same spec, the rule took the session from 45 skills to
the spec's 3. A deny only narrows, so it needs no pass through the governance ceiling.

KAS ANDs ``exclude`` patterns onto every statement it compiles and does not split
them; its compiler records that a condition tree past roughly 114 patterns overflows
cedar-wasm's stack, and Crew bounds every rule it sends at
``kas_permissions._MAX_PATTERNS_PER_RULE`` (64) for that reason (measured live here
up to 35). Past that bound the rule is written the other way round: one ``deny`` per
out-of-scope name found in the trees KAS loads by default, which splits like any
``match`` list. That form cannot see a skill added to those trees after session
start, and a name with glob characters cannot be denied literally, so such a skill
stays offered.

The scope is read from the spec the session is projected with (``skill://``
entries, else the inherited trees for a spec that may inherit them) -- the rules
``agent_discovery.agent_skill_globs`` applies on kiro-cli. Skill files are read
through ``hooks.safe_read_prefix``.

KAS also refuses a SKILL.md kiro-cli loads (:func:`refused_skill_copies`): one with no
front matter, no ``name`` or ``description``, or front matter its YAML parser rejects,
and, in its default trees, a name outside ``[a-z0-9-]`` or a description over 1024
characters. kiro-cli falls back to the path and "No description available"; KAS drops
the skill without a word. For each in-scope skill KAS would drop, Crew writes a fixed
copy under the data home and adds it to the agent's ``resources`` as ``skill://``.

See ``docs/system-specs/modules/memory-skills-hooks.md`` (KAS skill scope).
"""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

from kiro_crew.acp.kas_permissions import _MAX_PATTERN_LENGTH, _MAX_PATTERNS_PER_RULE

logger = logging.getLogger(__name__)

#: KAS's capability for a skill (``CAPABILITY_BY_TOOL["disclose_context"]``).
SKILL_CAPABILITY = "skill"

#: How many skill files one scope resolution reads, at most.
_MAX_SKILL_FILES = 2048

#: How much of a skill file is read for its frontmatter.
_FRONTMATTER_BYTES = 64 * 1024

#: Characters KAS's matcher reads as a pattern; a name carrying one cannot be an
#: exact ``exclude`` entry, so it is left out (left denied) rather than widened.
_PATTERN_CHARS = frozenset("*?[]{}")


def _skill_name(path: Path) -> str | None:
    """The skill's frontmatter ``name``, else its directory's name. KAS has no such
    fallback and drops a skill without a name; :func:`refused_skill_copies` gives
    that skill a copy KAS loads."""
    from kiro_crew.frontmatter import SKILL_LOADER, parse_frontmatter
    from kiro_crew.hooks import safe_read_prefix

    # A ``skill://`` path comes from a spec, which may point anywhere: the read
    # goes through the sensitive-path fence and refuses a credential store.
    raw = safe_read_prefix(str(path), _FRONTMATTER_BYTES)
    if raw is None:
        return None
    head = raw.decode("utf-8", errors="replace")
    name = (parse_frontmatter(head, SKILL_LOADER).get("name") or "").strip()
    if not name:
        name = path.parent.name if path.name == "SKILL.md" else path.stem
    return name or None


def _literal(name: str) -> bool:
    """Whether *name* can be a policy pattern that matches only itself."""
    return len(name) <= _MAX_PATTERN_LENGTH and not (_PATTERN_CHARS & set(name))


def skill_names(globs: list[str]) -> set[str]:
    """The names of the skill files *globs* match, bounded by :data:`_MAX_SKILL_FILES`."""
    names: set[str] = set()
    seen = 0
    for pattern in globs:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if not path.is_file():
                continue
            seen += 1
            if seen > _MAX_SKILL_FILES:
                logger.warning(
                    "KAS skill scope: more than %d skill files matched; the rest are left out",
                    _MAX_SKILL_FILES,
                )
                return names
            name = _skill_name(path)
            if name:
                names.add(name)
    return names


def _default_tree_globs(work_dir: str | Path | None) -> list[str]:
    """The skill trees KAS loads for every session, whatever the agent maps."""
    from kiro_crew.config import paths
    from kiro_crew.skills import project_skill_tree_present

    globs = [str(paths.kiro_home() / "skills" / "*" / "SKILL.md")]
    if work_dir and project_skill_tree_present(work_dir):
        globs.append(str(Path(work_dir) / ".kiro" / "skills" / "*" / "SKILL.md"))
    return globs


def _spec_globs(
    agent: str, spec: dict[str, Any], spec_path: Path, work_dir: Any
) -> list[str] | None:
    """The skill globs *spec* resolves to under ``agent_skill_globs``' rules, or
    ``None`` for an unmapped whole-catalog template.

    Its ``skill://`` entries, else the inherited trees for a spec that may inherit
    them (kiro-cli's rule, so inherited trees count); for the crewmate's conductor,
    its entries plus those trees.
    """
    from kiro_crew.agent_discovery import (
        WHOLE_CATALOG_TEMPLATES,
        can_inherit_skills,
        expand_skill_uri,
        inherited_skill_globs,
        mate_conductor_scope,
        skill_resource_uris,
    )

    globs = [
        g
        for uri in skill_resource_uris(spec)
        if (g := expand_skill_uri(uri, spec_path, project_dir=work_dir))
    ]
    if globs:
        # The crewmate conductor's mapping adds to the inherited trees, by the one
        # rule ``agent_skill_globs`` returns through.
        return mate_conductor_scope(agent, spec, globs, work_dir)
    if agent in WHOLE_CATALOG_TEMPLATES:
        return None
    return inherited_skill_globs(work_dir) if can_inherit_skills(spec) else []


def scope_globs(
    agent: str,
    spec: dict[str, Any],
    *,
    agents_dir: Path,
    work_dir: str | Path | None,
) -> list[str] | None:
    """The skill globs this session loads on kiro-cli, or ``None`` when KAS needs no rule.

    ``None`` is the whole catalog (an unmapped template in ``WHOLE_CATALOG_TEMPLATES``:
    ``kirocrew``, ``kirocrew-worker`` and the crewmate's conductor and worker)
    or a scope that is exactly the default trees KAS loads by itself.

    The scope is read from *spec*, the spec this session is projected with, and
    from nothing else. Never re-resolved by name: a checkout spec that shadows a
    user-level name is not the one KAS runs (``harness/kas.resolve_projected_spec``).
    Never from the session key's skill-scope pin (``skill_scope_pin``) either: the
    gateway sets that pin per turn, after the session is built, so a session
    started for a turn that switched away from an app agent would still read the
    app's file. An app-bound session's projected spec IS its registered file
    (``apps.agent_launch.read_bound_app_spec``), and a crewmate worker's is already
    its merged session spec.
    """
    globs = _spec_globs(agent, spec, agents_dir / f"{agent}.json", work_dir)
    if globs is None or _inherits_default_trees(agent, spec, globs, work_dir):
        return None
    return globs


def _inherits_default_trees(
    agent: str, spec: dict[str, Any], globs: list[str], work_dir: str | Path | None
) -> bool:
    """Whether the scope IS the default trees, which KAS loads on its own.

    Asked on the trees, not on the glob list alone: with neither tree on disk an
    inheriting agent resolves to ``[]``, the same answer as an agent that may not
    inherit, and only the second is held to no skills.
    """
    from kiro_crew.agent_discovery import (
        WHOLE_CATALOG_TEMPLATES,
        can_inherit_skills,
        inherited_skill_globs,
        skill_resource_uris,
    )
    from kiro_crew.agent_sdk.drivers import acp as acp_driver

    if globs:
        return sorted(globs) == sorted(inherited_skill_globs(work_dir))
    return (
        agent not in WHOLE_CATALOG_TEMPLATES
        and not skill_resource_uris(spec)
        and can_inherit_skills(spec)
        and acp_driver.inherits_default_resources(work_dir)
    )


def skill_scope_rules(
    globs: list[str] | None,
    *,
    work_dir: str | Path | None = None,
    also_allowed: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """The ``permissions`` rules that hold a KAS session to the skills *globs* match.

    ``None`` (:func:`scope_globs`: a whole-catalog template, or a scope that is the
    default trees KAS already loads) adds nothing. An empty scope denies every skill.
    *also_allowed* are names in scope that *globs* do not show: the copies
    :func:`refused_skill_copies` made.
    """
    if globs is None:
        return []
    names = sorted(n for n in skill_names(globs) | also_allowed if _literal(n))
    if not names:
        return [{"capability": SKILL_CAPABILITY, "effect": "deny"}]
    if len(names) <= _MAX_PATTERNS_PER_RULE:
        return [{"capability": SKILL_CAPABILITY, "exclude": names, "effect": "deny"}]
    in_scope = set(names)
    # Only literal names become ``match`` entries: KAS reads them as globs, so a
    # name like ``*`` would deny the in-scope skills too. Such a skill stays offered.
    outside = sorted(
        n for n in skill_names(_default_tree_globs(work_dir)) if n not in in_scope and _literal(n)
    )
    logger.info(
        "KAS skill scope maps %d skills, over the %d an exclude list may hold; "
        "denying the %d default-tree skills outside it by name instead",
        len(names),
        _MAX_PATTERNS_PER_RULE,
        len(outside),
    )
    step = _MAX_PATTERNS_PER_RULE
    return [
        {"capability": SKILL_CAPABILITY, "match": outside[i : i + step], "effect": "deny"}
        for i in range(0, len(outside), step)
    ]


# --- Skills KAS refuses (P0-26) -------------------------------------------------------

#: KAS's ``isValidSkillName``: 1-64 of ``a-z0-9-``, no edge or double hyphen.
_KAS_NAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")
_MAX_NAME_CHARS = 64
#: KAS's ``isValidSkillDescription`` bound, in UTF-16 units (JavaScript's length).
_MAX_DESCRIPTION_UNITS = 1024
#: kiro-cli's text for a skill with no description (``format_skill_hint``).
_NO_DESCRIPTION = "No description available"
#: The largest skill file Crew copies; KAS reads skill files whole.
# shortcut: a SKILL.md over 1 MiB gets no copy and stays dropped; raise if one is seen.
_MAX_COPY_BYTES = 1024 * 1024
#: Where the copies live, under the data home (``atomic_write`` makes it 0700).
COPY_DIR_NAME = "kas-skills"


class SkillCopy(NamedTuple):
    """A copy of a skill KAS would drop: the name KAS gives it and its ``skill://`` URI."""

    name: str
    uri: str


@functools.cache
def _kas_yaml_loader() -> type:
    """A YAML loader closer to js-yaml's: it refuses a duplicate key, and reads
    ``1e3`` as a number (YAML 1.2), as KAS does."""
    import yaml  # type: ignore[import-untyped]

    from kiro_crew.yaml_safe import NoAliasSafeLoader

    class _Loader(NoAliasSafeLoader):  # type: ignore[misc]
        def construct_mapping(self, node: Any, deep: bool = False) -> Any:
            seen: set[Any] = set()
            for key_node, _ in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in seen:
                    raise yaml.constructor.ConstructorError(
                        None, None, "duplicated mapping key", key_node.start_mark
                    )
                seen.add(key)
            return super().construct_mapping(node, deep=deep)

    _Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?$"),
        list("-+0123456789."),
    )
    return _Loader


def _kas_front_matter(text: str) -> tuple[dict[str, Any] | None, str]:
    """KAS's reading of *text*: ``(front matter, body)``, the front matter ``None``
    when KAS finds none or refuses it.

    Follows ``utils/front-matter`` (``extractFrontMatter`` and ``parseFrontMatter``
    with ``SkillFrontMatterSchema``): an opening ``---`` not followed by a fourth
    ``-``, an optional language, a closing ``\\n---``; one YAML (or JSON) document
    that is a non-empty mapping whose ``name``, ``description``, ``license`` and
    ``compatibility``, when set, are strings and whose ``metadata`` is a mapping.
    The body is the text after the closing fence, or the whole text with none.
    """
    import yaml  # type: ignore[import-untyped]

    text = text.removeprefix("\ufeff")
    if not text.startswith("---") or text[3:4] == "-":
        return None, text
    rest = text[3:]
    newline = re.search(r"\r?\n", rest)
    first = rest[: newline.start()] if newline else rest
    language = first.strip().lower()
    if language:
        rest = rest[len(first) :]
    close = rest.find("\n---")
    matter = rest if close == -1 else rest[:close]
    # No closing fence: KAS reads the whole text as front matter. The body is kept
    # whole for a copy, so a leading Markdown rule cannot erase the skill.
    body = text if close == -1 else rest[close + 4 :].removeprefix("\r").removeprefix("\n")
    if not re.sub(r"(?m)^\s*#[^\n]+", "", matter).strip():
        return None, body
    try:
        if language == "json":
            data = json.loads(matter)
        elif language in ("", "yaml", "yml"):
            # shortcut: PyYAML reads YAML 1.1 (``yes`` is a bool) where js-yaml's
            # CORE schema reads 1.2, and aliases are refused here; such a skill gets
            # a copy KAS did not need. Swap in a 1.2 reader if one is reported.
            docs = list(yaml.load_all(matter, Loader=_kas_yaml_loader()))  # noqa: S506
            if len(docs) > 1:
                return None, body
            data = docs[0] if docs else None
        else:
            return None, body
    except (ValueError, TypeError, yaml.YAMLError):
        return None, body
    if not isinstance(data, dict) or not data:
        return None, body
    if any(
        k in data and not isinstance(data[k], str)
        for k in ("name", "description", "license", "compatibility")
    ) or ("metadata" in data and not isinstance(data["metadata"], dict)):
        return None, body
    return data, body


def _utf16_len(text: str) -> int:
    return sum(2 if ord(c) > 0xFFFF else 1 for c in text)


def _cut_utf16(text: str, units: int) -> str:
    used = 0
    for i, c in enumerate(text):
        used += 2 if ord(c) > 0xFFFF else 1
        if used > units:
            return text[:i]
    return text


def _kas_loads(front: dict[str, Any] | None, *, native: bool) -> bool:
    """Whether KAS loads a skill with *front* matter: as a ``skill://`` resource
    (``resource-resolver.readSkill``) or, *native*, from a default tree
    (``validateSkillMd``, which also checks the name and description)."""
    if front is None or not front.get("name") or not front.get("description"):
        return False
    if not native:
        return True
    name, description = front["name"], front["description"]
    return (
        len(name) <= _MAX_NAME_CHARS
        and _KAS_NAME_RE.match(name) is not None
        and "--" not in name
        and _utf16_len(description) <= _MAX_DESCRIPTION_UNITS
    )


def _kas_name(raw: str) -> str:
    """*raw* as a name KAS accepts, or ``""``."""
    return re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")[:_MAX_NAME_CHARS].rstrip("-")


def _free_name(base: str, taken: set[str]) -> str:
    """*base*, or *base* with the first ``-N`` suffix no skill in *taken* uses."""
    name, n = base, 1
    while name in taken:
        n += 1
        suffix = f"-{n}"
        name = base[: _MAX_NAME_CHARS - len(suffix)].rstrip("-") + suffix
    return name


def _copy_text(path: Path, front: dict[str, Any] | None, body: str, name: str) -> str:
    """The copy of the skill at *path*: valid front matter, a note, the original body."""
    import yaml  # type: ignore[import-untyped]

    description = (front or {}).get("description") or ""
    if not description:
        first = next((ln.strip().lstrip("#").strip() for ln in body.splitlines() if ln.strip()), "")
        description = first or _NO_DESCRIPTION
    description = _cut_utf16(description, _MAX_DESCRIPTION_UNITS).strip() or _NO_DESCRIPTION
    head = yaml.safe_dump(
        {"name": name, "description": description},
        sort_keys=False,
        allow_unicode=True,
        width=1 << 30,
    )
    note = (
        f"_Kiro Crew copied this skill from `{path}` so KAS can load it. "
        f"Relative paths in it are relative to `{path.parent}`._"
    )
    return f"---\n{head}---\n{note}\n\n{body}"


def _resource_globs(spec: dict[str, Any], spec_path: Path, work_dir: Any) -> list[str]:
    """The paths of *spec*'s ``skill://`` entries, which KAS reads as resources."""
    from kiro_crew.agent_discovery import expand_skill_uri, skill_resource_uris

    return [
        g
        for uri in skill_resource_uris(spec)
        if (g := expand_skill_uri(uri, spec_path, project_dir=work_dir))
    ]


def _matched_files(globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in globs:
        for match in sorted(glob.glob(pattern, recursive=True)):
            path = Path(match)
            if path.is_file() and path not in files:
                files.append(path)
                if len(files) >= _MAX_SKILL_FILES:
                    return files
    return files


def refused_skill_copies(
    agent: str,
    spec: dict[str, Any],
    globs: list[str] | None,
    *,
    agents_dir: Path,
    work_dir: str | Path | None,
) -> list[SkillCopy]:
    """Copies of the in-scope skills KAS would drop, written for it to load.

    In scope is what *globs* (:func:`scope_globs`) match, or every default-tree
    skill when *globs* is ``None``. A skill is dropped when no way KAS reaches it
    loads it: the default trees (strict check) and the spec's ``skill://`` entries,
    forwarded as they are (front matter, name and description only). The copy gets
    a name KAS accepts, unused by any skill KAS loads, and a description: the
    original's, else the first line of the body, else kiro-cli's
    "No description available", cut to 1024. Each copy is rewritten at every
    session start, so it follows the original. A skill that cannot be read or
    written gets no copy and stays dropped, as before.
    """
    from kiro_crew.atomic_write import atomic_write
    from kiro_crew.config import paths
    from kiro_crew.hooks import safe_read_prefix

    native = set(_matched_files(_default_tree_globs(work_dir)))
    resource = set(_matched_files(_resource_globs(spec, agents_dir / f"{agent}.json", work_dir)))
    in_scope = sorted(native) if globs is None else _matched_files(globs)
    taken: set[str] = set()
    refused: list[tuple[Path, dict[str, Any] | None, str]] = []
    for path in sorted(native | resource | set(in_scope)):
        raw = safe_read_prefix(str(path), _MAX_COPY_BYTES + 1)
        if raw is None:
            continue
        front, body = _kas_front_matter(raw.decode("utf-8", errors="replace"))
        loads = (path in native and _kas_loads(front, native=True)) or (
            path in resource and _kas_loads(front, native=False)
        )
        if loads:
            taken.add(str((front or {})["name"]))
        elif path in in_scope and len(raw) <= _MAX_COPY_BYTES:
            refused.append((path, front, body))
    copies: list[SkillCopy] = []
    root = paths.data_home() / COPY_DIR_NAME
    for path, front, body in refused:
        base = (
            _kas_name(str((front or {}).get("name") or ""))
            or _kas_name(path.parent.name if path.name == "SKILL.md" else path.stem)
            or "skill"
        )
        name = _free_name(base, taken)
        # Keyed by the name too: another session may give the same skill another
        # name, and must not rewrite this session's copy.
        key = f"{path.resolve()}\0{name}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        target = root / digest / "SKILL.md"
        try:
            atomic_write(target, _copy_text(path, front, body, name), newline="")
        except OSError:
            logger.warning("KAS skill copy of %s could not be written", path, exc_info=True)
            continue
        taken.add(name)
        copies.append(SkillCopy(name, f"skill://{target}"))
    if copies:
        logger.info(
            "KAS would drop %d skill(s) kiro-cli loads; loading fixed copies: %s",
            len(copies),
            ", ".join(c.name for c in copies),
        )
    return copies
