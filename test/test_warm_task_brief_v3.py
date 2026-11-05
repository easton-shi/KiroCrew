"""Warm welcome v3 (RFC ``rfc-delayed-task-brief.md`` section 3.10): the fact
registry, the claim check, one card per PR or task, and the dashboard stamp.

The anchor case is a real 16:47Z welcome (thread "Crew Mate Harness Provider
Support"): GitHub was read at 16:47 and both PRs had
conflicts and 3 failed checks, yet the model wrote "blocked waiting for your
approval ... at 11:17" and the conflict showed only in a footnote.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from kiro_crew.dashboard import warm_task_brief
from kiro_crew.dashboard import warm_task_brief_cards as cards
from kiro_crew.dashboard import warm_task_brief_claims as claims
from kiro_crew.dashboard import warm_task_brief_facts as facts
from kiro_crew.dashboard import warm_task_brief_registry as registry
from kiro_crew.dashboard.handlers import work_ledger_board

_REPO = "kirodotdev/KiroCrew"
_READ_EPOCH = 1_793_897_220.0  # 2026-11-05T16:47:00Z
_REPORT_EPOCH = _READ_EPOCH - 3600  # the worker's report, an hour before the read


def _clean(value: object, cap: int) -> str:
    return warm_task_brief._clean(value, cap)


# ------------------------------------------------------------ (a) casing


def test_lower_case_provider_values_map_to_their_words():
    words = facts.fact_words({"review": "review_required", "mergeable": "conflicting"})
    assert words == ["review required", "has conflicts"]


def test_a_prs_blocked_mergeability_is_not_the_task_word():
    assert facts.fact_words({"mergeable": "blocked"}) == ["merge blocked"]
    assert facts.fact_words({"status": "blocked"}) == ["blocked"]
    assert facts.fact_words({"mergeable": "pending", "review": "none"}) == []


# ------------------------------------------------------------ the real 16:47Z row


def _live_1647():
    """The two PR reads the 16:47Z row stored, as ``read_live_prs`` returns them."""
    checks = {"failed": 3, "pending": 0, "passed": 100, "unknown": 0}
    return tuple(
        {
            "pr": f"#{n}",
            "repo": _REPO,
            "url": f"https://github.com/{_REPO}/pull/{n}",
            "state": "open",
            "review": "review_required",
            "mergeable": "conflicting",
            "checks": dict(checks),
            "checked": True,
            "source": "GitHub",
            "read_at": "16:47",
            "_read_epoch": _READ_EPOCH,
        }
        for n in (29509, 29498)
    )


def _items_1647():
    report = "2026-11-05T15:47:00+00:00"
    return (
        {
            "item_id": "it_follow",
            "title": "Code: crewmate harness inherits agent.acp_backend when unset",
            "state": "open",
            "status": "blocked",
            "reason": "approval",
            "summary": "Could not read GitHub. The read-only gh pr view was auto-declined.",
            "acceptance": {"kind": "pr_checks", "repo": _REPO, "pr": 29509},
            "pr": 29509,
            "last_report_at": report,
            "artifacts": {},
        },
        {
            "item_id": "it_picker",
            "title": "Code: crewmate harness picker (runnable harnesses only)",
            "state": "open",
            "status": "question",
            "reason": "needs_human",
            "summary": "UX's real BLOCK: tapping an option saves it straight away.",
            "acceptance": {"kind": "pr_checks", "repo": _REPO, "pr": 29498},
            "pr": 29498,
            "last_report_at": report,
            "artifacts": {},
        },
    )


def _table_1647(asks=None):
    inp = registry.Inputs("chat-935", _items_1647(), _live_1647(), (), _READ_EPOCH)
    found = [*registry._pr_facts(inp), *registry._task_facts(inp)]
    for item in inp.items:  # nobody answered either report
        found.append(registry._item_fact(item, "waits_on_you", {"waits": True}, "tasks"))
    for item_id, options in (asks or {}).items():
        item = next(i for i in inp.items if i["item_id"] == item_id)
        found.append(registry._item_fact(item, "ask", {"options": options}, "goal_board"))
    # The gateway clock, as the time checker states it on the day.
    now = {"values": {"now": "16:47"}, "read_at": "16:47", "source": "gateway"}
    found.append({"kind": "time", "subject": "now", "label": "now", **now})
    assert all(f["_reported"] == _REPORT_EPOCH for f in found if f["kind"] == "task")
    return cards.cleaned(registry.number(found, facts.MAX_FACTS), _clean)


# The model's reply that wrote the 16:47Z row, in the v3 shape: the same words.
_MODEL_1647 = {
    "status": "needs_you",
    "headline": "Two PRs stuck on RFC status and UX review. One fix ready to push.",
    "headline_facts": [],
    "cards": [
        {
            "subject": "it_follow",
            "said": {
                "text": "PR #29509 is blocked waiting for your approval on two tool calls.",
                "facts": ["f1"],
            },
            "ask": {
                "what": "Restart the Follow Settings chat to push its bug fix",
                "options": [
                    {"label": "I'm here, restart the Follow Settings chat", "consequence": "x"}
                ],
            },
        },
        {
            "subject": "it_picker",
            "said": {"text": "Dropdown picker waits on a UX decision.", "facts": []},
        },
    ],
}


def _brief_1647(table, model=_MODEL_1647):
    parsed = warm_task_brief._parse_result(json.dumps(model))
    assert parsed is not None
    brief = facts.checked_brief(parsed, list(table), facts.live_links(_live_1647()))
    return cards.apply(brief, table, "zh", _clean)


def test_the_1647_conflict_now_shows_on_each_card_and_in_the_headline():
    table = _table_1647()
    brief = _brief_1647(table)
    by_subject = {card["subject"]: card for card in brief.cards}
    for subject, number in (("it_follow", "#29509"), ("it_picker", "#29498")):
        card = by_subject[subject]
        assert card["pr"]["label"] == number and card["pr"]["checked"] is True
        tags = {tag["text"]: tag["bad"] for tag in card["tags"]}
        # The gateway draws the fresh read as tags, in the brief's language.
        assert tags["有冲突"] is True and tags["3 项检查失败"] is True
        assert tags["待审查"] is False and "等你处理" in tags
        # The older report is replaced by the 16:47 read and says so.
        assert card["replaced"] == "较早的汇报已被 16:47 的核实取代。"
    # The model's headline cited no bad fact: the gateway's fixed one is used.
    assert brief.headline == "2 件事等你处理。2 个 PR 有冲突。2 个 PR 有检查失败。"
    # "waiting for your approval" is no claim a PR fact backs, but no fact on
    # the card contradicts it either; the 11:17 time fails the time rule below.
    assert by_subject["it_picker"]["said"]["by"] == "执行者说："
    assert brief.check == ("✓ 核实于 16:47", "GitHub · 2 个 PR", "任务列表 · 2 项")


def test_the_1647_options_line_lands_on_no_card_without_an_ask_fact():
    # Q3: the conductor's free [OPTIONS:] line is not keyed by item.
    brief = _brief_1647(_table_1647())
    assert all("ask" not in card or not card["ask"]["options"] for card in brief.cards)
    # The model's ask text stays; it has no options to copy.
    follow = next(c for c in brief.cards if c["subject"] == "it_follow")
    assert follow["ask"]["what"].startswith("Restart the Follow Settings chat")


def test_a_goal_board_ask_puts_its_options_on_its_own_card_only():
    label = "I'm here, restart the Follow Settings chat"
    brief = _brief_1647(_table_1647({"it_follow": [label]}))
    follow, picker = (
        next(c for c in brief.cards if c["subject"] == s) for s in ("it_follow", "it_picker")
    )
    assert [o["label"] for o in follow["ask"]["options"]] == [label]
    # Copied exactly: its consequence is kept.
    assert follow["ask"]["options"][0]["consequence"] == "x"
    assert "ask" not in picker


def test_the_1647_time_only_the_transcript_held_is_a_grounding_miss():
    table = _table_1647()
    clocks = {"17:00"} | warm_task_brief._times_in("\n".join(registry.time_texts(table)))
    assert "16:47" in clocks and "11:17" not in clocks
    model = json.loads(json.dumps(_MODEL_1647))
    model["cards"][0]["said"]["text"] = "The chat made two tool calls at 11:17."
    brief = _brief_1647(table, model)
    assert "11:17" in brief.cards[0]["said"]["text"]
    grounding = "transcript: the chat made two tool calls at 11:17"
    content = warm_task_brief._brief_text(brief, "zh")
    assert warm_task_brief._grounding_miss(brief, content, {29509, 29498}, grounding, clocks)
    # The miss keeps the gateway's cards and tags, and drops every model word.
    kept = warm_task_brief._gateway_only(brief, "zh")
    assert kept.headline == warm_task_brief._FAILURE_TEXT["zh"]
    assert all("said" not in c for c in kept.cards)
    assert any(t["text"] == "有冲突" for c in kept.cards for t in c["tags"])


def test_a_headline_citing_a_bad_fact_stays():
    table = _table_1647()
    model = dict(_MODEL_1647, headline="两个 PR 都有冲突。", headline_facts=["f1"])
    assert _brief_1647(table, model).headline == "两个 PR 都有冲突。"


# ------------------------------------------------------------ (c) claim check


def test_a_sentence_that_contradicts_its_card_is_dropped():
    table = _table_1647()
    model = json.loads(json.dumps(_MODEL_1647))
    model["cards"][0]["said"]["text"] = "PR #29509 has no conflicts and is ready to merge."
    follow = next(c for c in _brief_1647(table, model).cards if c["subject"] == "it_follow")
    assert "said" not in follow


@pytest.mark.parametrize(
    ("text", "hit"),
    [
        ("All green, ready to merge.", True),
        ("没有冲突", True),
        ("Not mergeable yet.", False),
        ("greenhouse checks", False),
    ],
)
def test_the_claim_check_is_an_exact_lookup_by_value(text, hit):
    fact = {"kind": "pr", "values": {"mergeable": "conflicting", "checks": {"failed": 1}}}
    assert claims.contradicts(text, [fact]) is hit


def test_no_pending_approval_forbids_waiting_for_your_approval():
    fact = {"kind": "approval", "values": {"pending": 0}}
    assert claims.contradicts("Parked, waiting for your approval.", [fact])
    assert not claims.contradicts(
        "Parked, waiting for your approval.", [{**fact, "values": {"pending": 1}}]
    )


# ------------------------------------------------------------ (b) registry


def test_bad_facts_are_marked():
    table = _table_1647()
    assert [f["id"] for f in table if f["bad"]] == ["f1", "f2"]
    assert registry.is_bad({"kind": "chat", "values": {"state": "closed"}})
    assert not registry.is_bad({"kind": "chat", "values": {"state": "idle"}})


def test_options_lines_are_no_ask_source_and_cards_number_as_the_prompt_does():
    waiting = (
        {"kind": "options", "text": "Pick", "options": ["A", "B"]},
        {"kind": "question", "text": "Which layout?", "options": ["Grid", "List"]},
    )
    inp = registry.Inputs("chat-x", (), (), waiting, 0.0)
    found = registry._ask_facts(inp)
    assert [(f["subject"], f["values"]["options"]) for f in found] == [("w0", ["Grid", "List"])]


def test_the_file_checker_stats_only_allowed_paths_in_the_workspace(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    (root / "report.md").write_text("never read")
    monkeypatch.setattr("kiro_crew.config.loader.workspace_root", lambda create=True: root)
    artifacts = {f"a{n}": str(root / f"missing{n}.md") for n in range(12)}
    artifacts |= {"report": str(root / "report.md"), "out": str(tmp_path / "outside.md")}
    item = {
        "item_id": "it_1",
        "title": "t",
        "artifacts": {"report": artifacts.pop("report"), **artifacts},
    }
    found = registry._file_facts(registry.Inputs("chat-x", (item,), now=0.0))
    assert len(found) == registry.MAX_FILE_STATS
    first = found[0]
    assert first["values"]["exists"] is True and first["values"]["size"] == len("never read")
    assert "contents" not in first["values"]
    assert all(os.path.realpath(str(root)) in f["subject"] for f in found)


def test_a_sensitive_path_is_never_statted(tmp_path, monkeypatch):
    monkeypatch.setattr("kiro_crew.config.loader.workspace_root", lambda create=True: tmp_path)
    monkeypatch.setattr("kiro_crew.security.is_sensitive_path", lambda path: True)
    item = {"item_id": "it_1", "artifacts": {"key": str(tmp_path / ".env")}}
    assert registry._file_facts(registry.Inputs("chat-x", (item,), now=0.0)) == []


def test_a_checker_that_raises_yields_nothing_for_its_kind():
    def broken(inp):
        raise RuntimeError("boom")

    assert registry._run(broken, registry.Inputs("chat-x")) == []


def test_a_wake_file_check_is_preferred_over_an_uncited_stat():
    table = [
        {"id": "f1", "kind": "file", "source": "stat"},
        {"id": "f2", "kind": "file", "source": "stat"},
        {"id": "f3", "kind": "file", "source": "wake"},
    ]
    assert [f["id"] for f in registry.prefer_wake_files(table, {"f2"})] == ["f2", "f3"]
    assert registry.prefer_wake_files(table[:2], set()) == table[:2]


def test_chat_state_reports_interrupted_only_when_the_slot_shows_it(monkeypatch):
    slot = type("S", (), {"messages": [], "window_pending": False})()
    monkeypatch.setattr(work_ledger_board, "_alive_of", lambda state, key: "idle")
    monkeypatch.setattr(work_ledger_board, "worker_slot", lambda state, key: slot)
    monkeypatch.setattr("kiro_crew.dashboard.state.is_turn_interrupted", lambda rows: True)
    assert registry.chat_state(object(), "chat-w") == "interrupted"
    monkeypatch.setattr(work_ledger_board, "_alive_of", lambda state, key: "running")
    assert registry.chat_state(object(), "chat-w") == "running"


# ------------------------------------------------------------ (e) dashboard


def test_the_board_stamps_item_rows_from_the_cache_only(monkeypatch):
    table = _table_1647()
    monkeypatch.setattr(registry, "_cache", {})
    registry.remember("chat-935", table)  # gathered just now
    monkeypatch.setattr(work_ledger_board, "_slot_open", lambda state, key: True)
    monkeypatch.setattr(work_ledger_board, "_slot_running", lambda state, key: False)
    monkeypatch.setattr(work_ledger_board, "_slot_closed", lambda state, key: False)
    monkeypatch.setattr(work_ledger_board, "_alive_of", lambda state, key: "idle")
    row = {"item_id": "it_follow", "worker_session_key": "", "state": "open", "status": "blocked",
           "pr": 29509, "acceptance": {"repo": _REPO, "pr": 29509}}  # fmt: skip
    (stamped,) = work_ledger_board.with_worker_facts([row], object(), "chat-935")
    assert stamped["facts"]["pr"]["mergeable"] == "conflicting"
    assert stamped["facts"]["pr"]["bad"] is True
    (other,) = work_ledger_board.with_worker_facts([row], object(), "chat-unknown")
    assert "facts" not in other


def test_the_brief_meta_is_version_3_with_no_old_sections():
    meta = _brief_1647(_table_1647()).as_meta()
    assert meta["version"] == 3 and meta["status_label"] == "等你处理"
    assert not {"waiting", "asks", "lines", "asked"} & set(meta)
    assert replace(_brief_1647(_table_1647()), more=3).as_meta()["more"] == "还有 3 项"


def test_a_rejected_headline_is_never_posted():
    pr = {"id": "f1", "kind": "pr", "values": {"review": "changes_requested"}, "bad": True}
    # Contradicts its cited fact: the gateway's own count replaces it.
    assert cards.headline("PR #12 is approved.", ["f1"], [pr], "en") == (
        "Changes requested on PRs: 1."
    )
    merged = {"id": "f1", "kind": "pr", "values": {"state": "merged"}, "bad": False}
    # Nothing bad to count: a plain fixed line, never the rejected text.
    assert cards.headline("It is still open.", ["f1"], [merged], "en") == (
        "Here is where things stand."
    )


def test_coordinator_approvals_are_read_from_the_state_registry(monkeypatch):
    class _Future:
        def done(self):
            return False

    slot = type("S", (), {"key": "chat-w", "_approval_futures": {"a": _Future()}})()
    state = type("St", (), {"pending_coordinator_approvals": lambda self, key: [{"slot": key}]})()
    item = {"item_id": "it_1", "title": "t", "worker_session_key": "chat-w"}
    inp = registry.Inputs("chat-x", (item,), state=state)
    monkeypatch.setattr(work_ledger_board, "worker_slot", lambda st, key: slot)
    (fact,) = registry._approval_facts(inp)
    assert fact["values"] == {"pending": 2}


def test_a_closed_chat_on_a_task_reported_done_is_not_bad():
    closed = {"kind": "chat", "values": {"state": "closed"}}
    assert registry.is_bad({**closed, "_item_status": "blocked"})
    assert not registry.is_bad({**closed, "_item_status": "done"})


def test_a_task_closed_while_away_asks_nothing(monkeypatch):
    item = {
        "item_id": "it_gone",
        "title": "Dropped work",
        "state": "abandoned",
        "status": "blocked",
        "reason": "needs_human",
        "worker_session_key": "chat-w",
    }
    inp = registry.Inputs("chat-x", (item,), state=object())
    assert work_ledger_board.item_awaits_person("chat-x", item) is False
    assert registry._waits_facts(inp) == []
    assert registry._chat_facts(inp) == [] and registry._approval_facts(inp) == []
    table = registry.number(registry._task_facts(inp), facts.MAX_FACTS)
    built, _ = cards.build(table, (), "en", _clean)
    assert built == []


def test_a_remote_worker_is_never_read_as_a_local_chat():
    item = {
        "item_id": "it_r",
        "state": "open",
        "worker_session_key": "chat-r",
        "worker_instance_id": "peer",
    }
    inp = registry.Inputs("chat-x", (item,), state=object())
    assert registry._chat_facts(inp) == [] and registry._approval_facts(inp) == []


def test_an_unfinished_check_read_is_tagged_never_clear():
    fact = {"id": "f1", "kind": "pr", "values": {"state": "open", "checks": "not fully read"}}
    assert [t["text"] for t in cards.tags([fact], "en")] == ["checks not fully read"]


def test_the_board_cache_expires(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    registry.remember("chat-x", _table_1647(), at=0.0)
    assert registry.item_facts("chat-x", "it_follow", f"{_REPO}#29509") is None


def test_a_closed_task_keeps_no_ask_the_model_wrote():
    item = {"item_id": "it_done", "title": "Shipped", "state": "accepted", "status": "done"}
    inp = registry.Inputs("chat-x", (item,))
    table = registry.number(registry._task_facts(inp), facts.MAX_FACTS)
    model = cards.raw_cards(
        [{"subject": "it_done", "said": {"text": "Shipped.", "facts": ["f1"]},
          "ask": {"what": "Approve the release"}}],
        _clean,
    )  # fmt: skip
    (card,) = cards.build(table, model, "en", _clean)[0]
    assert "ask" not in card and card["said"]["text"] == "Shipped."
    assert not cards.owed([card], table)


def test_the_board_shows_no_cached_chat_or_approval(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    item = {"item_id": "it_1", "title": "t", "state": "open"}
    inp = registry.Inputs("chat-x", (item,))
    chat = registry._item_fact(item, "chat", {"state": "interrupted"}, "chat")
    approval = registry._item_fact(item, "approval", {"pending": 1}, "chat")
    registry.remember("chat-x", registry.number([*registry._task_facts(inp), chat, approval], 9))
    # No PR read: nothing is stamped, not even a "checked" time.
    assert registry.item_facts("chat-x", "it_1", "") is None


def test_the_board_stamps_the_prs_own_read_time_and_never_an_unread_pr(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    live = list(_live_1647())
    live[1] = {"pr": "#29498", "repo": _REPO, "checked": False, "not_checked_since": "16:30"}
    inp = registry.Inputs("chat-x", _items_1647(), tuple(live), (), 0.0)
    registry.remember(
        "chat-x", registry.number([*registry._pr_facts(inp), *registry._task_facts(inp)], 9)
    )
    assert registry.item_facts("chat-x", "it_follow", f"{_REPO}#29509")["checked_at"] == "16:47"
    assert registry.item_facts("chat-x", "it_picker", f"{_REPO}#29498") is None


def test_the_board_gives_a_replaced_pr_no_old_tags(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    registry.remember("chat-x", _table_1647())
    assert registry.item_facts("chat-x", "it_follow", f"{_REPO}#29509") is not None
    # The item now holds another PR: the cached read was of the old one.
    assert registry.item_facts("chat-x", "it_follow", f"{_REPO}#31000") is None


def test_a_goal_board_ask_counts_only_while_the_item_waits(monkeypatch):
    item = {"item_id": "it_1", "title": "t", "state": "open", "status": "progress"}
    monkeypatch.setattr(
        work_ledger_board,
        "_conductor_asks",
        lambda slot, ids: {"it_1": ["A"]} if "it_1" in ids else {},
    )
    inp = registry.Inputs("chat-x", (item,))
    # Answered (the worker moved on): the stored entry is no ask any more.
    assert registry._ask_facts(inp) == []
    asking = {**item, "status": "question", "reason": "needs_human"}
    (fact,) = registry._ask_facts(registry.Inputs("chat-x", (asking,)))
    assert fact["values"] == {"options": ["A"]}


def test_a_closed_chat_after_a_done_report_is_no_red_tag():
    item = {"item_id": "it_1", "title": "t", "state": "open", "status": "done"}
    chat = {
        **registry._item_fact(item, "chat", {"state": "closed"}, "chat"),
        "_item_status": "done",
    }
    (fact,) = registry.number([chat], 9)
    assert [(t["text"], t["bad"]) for t in cards.tags([fact], "en")] == [("chat · closed", False)]


def test_a_skipped_task_leaves_its_pr_its_own_card():
    live = ({"pr": "#12", "repo": _REPO, "checked": True, "state": "merged", "read_at": "16:47"},)
    item = {"item_id": "it_1", "title": "Ship", "state": "open", "status": "progress",
            "acceptance": {"repo": _REPO, "pr": 12}, "pr": 12}  # fmt: skip
    inp = registry.Inputs("chat-x", (item,), live)
    table = registry.number([*registry._pr_facts(inp), *registry._task_facts(inp)], 9)
    # No card for the quiet task, so the merged PR keeps its own.
    (card,) = cards.build(table, (), "en", _clean)[0]
    assert card["subject"] == f"{_REPO}#12" and "merged" in [t["text"] for t in card["tags"]]
    # A sentence the model keyed by the PR lands on the item's card, once.
    said = [{"subject": f"{_REPO}#12", "said": {"text": "It merged.", "facts": ["f1"]}}]
    (card,) = cards.build(table, cards.raw_cards(said, _clean), "en", _clean)[0]
    assert card["subject"] == "it_1" and card["said"]["text"] == "It merged."


def test_an_approval_answered_during_the_model_call_is_read_again(monkeypatch):
    item = {"item_id": "it_1", "title": "t", "state": "open", "worker_session_key": "chat-w"}
    pending = registry._item_fact(item, "approval", {"pending": 1}, "chat")
    (fact,) = registry.number([pending], 9)
    assert fact["bad"]
    monkeypatch.setattr(
        registry,
        "ON_LOOP",
        (lambda inp: [registry._item_fact(item, "approval", {"pending": 0}, "chat")],),
    )
    (now,) = registry.refreshed([fact], registry.Inputs("chat-x", (item,)))
    assert now["id"] == "f1" and now["values"] == {"pending": 0} and not now["bad"]


def test_a_relative_artifact_path_is_never_statted(tmp_path, monkeypatch):
    monkeypatch.setattr("kiro_crew.config.loader.workspace_root", lambda create=True: tmp_path)
    assert registry._stat_path("notes/report.md", os.path.realpath(str(tmp_path))) is None
    inside = str(tmp_path / "report.md")
    assert registry._stat_path(inside, os.path.realpath(str(tmp_path))) == os.path.realpath(inside)


def test_a_consequence_that_contradicts_the_card_drops_the_models_guidance():
    pr = {"id": "f1", "kind": "pr", "values": {"mergeable": "conflicting"}, "bad": True}
    model = {"what": "Pick one", "why": "", "recommended": "Ship",
             "recommended_why": "No conflicts, so it is safe",
             "options": [{"label": "Ship", "consequence": "It merges"}]}  # fmt: skip
    ask = cards._ask(model, ["Ship"], [pr], _clean)
    assert ask["options"] == [{"label": "Ship", "consequence": ""}]
    assert ask["what"] == "" and ask["recommended_why"] == ""


def test_a_worker_chat_closed_during_the_call_leaves_no_approval_waiting(monkeypatch):
    item = {"item_id": "it_1", "title": "t", "state": "open", "worker_session_key": "chat-w"}
    (fact,) = registry.number([registry._item_fact(item, "approval", {"pending": 2}, "chat")], 9)
    # The slot is gone: the approval checker finds nothing to report.
    monkeypatch.setattr(registry, "ON_LOOP", (lambda inp: [],))
    (now,) = registry.refreshed([fact], registry.Inputs("chat-x", (item,)))
    assert now["values"] == {"pending": 0} and not now["bad"]


def test_a_headline_is_checked_against_every_fact_not_only_its_citations():
    conflict = {"id": "f1", "kind": "pr", "values": {"mergeable": "conflicting"}, "bad": True}
    failed = {"id": "f2", "kind": "pr", "values": {"checks": {"failed": 3}}, "bad": True}
    text = "One PR has conflicts, but all checks passed."
    got = cards.headline(text, ["f1"], [conflict, failed], "en")
    assert got != text and "PRs with failed checks: 1." in got
