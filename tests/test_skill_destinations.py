"""The /transcribe skill's Destinations step: exact commands, the explicit-yes question, next steps."""

import re
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parent.parent / "skills/transcribe/SKILL.md"


@pytest.fixture(scope="module")
def skill() -> str:
    return SKILL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def destinations(skill: str) -> str:
    return skill.split("\n## Destinations\n", 1)[1]


@pytest.mark.parametrize(
    "command",
    [
        "uv run --extra asr scribe config show",
        'uv run --extra asr scribe config set vault "<vault path>"',
        'uv run --extra asr scribe config set folder "<folder>"',
        'uv run --extra asr scribe vault-note "<meeting>"',
        'uv run --extra asr scribe slack "<meeting>"',
    ],
)
def test_destination_commands_written_exactly(destinations: str, command: str) -> None:
    assert command in destinations


def test_every_scribe_command_uses_the_asr_extra(skill: str) -> None:
    for line in skill.splitlines():
        for match in re.finditer(r"uv run (?!--extra asr)\S*\s*scribe", line):
            pytest.fail(f"scribe run without --extra asr: {line.strip()}")
    assert "uv run scribe" not in skill
    assert not re.search(r"uv sync", skill.replace("plain `uv sync`", ""))


def test_vault_question_needs_an_explicit_yes(destinations: str) -> None:
    assert 'Ask exactly: "Save the Summary and Minutes to your vault? (y/N)"' in destinations
    assert "Only an\nexplicit yes" in destinations
    assert "Enter, empty or anything else skips the vault note" in destinations


def test_no_answer_skips_vault_note_but_slack_still_runs(destinations: str) -> None:
    vault, _, slack = destinations.partition("**Slack message.**")
    assert "do not run `config` or `vault-note`" in vault
    assert "Always, whatever the user answered about the vault" in slack
    assert 'scribe slack "<meeting>"' in slack


def test_vault_note_runs_only_in_the_yes_branch(destinations: str) -> None:
    before_yes, _, after_yes = destinations.partition("On yes:")
    assert "scribe vault-note" not in before_yes.replace("do not run `config` or `vault-note`", "")
    assert 'scribe vault-note "<meeting>"' in after_yes


@pytest.mark.parametrize(
    "marker",
    [
        # config set: exit 2
        "must be relative to the vault, without '..'",
        "`unknown key`",
        "`cannot read ... as JSON`",
        # vault-note: exit 1 and every exit 2 message
        "`already exists and was left unchanged`",
        "`no vault configured` or `is not a directory`",
        "`no summary.md`, `no minutes.md` or `no transcript.json`",
        "`does not start with YYYY-MM-DD`",
        # slack: exit 2 and exit 1
        "Exit 2, `no stats.json`",
        "Exit 2, `no summary.md`",
        "Exit 1, `no '## ... ' heading`",
        "Exit 1, `stats.json` unreadable",
        "Exit 1, `cannot read summary.md`",
    ],
)
def test_every_refusal_has_a_next_step(destinations: str, marker: str) -> None:
    assert marker in destinations


def test_classifier_denial_next_step_uses_the_issue_wording(destinations: str) -> None:
    assert "only you can approve it in this session, or run the printed command yourself" in (
        destinations.replace("\n", " ")
    )
    assert "with !" in destinations
    assert "Do not retry" in destinations


def test_slack_md_is_posted_by_the_user(destinations: str) -> None:
    assert "post it in Slack yourself; scribe never posts" in destinations.replace("\n", " ")


def test_renaming_refreshes_slack_and_explains_the_stale_vault_note(skill: str) -> None:
    done = skill.split("11. **Done.**", 1)[1].split("\n## Cleanup\n", 1)[0]
    done = " ".join(done.split())
    assert 'runs `uv run --extra asr scribe slack "<meeting>"` again' in done
    assert "An existing vault note is left unchanged" in done
    assert "vault-note` refuses to overwrite it" in done
    assert (
        "Only when this run is a re-Naming (choice (3)) and `vault-note` exited 1 with "
        "`already exists and was left unchanged` in step 10, tell the user:"
    ) in done
    assert "Delete or rename it" in done
    assert 'scribe vault-note "<meeting>"' in done
    assert "On a first run, or when no vault note exists, say nothing about it." in done


def test_slack_error_sentence_reads_as_one_instruction(destinations: str) -> None:
    assert "show it, then, and" not in destinations
    assert "show it and\ngive the user the next step for it below, then go on to step 11:" in destinations
