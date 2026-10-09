"""vault-note and config tests: synthetic Meeting, invented text, tmp vault and config path."""

from pathlib import Path

import pytest

from scribe.cli import SUBCOMMANDS, USAGE, main
from scribe.transcript import Segment, Transcript, write_json

SUMMARY = """# Summary

Source: transcript.clean.md

## Overview

Alex and Sam planned the invented garden project.

## Decisions

- Build the shed first.

## Action items

- Order wood (owner: Alex)
"""
MINUTES = """# Minutes

Source: transcript.clean.md

## Topic: Shed

### Discussed

- Size of the shed.

### Decided

- Eight square metres.

### Agreed

- None.
"""


def make_meeting(root: Path, name: str = "2026-01-05-weekly-sync") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    write_json(
        Transcript(
            segments=(
                Segment(0.0, 5.0, "Speaker 1", "Hallo.", participant="Alex"),
                Segment(5.0, 9.0, "Speaker 2", "Dag."),
                Segment(9.0, 12.0, "Speaker 1", "Weer.", participant="Alex"),
            ),
            speakers=("Speaker 1", "Speaker 2"),
            models={"asr": "t"},
        ),
        folder / "transcript.json",
    )
    (folder / "summary.md").write_text(SUMMARY, encoding="utf-8")
    (folder / "minutes.md").write_text(MINUTES, encoding="utf-8")
    return folder


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path, Path]:
    vault = tmp_path / "vault"
    vault.mkdir()
    return tmp_path, vault, tmp_path / "cfg" / "config.json"


def run_note(meeting: Path, vault: Path, cfg: Path, *extra: str, with_vault: bool = True) -> int:
    args = ["vault-note", str(meeting), "--config", str(cfg)]
    if with_vault:
        args += ["--vault", str(vault)]
    return main([*args, *extra])


def test_note_named_from_date_and_slug_of_messy_name(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root, "2026-01-05-Weekly Sync  Q1")
    assert run_note(meeting, vault, cfg) == 0
    assert (vault / "Meetings" / "2026-01-05-weekly-sync-q1.md").is_file()


def test_default_name_with_time_keeps_time_and_single_date(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root, "2026-01-05-11-19-53")
    assert run_note(meeting, vault, cfg, "--folder", "m") == 0
    assert [p.name for p in (vault / "m").iterdir()] == ["2026-01-05-11-19-53.md"]


def test_date_only_folder_name(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root, "2026-01-05")
    assert run_note(meeting, vault, cfg, "--folder", "m") == 0
    assert [p.name for p in (vault / "m").iterdir()] == ["2026-01-05.md"]


def test_frontmatter_and_body_order(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert run_note(meeting, vault, cfg, "--folder", "m") == 0
    text = (vault / "m" / "2026-01-05-weekly-sync.md").read_text(encoding="utf-8")
    front, _, body = text.removeprefix("---\n").partition("\n---\n")
    assert front.splitlines() == [
        "type: meeting",
        "status: completed",
        "created: 2026-01-05",
        "updated: 2026-01-05",
        "tags: [meeting, scribe]",
        "participants: [Alex]",
        'meeting-type: ""',
        'linked-project: ""',
        'linked-weekly: ""',
    ]
    assert "Speaker 2" not in front
    assert body.index("# Summary") < body.index("# Minutes")
    assert body.lstrip().startswith("# Summary")
    assert "Order wood (owner: Alex)" in body
    assert "Eight square metres." in body


def test_no_named_participants_gives_empty_list(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    write_json(
        Transcript(segments=(Segment(0.0, 1.0, "Speaker 1", "Hoi."),), speakers=("Speaker 1",)),
        meeting / "transcript.json",
    )
    assert run_note(meeting, vault, cfg, "--folder", "m") == 0
    assert "participants: []\n" in (vault / "m" / "2026-01-05-weekly-sync.md").read_text("utf-8")


def test_existing_note_exits_1_and_is_unchanged(
    env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    target = vault / "m" / "2026-01-05-weekly-sync.md"
    target.parent.mkdir()
    target.write_text("mine\n", encoding="utf-8")
    assert run_note(meeting, vault, cfg, "--folder", "m") == 1
    assert target.read_text(encoding="utf-8") == "mine\n"
    assert "delete or rename" in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["summary.md", "minutes.md"])
def test_missing_input_exits_2_naming_file(
    env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    (meeting / missing).unlink()
    assert run_note(meeting, vault, cfg) == 2
    assert missing in capsys.readouterr().err
    assert not (vault / "20-work").exists()


def test_no_vault_exits_2_with_config_command(
    env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert run_note(meeting, vault, cfg, with_vault=False) == 2
    assert "scribe config set vault" in capsys.readouterr().err


def test_dry_run_prints_path_and_frontmatter_and_writes_nothing(
    env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert run_note(meeting, vault, cfg, "--dry-run") == 0
    out = capsys.readouterr().out
    assert str(vault / "Meetings" / "2026-01-05-weekly-sync.md") in out
    assert "participants: [Alex]" in out
    assert list(vault.iterdir()) == []


def test_vault_and_folder_come_from_config(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert main(["config", "--config", str(cfg), "set", "vault", str(vault)]) == 0
    assert main(["config", "--config", str(cfg), "set", "folder", "notes/m"]) == 0
    assert main(["vault-note", str(meeting), "--config", str(cfg)]) == 0
    assert (vault / "notes/m/2026-01-05-weekly-sync.md").is_file()


def test_folder_escaping_the_vault_is_refused(env: tuple[Path, Path, Path]) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert run_note(meeting, vault, cfg, "--folder", "../out") == 2
    assert run_note(meeting, vault, cfg, "--folder", str(root)) == 2


@pytest.mark.parametrize("folder", ["../out", "a/../b", "/abs/path"])
def test_config_set_folder_refuses_escape_like_vault_note(
    env: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str], folder: str
) -> None:
    root, vault, cfg = env
    meeting = make_meeting(root)
    assert main(["config", "--config", str(cfg), "set", "folder", folder]) == 2
    set_err = capsys.readouterr().err
    assert not cfg.exists()
    assert run_note(meeting, vault, cfg, "--folder", folder) == 2
    assert capsys.readouterr().err == set_err
    assert "must be relative to the vault, without '..'" in set_err


def test_config_show_and_set_use_the_given_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "c" / "config.json"
    assert main(["config", "--config", str(cfg), "show"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "vault: (not set)",
        "folder: Meetings (default)",
    ]
    assert main(["config", "--config", str(cfg), "set", "vault", "/v"]) == 0
    assert main(["config", "--config", str(cfg), "show"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "vault: /v"
    assert cfg.is_file()


def test_config_path_from_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "env.json"
    monkeypatch.setenv("SCRIBE_CONFIG", str(cfg))
    assert main(["config", "set", "folder", "x"]) == 0
    assert main(["config", "show"]) == 0
    assert "folder: x" in capsys.readouterr().out
    assert cfg.is_file()


def test_config_unknown_key_and_bad_file_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "config.json"
    assert main(["config", "--config", str(cfg), "set", "colour", "red"]) == 2
    assert "vault, folder" in capsys.readouterr().err
    cfg.write_text("{nope", encoding="utf-8")
    assert main(["config", "--config", str(cfg), "show"]) == 2


def test_subcommands_registered_and_gitignored() -> None:
    assert "vault-note" in SUBCOMMANDS
    assert "config" in SUBCOMMANDS
    assert "vault-note" in USAGE
    assert "config" in USAGE
    root = Path(__file__).resolve().parent.parent
    assert ".scribe/" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
