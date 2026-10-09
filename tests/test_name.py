"""Naming tests use a synthetic Meeting folder with invented text."""

import os
from pathlib import Path

import pytest

from scribe.cli import main
from scribe.transcript import Segment, Transcript, write_json, write_markdown


def make_meeting(root: Path) -> Path:
    folder = root / "2026-10-01-demo"
    folder.mkdir(parents=True)
    transcript = Transcript(
        segments=(
            Segment(0.0, 4.2, "Speaker 1", "Welkom allemaal."),
            Segment(4.25, 9.8, "Speaker 2", "Ja, prima."),
            Segment(3599.9996, 3601.0, "Speaker 1", "Bijna een uur."),
        ),
        speakers=("Speaker 1", "Speaker 2"),
        models={"asr": "test-1"},
    )
    write_json(transcript, folder / "transcript.json")
    write_markdown(transcript, folder / "transcript.md")
    return folder


def snapshot(folder: Path) -> tuple[str, str]:
    return (
        (folder / "transcript.json").read_text(encoding="utf-8"),
        (folder / "transcript.md").read_text(encoding="utf-8"),
    )


def test_names_one_of_two_speakers(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 1=Alex"]) == 0
    json_text, md = snapshot(folder)
    assert '"participant": "Alex"' in json_text
    assert '"participant": null' in json_text
    assert "**Alex** [00:00]" in md
    assert "**Speaker 2** [00:04]" in md
    assert "**Speaker 1**" not in md


def test_renaming_replaces_earlier_name(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 1=Alex"]) == 0
    assert main(["name", str(folder), "Speaker 1=Sam"]) == 0
    json_text, md = snapshot(folder)
    assert '"participant": "Sam"' in json_text
    assert "Alex" not in json_text
    assert "**Sam** [00:00]" in md
    assert "Alex" not in md


def test_unknown_speaker_fails_and_leaves_files_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path)
    before = snapshot(folder)
    assert main(["name", str(folder), "Speaker 1=Alex", "Speaker 9=Nobody"]) == 1
    assert "unknown Speaker 'Speaker 9'" in capsys.readouterr().err
    assert snapshot(folder) == before


@pytest.mark.parametrize("pair", ["Speaker 1", "Speaker 1=", "Speaker 1=  ", "=Alex"])
def test_malformed_pair_fails_and_leaves_files_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pair: str
) -> None:
    folder = make_meeting(tmp_path)
    before = snapshot(folder)
    assert main(["name", str(folder), pair]) == 1
    assert "malformed pair" in capsys.readouterr().err
    assert snapshot(folder) == before


def test_meeting_resolved_under_meetings_dir(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path / "meetings")
    assert (
        main(
            [
                "name",
                "2026-10-01-demo",
                "Speaker 2=Sam",
                "--meetings-dir",
                str(tmp_path / "meetings"),
            ]
        )
        == 0
    )
    assert "**Sam** [00:04]" in snapshot(folder)[1]


def test_missing_meeting_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["name", "nope", "Speaker 1=Alex", "--meetings-dir", str(tmp_path)]) == 1
    assert "no Meeting folder" in capsys.readouterr().err


def test_markdown_uses_millisecond_rounded_times(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 1=Alex"]) == 0
    # 3599.9996 s rounds to 3600.0 in JSON, so Markdown must agree with it.
    assert "**Alex** [1:00:00]" in snapshot(folder)[1]


def test_failed_write_leaves_files_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path)
    before = snapshot(folder)
    real_replace = os.replace

    def failing(src: Path, dst: Path) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", failing)
    assert main(["name", str(folder), "Speaker 1=Alex"]) == 1
    monkeypatch.setattr(os, "replace", real_replace)
    assert "cannot write" in capsys.readouterr().err
    assert snapshot(folder) == before
    assert sorted(p.name for p in folder.iterdir()) == ["transcript.json", "transcript.md"]
