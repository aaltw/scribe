"""`scribe name` with a Clean Transcript present; synthetic Meeting with invented text and names."""

import os
from pathlib import Path

import pytest

from scribe.cli import main
from scribe.transcript import (
    Segment,
    Transcript,
    read_json,
    render_header,
    write_json,
    write_markdown,
)

NAME = "Anne-Marie O'Neil [ext]"

RAW = Transcript(
    segments=(
        Segment(0.0, 5.1, "Speaker 1", "goedemorgen zulle we beginne"),
        Segment(5.5, 60.0, "Speaker 2", "ja de planing staat klaar"),
        Segment(3725.4, 3730.0, "Speaker 1", "tot volgende week"),
    ),
    speakers=("Speaker 1", "Speaker 2"),
    models={"asr": "test-1"},
)

# Clean texts with uneven formatting: a two-line text, trailing spaces, extra blank lines.
TEXTS = (
    "Goedemorgen, zullen we beginnen?  ",
    "Ja, de planning\nstaat klaar.",
    "Tot volgende week.",
)
GAPS = ("\n", "\n\n\n", "")


def write_clean(folder: Path) -> None:
    """Write transcript.clean.md from the folder's current transcript.json, as cleanup would."""
    transcript = read_json(folder / "transcript.json")
    text = "".join(
        f"{render_header(segment)}\n{body}\n{gap}"
        for segment, body, gap in zip(transcript.segments, TEXTS, GAPS, strict=True)
    )
    (folder / "transcript.clean.md").write_text(text, encoding="utf-8", newline="")


def make_meeting(root: Path) -> Path:
    folder = root / "2026-10-02-demo"
    folder.mkdir(parents=True)
    write_json(RAW, folder / "transcript.json")
    write_markdown(RAW, folder / "transcript.md")
    return folder


def clean_bytes(folder: Path) -> bytes:
    return (folder / "transcript.clean.md").read_bytes()


def split_lines(data: bytes) -> tuple[list[bytes], list[bytes]]:
    """(header lines, every other line) of a clean file."""
    lines = data.split(b"\n")
    headers = [line for line in lines if line.startswith(b"**")]
    return headers, [line for line in lines if not line.startswith(b"**")]


def test_naming_before_and_after_cleanup_give_the_same_clean_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = make_meeting(tmp_path / "before")
    assert main(["name", str(before), f"Speaker 1={NAME}"]) == 0
    write_clean(before)

    after = make_meeting(tmp_path / "after")
    write_clean(after)
    unnamed = clean_bytes(after)
    assert main(["name", str(after), f"Speaker 1={NAME}"]) == 0
    assert capsys.readouterr().err == ""

    headers_before, text_before = split_lines(clean_bytes(before))
    headers_after, text_after = split_lines(clean_bytes(after))
    assert headers_after == headers_before
    assert headers_after == [
        f"**{NAME}** [00:00]".encode(),
        b"**Speaker 2** [00:05]",
        f"**{NAME}** [1:02:05]".encode(),
    ]
    assert text_after == text_before == split_lines(unnamed)[1]
    assert clean_bytes(after) == clean_bytes(before)
    assert (before / "transcript.json").read_bytes() == (after / "transcript.json").read_bytes()

    for folder in (before, after):
        assert main(["check-clean", str(folder)]) == 0
        assert capsys.readouterr().out == "3 of 3\n"


def test_renaming_rewrites_already_named_clean_headers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 1=Sam", "Speaker 2=Robin"]) == 0
    write_clean(folder)
    text_lines = split_lines(clean_bytes(folder))[1]
    assert main(["name", str(folder), f"Speaker 1={NAME}"]) == 0
    headers, others = split_lines(clean_bytes(folder))
    assert headers == [
        f"**{NAME}** [00:00]".encode(),
        b"**Robin** [00:05]",
        f"**{NAME}** [1:02:05]".encode(),
    ]
    assert others == text_lines
    assert main(["check-clean", str(folder)]) == 0
    assert capsys.readouterr().err == ""


def test_crlf_clean_file_keeps_its_line_endings(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    write_clean(folder)
    path = folder / "transcript.clean.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert main(["name", str(folder), "Speaker 2=Robin"]) == 0
    data = path.read_bytes()
    assert b"**Robin** [00:05]\r\n" in data
    assert data.count(b"\r\n") == data.count(b"\n")
    assert main(["check-clean", str(folder)]) == 0


def test_stale_clean_file_is_kept_and_naming_still_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path)
    write_clean(folder)
    path = folder / "transcript.clean.md"
    stale = path.read_bytes().split(b"\n**Speaker 1** [1:02:05]")[0] + b"\n"  # last Segment gone
    path.write_bytes(stale)

    assert main(["name", str(folder), f"Speaker 1={NAME}"]) == 0
    err = capsys.readouterr().err
    assert "scribe: warning: transcript.clean.md does not match transcript.json" in err
    assert "scribe check-clean" in err
    assert path.read_bytes() == stale
    assert f'"participant": "{NAME}"' in (folder / "transcript.json").read_text(encoding="utf-8")
    assert f"**{NAME}** [00:00]" in (folder / "transcript.md").read_text(encoding="utf-8")


def test_no_clean_file_means_no_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = make_meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 2=Robin"]) == 0
    assert capsys.readouterr().err == ""
    assert not (folder / "transcript.clean.md").exists()


def test_failed_write_leaves_all_three_files_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path)
    write_clean(folder)
    names = ["transcript.clean.md", "transcript.json", "transcript.md"]
    before = [(folder / name).read_bytes() for name in names]

    def failing(src: Path, dst: Path) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", failing)
    assert main(["name", str(folder), "Speaker 1=Sam"]) == 1
    monkeypatch.undo()
    assert "cannot write" in capsys.readouterr().err
    assert [(folder / name).read_bytes() for name in names] == before
    assert sorted(p.name for p in folder.iterdir()) == names
