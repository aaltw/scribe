"""A non-UTF-8 transcript.json gives each reader its error line, not a traceback (synthetic bytes)."""

import subprocess
import sys
from pathlib import Path

import pytest

from scribe.cli import main
from scribe.transcript import Segment, Transcript, write_json

BAD = b"\xc3\x28"
HELPER = Path(__file__).resolve().parent.parent / "skills/transcribe/sample_lines.py"


def meeting(tmp_path: Path, *, clean: bool = False) -> Path:
    folder = tmp_path / "2026-10-02-demo"
    folder.mkdir()
    (folder / "transcript.json").write_bytes(BAD)
    if clean:
        (folder / "transcript.clean.md").write_bytes(BAD)
    return folder


def test_sample_lines_prints_its_error_line(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(HELPER), str(meeting(tmp_path))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("sample_lines: error: cannot read transcript.json as UTF-8")
    assert "Traceback" not in result.stderr


def test_name_prints_its_error_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = meeting(tmp_path)
    assert main(["name", str(folder), "Speaker 1=Ann"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("scribe: error: cannot read ")
    assert "as UTF-8 text; " in err


def test_check_clean_prints_its_error_line_for_transcript_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check-clean", str(meeting(tmp_path))]) == 2
    assert capsys.readouterr().err == "scribe: error: cannot read transcript.json as a Transcript\n"


def test_check_clean_prints_its_error_line_for_transcript_clean_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = meeting(tmp_path, clean=True)
    segment = Segment(0.0, 1.0, "Speaker 1", "synthetic words")
    write_json(
        Transcript(segments=(segment,), speakers=("Speaker 1",), models={}),
        folder / "transcript.json",
    )
    assert main(["check-clean", str(folder)]) == 2
    assert (
        capsys.readouterr().err == "scribe: error: cannot read transcript.clean.md as UTF-8 text\n"
    )
