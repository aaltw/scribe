"""`scribe slack` tests use a synthetic Meeting folder with invented text and names."""

import subprocess
import sys
from pathlib import Path

import pytest

from scribe.cli import SUBCOMMANDS, USAGE, main
from scribe.stats import SpeakerStats, render_json

SUMMARY = """# Summary

Source: transcript.clean.md

## Overview

Alex and Sam planned the Q3 launch & agreed a smaller scope <draft>.

## Decisions

- Ship the beta on 1 > 0 terms.
- Drop the R&D report.

## Action items

- Alex writes the plan (owner: Alex)
- Someone books a room (owner: unclear)
"""

SUMMARY_NONE = (
    SUMMARY.split("## Decisions")[0] + "## Decisions\n\n- None.\n\n## Action items\n\n- None\n"
)

NO_NETWORK = """
import socket
import sys

def guarded(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        print(f"attempt {address!r}", file=sys.stderr)
        raise OSError("network blocked by test")
    return connect(self, address)

connect = socket.socket.connect
socket.socket.connect = guarded
socket.socket.connect_ex = lambda self, address: guarded(self, address) or 0

from scribe.cli import main

sys.exit(main(sys.argv[1:]))
"""


def make_meeting(root: Path, summary: str = SUMMARY, *, stats: bool = True) -> Path:
    folder = root / "2026-10-01-weekly-sync"
    folder.mkdir(parents=True)
    if summary:
        (folder / "summary.md").write_text(summary, encoding="utf-8")
    if stats:
        rows = [
            SpeakerStats("Speaker 1", "Alex", 3725.0, 0.806, 20),
            SpeakerStats("Speaker 2", None, 3.5, 0.194, 2),
            SpeakerStats("Speaker 3", "R&D <Sam>", 59.9, 0.0, 1),
        ]
        (folder / "stats.json").write_text(render_json(rows), encoding="utf-8")
    return folder


def test_slack_md_line_by_line(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    assert main(["slack", str(folder)]) == 0
    assert (folder / "slack.md").read_text(encoding="utf-8").split("\n") == [
        "*2026-10-01 weekly-sync*",
        "",
        "*Overview*",
        "Alex and Sam planned the Q3 launch &amp; agreed a smaller scope &lt;draft&gt;.",
        "",
        "*Decisions*",
        "• Ship the beta on 1 &gt; 0 terms.",
        "• Drop the R&amp;D report.",
        "",
        "*Action items*",
        "• Alex writes the plan (owner: Alex)",
        "• Someone books a room (owner: unclear)",
        "",
        "*Speaker stats*",
        "• Alex: 1:02:05, 80.6%",
        "• Speaker 2: 00:03, 19.4%",
        "• R&amp;D &lt;Sam&gt;: 00:59, 0.0%",
        "",
    ]


def test_slack_md_has_no_markdown_headings(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    main(["slack", str(folder)])
    text = (folder / "slack.md").read_text(encoding="utf-8")
    assert not any(line.startswith(("#", "- ")) for line in text.split("\n"))
    headings = [line for line in text.split("\n") if line.startswith("*")]
    assert headings == [
        "*2026-10-01 weekly-sync*",
        "*Overview*",
        "*Decisions*",
        "*Action items*",
        "*Speaker stats*",
    ]


def test_none_sections_render_as_none_bullet(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path, SUMMARY_NONE)
    assert main(["slack", str(folder)]) == 0
    lines = (folder / "slack.md").read_text(encoding="utf-8").split("\n")
    assert lines[lines.index("*Decisions*") + 1] == "• None"
    assert lines[lines.index("*Action items*") + 1] == "• None"


def test_missing_summary_exits_2_naming_file_and_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path, "")
    assert main(["slack", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "summary.md" in err and "/transcribe" in err
    assert not (folder / "slack.md").exists()


def test_missing_stats_exits_2_naming_file_and_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path, stats=False)
    assert main(["slack", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "stats.json" in err and "scribe stats" in err
    assert not (folder / "slack.md").exists()


def test_summary_without_section_is_refused_with_next_step(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path, "# Summary\n\n## Overview\n\nText.\n")
    assert main(["slack", str(folder)]) == 1
    assert "/transcribe" in capsys.readouterr().err


def test_slack_is_a_subcommand_and_in_usage(tmp_path: Path) -> None:
    assert "slack" in SUBCOMMANDS
    assert "slack" in USAGE


def test_slack_makes_no_network_connect(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", NO_NETWORK, "slack", str(folder)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "attempt" not in result.stderr
    assert (folder / "slack.md").is_file()
