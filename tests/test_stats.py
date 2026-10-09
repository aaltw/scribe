"""Speaker stats tests use a synthetic Meeting folder with invented text."""

import json
from pathlib import Path

import pytest

from scribe.cli import SUBCOMMANDS, USAGE, main
from scribe.stats import talk_time
from scribe.transcript import Segment, Transcript, write_json


def make_meeting(root: Path, segments: tuple[Segment, ...], speakers: tuple[str, ...]) -> Path:
    folder = root / "2026-10-01-demo"
    folder.mkdir(parents=True)
    write_json(
        Transcript(segments=segments, speakers=speakers, models={"asr": "t"}),
        folder / "transcript.json",
    )
    return folder


def two_speakers(root: Path) -> Path:
    return make_meeting(
        root,
        (
            Segment(0.0, 10.0, "Speaker 1", "Een.", participant="Alex"),
            Segment(10.0, 13.5, "Speaker 2", "Twee."),
            Segment(13.5, 18.0005, "Speaker 1", "Drie.", participant="Alex"),
            Segment(20.0, 20.0, "Speaker 2", "Vier."),
        ),
        ("Speaker 1", "Speaker 2"),
    )


def test_writes_stats_json_with_every_value(tmp_path: Path) -> None:
    folder = two_speakers(tmp_path)
    assert main(["stats", str(folder)]) == 0
    document = json.loads((folder / "stats.json").read_text(encoding="utf-8"))
    assert document == {
        "version": 1,
        "speakers": [
            {
                "speaker": "Speaker 1",
                "participant": "Alex",
                "seconds": 14.5,
                "share": 0.806,
                "segments": 2,
            },
            {
                "speaker": "Speaker 2",
                "participant": None,
                "seconds": 3.5,
                "share": 0.194,
                "segments": 2,
            },
        ],
    }


def test_prints_a_line_per_speaker(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = two_speakers(tmp_path)
    assert main(["stats", str(folder)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Alex: 00:14, 80.6%, 2 Segments",
        "Speaker 2: 00:03, 19.4%, 2 Segments",
    ]


def test_hour_long_talk_time_uses_h_mm_ss(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path, (Segment(0.0, 3725.0, "Speaker 1", "Lang."),), ("Speaker 1",))
    assert main(["stats", str(folder)]) == 0
    assert capsys.readouterr().out == "Speaker 1: 1:02:05, 100.0%, 1 Segments\n"


def test_zero_total_talk_time_gives_share_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(
        tmp_path,
        (Segment(1.0, 1.0, "Speaker 1", "Hm."), Segment(2.0, 2.0, "Speaker 2", "Ja.")),
        ("Speaker 1", "Speaker 2"),
    )
    assert main(["stats", str(folder)]) == 0
    shares = [
        s["share"]
        for s in json.loads((folder / "stats.json").read_text(encoding="utf-8"))["speakers"]
    ]
    assert shares == [0, 0]
    assert capsys.readouterr().out.splitlines()[0] == "Speaker 1: 00:00, 0.0%, 1 Segments"


def test_missing_transcript_exits_1_naming_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "empty"
    folder.mkdir()
    assert main(["stats", str(folder)]) == 1
    assert "transcript.json" in capsys.readouterr().err
    assert not (folder / "stats.json").exists()


def test_meeting_name_resolves_under_meetings_dir(tmp_path: Path) -> None:
    folder = two_speakers(tmp_path)
    assert main(["stats", folder.name, "--meetings-dir", str(tmp_path)]) == 0
    assert (folder / "stats.json").is_file()


def test_stats_is_a_subcommand_and_in_usage() -> None:
    assert "stats" in SUBCOMMANDS
    assert "stats" in USAGE


def test_talk_time_is_the_unrounded_sum() -> None:
    assert talk_time([Segment(0.0, 1.2345, "a", ""), Segment(2.0, 3.0, "a", "")]) == pytest.approx(
        2.2345
    )
