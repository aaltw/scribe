"""Transcript tests use synthetic Segments with invented text, never a real Meeting."""

from pathlib import Path

import pytest

from scribe.transcript import (
    Segment,
    Transcript,
    TranscriptError,
    format_timestamp,
    parse_json,
    read_json,
    render_json,
    render_markdown,
    write_json,
    write_markdown,
)

GOLDEN = Path(__file__).parent / "golden"


def synthetic() -> Transcript:
    return Transcript(
        segments=(
            Segment(0.0, 4.2, "Speaker 1", "Welkom allemaal, shall we start?", "Sam"),
            Segment(4.25, 9.8004, "Speaker 2", "Ja, prima. Let's begin with the demo."),
            Segment(3725.0, 3731.5, "Speaker 1", "Tot slot: één actiepunt.", "Sam"),
        ),
        speakers=("Speaker 1", "Speaker 2"),
        models={"transcribe": "parakeet-tdt-0.6b-v3", "diarize": "pyannote-community-1"},
    )


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (59.99, "00:59"),
        (60, "01:00"),
        (3599.9, "59:59"),
        (3600, "1:00:00"),
        (3725, "1:02:05"),
        (36000, "10:00:00"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_markdown_matches_golden() -> None:
    expected = (GOLDEN / "synthetic_meeting.md").read_text(encoding="utf-8")
    assert render_markdown(synthetic()) == expected


def test_markdown_past_an_hour_uses_hour_format() -> None:
    assert "**Sam** [1:02:05]\n" in render_markdown(synthetic())


def test_json_matches_golden() -> None:
    expected = (GOLDEN / "synthetic_meeting.json").read_text(encoding="utf-8")
    assert render_json(synthetic()) == expected


def test_json_round_trip_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json(synthetic(), first)
    write_json(read_json(first), second)
    assert second.read_bytes() == first.read_bytes()


def test_read_json_restores_the_model(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    write_json(synthetic(), path)
    loaded = read_json(path)
    assert loaded.speakers == ("Speaker 1", "Speaker 2")
    assert loaded.segments[1].participant is None
    assert loaded.segments[1].end == 9.8
    assert dict(loaded.models) == dict(synthetic().models)


def test_name_speakers_rewrites_both_outputs(tmp_path: Path) -> None:
    named = synthetic().name_speakers({"Speaker 2": "Robin"})
    assert [s.label for s in named.segments] == ["Sam", "Robin", "Sam"]
    write_markdown(named, tmp_path / "out.md")
    assert "**Robin** [00:04]" in (tmp_path / "out.md").read_text(encoding="utf-8")
    assert parse_json(render_json(named)).segments[1].participant == "Robin"


def test_name_speakers_leaves_original_unchanged() -> None:
    original = synthetic()
    original.name_speakers({"Speaker 2": "Robin"})
    assert original.segments[1].participant is None


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "{}",
        '{"version": 2, "models": {}, "speakers": [], "segments": []}',
        '{"version": 1, "models": {}, "speakers": [], "segments": [{"start": "x"}]}',
    ],
)
def test_parse_json_rejects_bad_input(text: str) -> None:
    with pytest.raises(TranscriptError):
        parse_json(text)
