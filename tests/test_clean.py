"""check-clean tests use a hand-written synthetic Meeting with invented text and names."""

import re
from pathlib import Path

import pytest

from scribe import cli
from scribe.cli import main
from scribe.transcript import Segment, Transcript, write_json, write_markdown

NAME = "Anne-Marie O'Neil [ext]"  # a space, an apostrophe, a hyphen and a `[`

RAW = Transcript(
    segments=(
        Segment(0.0, 5.1, "Speaker 1", "goedemorgen zulle we beginne", NAME),
        Segment(5.5, 60.0, "Speaker 2", "ja de planing staat klaar"),
        Segment(61.2, 70.0, "Speaker 1", "eerst het budjet en dan de demo", NAME),
        Segment(3725.4, 3730.0, "Speaker 2", "tot volgende week"),
    ),
    speakers=("Speaker 1", "Speaker 2"),
    models={"asr": "test-1"},
)

H1 = f"**{NAME}** [00:00]"
H2 = "**Speaker 2** [00:05]"
H3 = f"**{NAME}** [01:01]"
H4 = "**Speaker 2** [1:02:05]"  # 3725.4 s, past an hour
T1 = "Goedemorgen, zullen we beginnen?"
T2 = "Ja, de planning staat klaar."
T3 = "Eerst het budget,\nen dan de demo."  # clean text over two lines
T4 = "Tot volgende week."

CLEAN = [(H1, T1), (H2, T2), (H3, T3), (H4, T4)]
SECRETS = [NAME, "Anne", "O'Neil", "Speaker", *re.findall(r"\w{4,}", T1 + T2 + T3 + T4)]


def render(blocks: list[tuple[str, str]], gap: str = "\n") -> str:
    return gap.join(f"{header}\n{text}\n" for header, text in blocks)


def make_meeting(root: Path, clean: str | None) -> Path:
    folder = root / "2026-10-02-demo"
    folder.mkdir(parents=True)
    write_json(RAW, folder / "transcript.json")
    write_markdown(RAW, folder / "transcript.md")
    if clean is not None:
        (folder / "transcript.clean.md").write_text(clean, encoding="utf-8", newline="")
    return folder


def run(folder: Path, *extra: str) -> int:
    return main(["check-clean", str(folder), *extra])


MATCHING = {
    "canonical": render(CLEAN),
    "no blank lines": render(CLEAN, gap=""),
    "many blank lines": "\n\n" + render(CLEAN, gap="\n\n\n") + "\n\n",
    "trailing whitespace": "".join(
        f"{h}  \t\n{t.replace(chr(10), ' ' + chr(10))}   \n\n" for h, t in CLEAN
    ),
    "blank line inside text": render(
        [(H1, T1), (H2, T2), (H3, "Eerst het budget,\n\nen dan de demo."), (H4, T4)]
    ),
    "crlf line endings": render(CLEAN).replace("\n", "\r\n"),
}


@pytest.mark.parametrize("clean", MATCHING.values(), ids=MATCHING.keys())
def test_matching_clean_file_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], clean: str
) -> None:
    assert run(make_meeting(tmp_path, clean)) == 0
    out, err = capsys.readouterr()
    assert out == "4 of 4\n"
    assert err == ""


BROKEN = {
    "dropped Segment": (
        [(H1, T1), (H3, T3), (H4, T4)],
        "Segment 2 of 4 raw / 3 clean: header and timestamp",
    ),
    "two merged Segments": (
        [(H1, T1), (H2, T2 + "\n" + T3), (H4, T4)],
        "Segment 3 of 4 raw / 3 clean: header and timestamp",
    ),
    "swapped Speaker": (
        [(H1, T1), (f"**{NAME}** [00:05]", T2), (H3, T3), (H4, T4)],
        "Segment 2 of 4 raw / 4 clean: header",
    ),
    "changed timestamp": (
        [(H1, T1), (H2, T2), (f"**{NAME}** [01:02]", T3), (H4, T4)],
        "Segment 3 of 4 raw / 4 clean: timestamp",
    ),
    "hour timestamp written as minutes": (
        [(H1, T1), (H2, T2), (H3, T3), ("**Speaker 2** [62:05]", T4)],
        "Segment 4 of 4 raw / 4 clean: timestamp",
    ),
    "reordered pair": (
        [(H1, T1), (H3, T3), (H2, T2), (H4, T4)],
        "Segment 2 of 4 raw / 4 clean: header and timestamp",
    ),
    "empty text in one Segment": (
        [(H1, T1), (H2, T2), (H3, "  "), (H4, T4)],
        "Segment 3 of 4 raw / 4 clean: empty text",
    ),
    "last Segment missing": (
        [(H1, T1), (H2, T2), (H3, T3)],
        "Segment 4 of 4 raw / 3 clean: missing",
    ),
    "extra Segment": (
        [*CLEAN, ("**Speaker 2** [1:02:10]", T4)],
        "Segment 5 of 4 raw / 5 clean: extra",
    ),
}


@pytest.mark.parametrize(("blocks", "line"), BROKEN.values(), ids=BROKEN.keys())
def test_broken_clean_file_fails_with_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], blocks: list[tuple[str, str]], line: str
) -> None:
    assert run(make_meeting(tmp_path, render(blocks))) == 1
    out, err = capsys.readouterr()
    assert out == line + "\n"
    assert err == ""


def test_text_before_first_header_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(make_meeting(tmp_path, "# Clean Transcript\n\n" + render(CLEAN))) == 1
    assert capsys.readouterr().out == "Segment 1 of 4 raw / 4 clean: text before first header\n"


@pytest.mark.parametrize(("blocks", "line"), BROKEN.values(), ids=BROKEN.keys())
@pytest.mark.parametrize("partial", [False, True])
def test_output_never_contains_segment_text_or_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    blocks: list[tuple[str, str]],
    line: str,
    partial: bool,
) -> None:
    run(make_meeting(tmp_path, render(blocks)), *(["--partial"] if partial else []))
    out, err = capsys.readouterr()
    for secret in SECRETS:
        assert secret not in out + err


def test_missing_clean_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(make_meeting(tmp_path, None)) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err == "scribe: error: no transcript.clean.md in the Meeting folder\n"


def test_missing_meeting_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-clean", "nope", "--meetings-dir", str(tmp_path)]) == 2
    assert capsys.readouterr().err == "scribe: error: no such Meeting folder\n"


def test_meeting_resolved_under_meetings_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_meeting(tmp_path / "meetings", render(CLEAN))
    args = ["check-clean", "2026-10-02-demo", "--meetings-dir", str(tmp_path / "meetings")]
    assert main(args) == 0
    assert capsys.readouterr().out == "4 of 4\n"


RATIO_WARNING = re.compile(
    r"scribe: warning: clean/raw text length ratio \d+\.\d\d is outside 0\.5-2\.0\n"
)


@pytest.mark.parametrize(
    ("factor", "warns"), [(0.3, True), (0.6, False), (1.0, False), (1.9, False), (2.5, True)]
)
def test_length_ratio_outside_range_warns_but_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], factor: float, warns: bool
) -> None:
    raw_length = sum(len(s.text) for s in RAW.segments)
    per_segment = round(raw_length * factor / 4)
    blocks = [(header, "x" * per_segment) for header, _ in CLEAN]
    assert run(make_meeting(tmp_path, render(blocks))) == 0
    out, err = capsys.readouterr()
    assert out == "4 of 4\n"
    assert bool(RATIO_WARNING.fullmatch(err)) is warns
    if not warns:
        assert err == ""


def test_partial_accepts_a_correct_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path, render(CLEAN[:2]))
    assert run(folder, "--partial") == 0
    assert capsys.readouterr() == ("2 of 4\n", "")
    assert run(folder) == 1
    assert capsys.readouterr().out == "Segment 3 of 4 raw / 2 clean: missing\n"


def test_partial_accepts_the_whole_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(make_meeting(tmp_path, render(CLEAN)), "--partial") == 0
    assert capsys.readouterr().out == "4 of 4\n"


def test_partial_ratio_covers_only_the_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # One clean Segment out of four would be far below 0.5 against the whole raw text.
    assert run(make_meeting(tmp_path, render(CLEAN[:1])), "--partial") == 0
    assert capsys.readouterr() == ("1 of 4\n", "")


@pytest.mark.parametrize(
    ("blocks", "line"),
    [
        ([(H1, T1), (H3, T3)], "Segment 2 of 4 raw / 2 clean: header and timestamp"),
        ([(H1, T1), (H2, "")], "Segment 2 of 4 raw / 2 clean: empty text"),
        ([*CLEAN, (H4, T4)], "Segment 5 of 4 raw / 5 clean: extra"),
    ],
)
def test_partial_still_fails_a_wrong_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], blocks: list[tuple[str, str]], line: str
) -> None:
    assert run(make_meeting(tmp_path, render(blocks)), "--partial") == 1
    assert capsys.readouterr().out == line + "\n"


def test_check_clean_is_a_subcommand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "check-clean" in cli.SUBCOMMANDS
    assert "check-clean" in cli.USAGE

    def no_pipeline(*args: object, **kwargs: object) -> None:
        raise AssertionError("check-clean fell through to the pipeline")

    monkeypatch.setattr(cli, "run", no_pipeline)
    assert run(make_meeting(tmp_path, render(CLEAN))) == 0
    assert capsys.readouterr().out == "4 of 4\n"
