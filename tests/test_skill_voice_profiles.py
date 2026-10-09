"""The /transcribe skill's Voice-profile steps: suggest in step 5, profile save in step 6.

Every SKILL.md match collapses whitespace first (AGENTS.md). The refusal markers are also
checked against the real CLI's stderr on synthetic Meetings under tmp_path, so the skill and
the code cannot drift apart. Names are invented; vectors are constructed.
"""

import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from scribe.cli import main
from scribe.embeddings import FILE_NAME as EMBEDDINGS_NAME
from scribe.transcript import Segment, Transcript, write_json

SKILL = Path(__file__).resolve().parent.parent / "skills/transcribe/SKILL.md"
MODEL = "test-model@1 (pyannote.audio 0)"
OTHER_MODEL = "other-model@2 (pyannote.audio 0)"


def flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture(scope="module")
def skill() -> str:
    return SKILL.read_text(encoding="utf-8")


def step(skill: str, number: int) -> str:
    """The text of one numbered step under ## Steps, whitespace collapsed."""
    steps = skill.split("\n## Steps\n", 1)[1].split("\n## Cleanup\n", 1)[0]
    start = re.search(rf"^{number}\. \*\*", steps, re.MULTILINE)
    assert start is not None, f"step {number} missing"
    end = re.search(rf"^{number + 1}\. \*\*", steps, re.MULTILINE)
    return flat(steps[start.start() : end.start() if end else len(steps)])


@pytest.fixture(scope="module")
def step5(skill: str) -> str:
    return step(skill, 5)


@pytest.fixture(scope="module")
def step6(skill: str) -> str:
    return step(skill, 6)


def test_steps_keep_their_numbers(skill: str) -> None:
    titles = re.findall(
        r"^(\d+)\. \*\*([^*]+)\*\*", skill.split("\n## Cleanup\n", 1)[0], re.MULTILINE
    )
    assert titles == [
        ("1", "Recording."),
        ("2", "Speaker hint."),
        ("3", "Run."),
        ("4", "Show lines."),
        ("5", "Naming."),
        ("6", "Record the Naming."),
        ("7", "Speaker stats."),
        ("8", "Cleanup."),
        ("9", "Summary and Minutes."),
        ("10", "Destinations."),
        ("11", "Done."),
    ]


def test_suggest_runs_before_the_naming_question(step5: str) -> None:
    command = 'uv run --extra asr scribe suggest "<meeting>"'
    assert command in step5
    assert step5.index(command) < step5.index('Then ask: "Who is each Speaker?')


def test_suggestion_is_decided_on_participant_never_on_cosine(step5: str) -> None:
    assert (
        "A Speaker has a suggestion only when its entry's `participant` is not null; decide on "
        "`participant` alone, never on `cosine`"
    ) in step5
    assert "A Speaker label from step 4 that is missing from the file has no suggestion." in step5
    assert "Only when it exits 0, Read `<meeting>/suggestions.json` with the Read tool" in step5
    assert "When suggest exits non-zero, no Speaker has a suggestion" in step5


def test_question_shows_the_suggestion_and_still_asks(step5: str) -> None:
    assert '"Speaker N looks like X (0.93)"' in step5
    assert "with X the `participant` value exactly as stored and its `cosine` to 2 decimals" in (
        step5
    )
    assert "Only when some Speakers have a suggestion" in step5


def test_enter_or_empty_never_accepts_a_suggestion(step5: str) -> None:
    assert "A suggestion is accepted only by an explicit yes for that Speaker" in step5
    assert (
        "Enter, an empty answer, or an answer that leaves out a Speaker with a suggestion never "
        "accepts it"
    ) in step5
    assert "A suggestion never reaches `scribe name` unless the user accepted it." in step5


def test_accepted_uses_stored_spelling_typed_uses_the_users(step5: str) -> None:
    assert "the `participant` value, in its stored spelling, is the name for step 6" in step5
    assert "When the user types a name for a Speaker, that name in their spelling is the name" in step5


def test_save_question_only_after_name_exits_zero(step6: str) -> None:
    voice = step6.split("**Voice profiles.**", 1)[1]
    assert voice.startswith(
        " Only after the last `scribe name` of this step exits 0, and only for the Participants "
        "named in this run"
    )
    assert 'ask exactly "Save a Voice profile for <name>? (y/N)" per Participant' in voice
    assert 'Only an explicit yes ("y", "yes") saves; no, "n", Enter, empty or anything else' in (
        voice
    )
    assert "When every Speaker was skipped, or `scribe name` never exited 0, ask nothing" in voice
    assert 'uv run --extra asr scribe profile save "<meeting>" "<name>"' in voice
    assert step6.index("scribe name") < step6.index("**Voice profiles.**")


def test_step_11_names_profile_files_only_when_saved(skill: str) -> None:
    done = step(skill, 11)
    assert "Only when step 6 saved a Voice profile in this run, name each profile file" in done
    assert "Deleting .scribe/profiles/<slug>.json (the file named above) undoes that save." in done
    assert "Only when that save printed more than 1 sample, add:" in done
    assert "When no profile was saved in this run, say nothing about profiles." in done


def test_denied_write_covers_suggest_and_profile_save(skill: str) -> None:
    denied = flat(skill.split("**A denied write.**", 1)[1])
    assert "`scribe suggest` command of step 5 and a `scribe profile save` of step 6" in denied
    assert '`! uv run --extra asr scribe profile save "<meeting>" "<name>"`' in denied
    assert "Only when the denied command is `scribe suggest`, go on with the Naming" in denied
    assert "Only when it is a `profile save`, nothing was saved for that Participant" in denied
    assert "Do not retry" in denied


def test_privacy_forbids_reading_vectors(skill: str) -> None:
    privacy = flat(skill.split("## Privacy", 1)[1].split("## Steps", 1)[0])
    assert "Never Read or print embeddings.json or a profile file" in privacy


# Synthetic Meetings for the drift checks.


def at(c: float) -> list[float]:
    return [c, math.sqrt(1 - c * c)]


def make_meeting(
    root: Path,
    names: dict[str, str | None],
    *,
    windows: int = 9,
    model: str = MODEL,
    embeddings: bool = True,
    entries: dict[str, dict[str, object]] | None = None,
) -> Path:
    folder = root / "2026-10-01-demo"
    folder.mkdir(parents=True)
    segments = tuple(
        Segment(float(k), float(k) + 1, label, f"invented line {k}", name)
        for k, (label, name) in enumerate(names.items())
    )
    write_json(Transcript(segments, tuple(names), {"diarize": model}), folder / "transcript.json")
    if embeddings:
        default = {
            label: {
                "model": model,
                "centroid": at(1.0),
                "mean_of_windows": at(1.0),
                "windows": windows,
            }
            for label in names
        }
        speakers = default if entries is None else entries
        document = {"version": 1, "window_rule": {}, "speakers": speakers}
        (folder / EMBEDDINGS_NAME).write_text(json.dumps(document), encoding="utf-8")
    return folder


def write_profile(
    profiles: Path, name: str, *, model: str = MODEL, vector: list[float] | None = None
) -> None:
    profiles.mkdir(exist_ok=True)
    sample = {"source": "src", "vector": vector or at(1.0), "windows": 9, "model": model}
    document = {"version": 1, "name": name, "samples": [sample]}
    (profiles / f"{name.casefold().replace(' ', '-')}.json").write_text(
        json.dumps(document), encoding="utf-8"
    )


Setup = Callable[[Path], tuple[Path, Path]]


def no_embeddings(root: Path) -> tuple[Path, Path]:
    return make_meeting(root, {"Speaker 1": None}, embeddings=False), root / "p"


def unreadable_embeddings(root: Path) -> tuple[Path, Path]:
    folder = make_meeting(root, {"Speaker 1": None}, embeddings=False)
    (folder / EMBEDDINGS_NAME).write_text("{", encoding="utf-8")
    return folder, root / "p"


def other_model(root: Path) -> tuple[Path, Path]:
    write_profile(root / "p", "Zora Quill", model=OTHER_MODEL)
    return make_meeting(root, {"Speaker 1": None}), root / "p"


def broken_profile(root: Path) -> tuple[Path, Path]:
    (root / "p").mkdir()
    (root / "p" / "zora.json").write_text("{", encoding="utf-8")
    return make_meeting(root, {"Speaker 1": None}), root / "p"


def too_many(root: Path) -> tuple[Path, Path]:
    for k in range(17):
        write_profile(root / "p", f"Zora Quill {k}")
    return make_meeting(root, {f"Speaker {k}": None for k in range(1, 18)}), root / "p"


def missing_folder(root: Path) -> tuple[Path, Path]:
    return root / "2026-10-01-gone", root / "p"


@pytest.mark.parametrize(
    ("marker", "needle", "setup"),
    [
        (
            "`no embeddings.json in this Meeting folder`",
            "no embeddings.json in this Meeting folder",
            no_embeddings,
        ),
        ("`cannot read embeddings.json`", "cannot read embeddings.json", unreadable_embeddings),
        (
            "`none of the ... saved Voice profile(s) comes from this Meeting's embedding model`",
            "saved Voice profile(s) comes from this Meeting's embedding model",
            other_model,
        ),
        ("`cannot read ... as a Voice profile`", "as a Voice profile", broken_profile),
        (
            "`more than the 16 scribe suggest can assign exactly`",
            "more than the 16 scribe suggest can assign exactly",
            too_many,
        ),
        ("`no Meeting folder`", "no Meeting folder", missing_folder),
    ],
)
def test_every_suggest_refusal_has_a_next_step_and_matches_the_cli(
    step5: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    marker: str,
    needle: str,
    setup: Setup,
) -> None:
    assert f"Exit 2, {marker}" in step5 or f", or {marker}" in step5
    folder, profiles = setup(tmp_path)
    assert main(["suggest", str(folder), "--profiles-dir", str(profiles)]) == 2
    assert needle in capsys.readouterr().err


def test_suggest_outcomes_without_suggestion_each_have_a_sentence(step5: str) -> None:
    assert "Exit 0 and no Speaker has a suggestion:" in step5
    assert "Exit 2, `cannot write`:" in step5
    assert "Any other exit: show its last stderr line" in step5
    assert "then ask the Naming question as usual" in step5


def test_skipped_profiles_note_matches_the_cli(
    step5: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "Only when suggest exits 0 and stderr has `scribe: note: N profile(s) skipped`" in step5
    write_profile(tmp_path / "p", "Zora Quill")
    write_profile(tmp_path / "p", "Ivo Brant", model=OTHER_MODEL)
    folder = make_meeting(tmp_path, {"Speaker 1": None})
    assert main(["suggest", str(folder), "--profiles-dir", str(tmp_path / "p")]) == 0
    assert "scribe: note: 1 profile(s) skipped" in capsys.readouterr().err


def save_setup(names: dict[str, str | None], **kwargs: object) -> Callable[[Path], Path]:
    def build(root: Path) -> Path:
        return make_meeting(root, names, **kwargs)  # type: ignore[arg-type]

    return build


def with_profile(build: Callable[[Path], Path], **kwargs: object) -> Callable[[Path], Path]:
    def both(root: Path) -> Path:
        write_profile(root / "p", "Zora Quill", **kwargs)  # type: ignore[arg-type]
        return build(root)

    return both


def broken_save_profile(root: Path) -> Path:
    (root / "p").mkdir()
    (root / "p" / "zora-quill.json").write_text("{", encoding="utf-8")
    return make_meeting(root, {"Speaker 1": "Zora Quill"})


def no_transcript(root: Path) -> Path:
    folder = root / "2026-10-01-demo"
    folder.mkdir()
    return folder


NAMED: dict[str, str | None] = {"Speaker 1": "Zora Quill"}


@pytest.mark.parametrize(
    ("marker", "needle", "build", "who"),
    [
        (
            "`is not named in this Meeting`",
            "is not named in this Meeting",
            save_setup({"Speaker 1": None}),
            "Zora Quill",
        ),
        (
            "`is named on N Speakers`",
            "is named on 2 Speakers",
            save_setup({"Speaker 1": "Zora Quill", "Speaker 2": "Zora Quill"}),
            "Zora Quill",
        ),
        (
            "`no embeddings.json in this Meeting folder`",
            "no embeddings.json in this Meeting folder",
            save_setup(NAMED, embeddings=False),
            "Zora Quill",
        ),
        (
            "`is unreadable or has no entry for`",
            "is unreadable or has no entry for",
            save_setup(NAMED, entries={}),
            "Zora Quill",
        ),
        (
            "`holds no usable mean of windows or model`",
            "holds no usable mean of windows or model",
            save_setup(
                NAMED,
                entries={
                    "Speaker 1": {
                        "model": MODEL,
                        "centroid": None,
                        "mean_of_windows": None,
                        "windows": 9,
                    }
                },
            ),
            "Zora Quill",
        ),
        (
            "`fewer than the 4 a profile needs`",
            "fewer than the 4 a profile needs",
            save_setup(NAMED, windows=3),
            "Zora Quill",
        ),
        (
            "`different model than the saved samples`",
            "different model than the saved samples",
            with_profile(save_setup(NAMED), model=OTHER_MODEL),
            "Zora Quill",
        ),
        (
            "`dimensions but the saved samples`",
            "dimensions but the saved samples",
            with_profile(save_setup(NAMED), vector=[1.0, 0.0, 0.0]),
            "Zora Quill",
        ),
        (
            "`cannot read ... as a Voice profile`",
            "as a Voice profile",
            broken_save_profile,
            "Zora Quill",
        ),
        (
            "`cannot make a profile file name`",
            "cannot make a profile file name",
            save_setup(NAMED),
            "!!!",
        ),
        (
            "`no Meeting folder`",
            "no Meeting folder",
            lambda root: root / "2026-10-01-gone",
            "Zora Quill",
        ),
        (
            "`cannot read transcript.json as a Transcript`",
            "cannot read transcript.json as a Transcript",
            no_transcript,
            "Zora Quill",
        ),
    ],
)
def test_every_profile_save_refusal_has_a_next_step_and_matches_the_cli(
    step6: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    marker: str,
    needle: str,
    build: Callable[[Path], Path],
    who: str,
) -> None:
    voice = step6.split("**Voice profiles.**", 1)[1]
    assert f"- {marker}" in voice or f", {marker}" in voice or f"or {marker}" in voice
    folder = build(tmp_path)
    assert main(["profile", "save", str(folder), who, "--profiles-dir", str(tmp_path / "p")]) == 2
    assert needle in capsys.readouterr().err


def test_profile_save_other_refusals_have_a_next_step(step6: str) -> None:
    voice = step6.split("**Voice profiles.**", 1)[1]
    assert "- `cannot write`:" in voice
    assert '- Any other exit: show its last stderr line and say "No profile was saved' in voice
    assert "go on to the next Participant's question, or to step 7 after the last one" in voice
