"""`scribe suggest` tests: synthetic Meetings and profiles under tmp_path, invented names."""

import json
import math
from pathlib import Path

import pytest

from scribe.cli import SUBCOMMANDS, USAGE, main
from scribe.embeddings import FILE_NAME as EMBEDDINGS_NAME
from scribe.suggest import SUGGEST_THRESHOLD, cosine

MODEL = "test-model@1 (pyannote.audio 0)"


def at(c: float) -> list[float]:
    """A unit vector whose dot product with [1, 0] is exactly `c`."""
    return [c, math.sqrt(1 - c * c)]


def meeting(
    root: Path,
    centroids: dict[str, list[float] | None],
    *,
    means: dict[str, list[float] | None] | None = None,
    model: str = MODEL,
    name: str = "2026-10-01-demo",
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    means = means or {}
    speakers = {
        label: {
            "model": model,
            "centroid": centroid,
            "mean_of_windows": means.get(label),
            "windows": 9,
        }
        for label, centroid in centroids.items()
    }
    document = {"version": 1, "window_rule": {}, "speakers": speakers}
    (folder / EMBEDDINGS_NAME).write_text(json.dumps(document), encoding="utf-8")
    return folder


def profile(profiles: Path, name: str, vectors: list[list[float]], *, model: str = MODEL) -> None:
    profiles.mkdir(exist_ok=True)
    samples = [
        {"source": f"src-{k}", "vector": v, "windows": 9, "model": model}
        for k, v in enumerate(vectors)
    ]
    document = {"version": 1, "name": name, "samples": samples}
    stem = name.casefold().replace(" ", "-")
    (profiles / f"{stem}.json").write_text(json.dumps(document), encoding="utf-8")


def run(folder: Path, profiles: Path) -> int:
    return main(["suggest", str(folder), "--profiles-dir", str(profiles)])


def suggestions(folder: Path) -> dict[str, dict[str, object]]:
    document = json.loads((folder / "suggestions.json").read_text(encoding="utf-8"))
    return {s["label"]: s for s in document["suggestions"]}


def test_suggest_is_a_subcommand() -> None:
    assert "suggest" in SUBCOMMANDS
    assert "suggest" in USAGE


def test_threshold_constant() -> None:
    assert SUGGEST_THRESHOLD == 0.8401


@pytest.mark.parametrize(("value", "suggested"), [(0.8400, False), (0.8401, True)])
def test_threshold_edge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: float, suggested: bool
) -> None:
    folder = meeting(tmp_path / "m", {"Speaker 1": at(value)})
    profile(tmp_path / "p", "Zora Quill", [[1.0, 0.0]])
    assert cosine(at(value), [1.0, 0.0]) == value
    assert run(folder, tmp_path / "p") == 0
    entry = suggestions(folder)["Speaker 1"]
    assert entry["participant"] == ("Zora Quill" if suggested else None)
    assert entry["cosine"] == value
    line = capsys.readouterr().out.strip()
    assert line == (
        f"Speaker 1: Zora Quill (cosine {value:.3f})"
        if suggested
        else f"Speaker 1: no suggestion (best cosine {value:.3f})"
    )


def test_swapped_labels_get_the_same_names(tmp_path: Path) -> None:
    zora, bram = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [zora])
    profile(profiles, "Bram Tulip", [bram])
    straight = meeting(tmp_path / "a", {"Speaker 1": zora, "Speaker 2": bram})
    swapped = meeting(tmp_path / "b", {"Speaker 2": bram, "Speaker 1": zora})
    assert run(straight, profiles) == run(swapped, profiles) == 0
    for folder in (straight, swapped):
        found = suggestions(folder)
        assert found["Speaker 1"]["participant"] == "Zora Quill"
        assert found["Speaker 2"]["participant"] == "Bram Tulip"
    flipped = meeting(tmp_path / "c", {"Speaker 1": bram, "Speaker 2": zora})
    assert run(flipped, profiles) == 0
    assert suggestions(flipped)["Speaker 1"]["participant"] == "Bram Tulip"
    assert suggestions(flipped)["Speaker 2"]["participant"] == "Zora Quill"


def test_speaker_below_threshold_gets_no_suggestion(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    profile(profiles, "Bram Tulip", [[0.0, 1.0]])
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0], "Speaker 2": deg(45)})
    assert run(folder, profiles) == 0
    found = suggestions(folder)
    assert found["Speaker 1"]["participant"] == "Zora Quill"
    assert found["Speaker 2"]["participant"] is None
    assert found["Speaker 2"]["cosine"] == pytest.approx(math.cos(math.radians(45)))


def test_three_speakers_against_one_profile(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    folder = meeting(
        tmp_path / "m", {"Speaker 1": at(0.90), "Speaker 2": at(0.95), "Speaker 3": at(0.3)}
    )
    assert run(folder, profiles) == 0
    found = suggestions(folder)
    assert [e["participant"] for e in found.values()] == [None, "Zora Quill", None]


def deg(angle: float) -> list[float]:
    return [math.cos(math.radians(angle)), math.sin(math.radians(angle))]


def test_below_threshold_pair_never_pushes_out_a_valid_one(tmp_path: Path) -> None:
    # Profiles at 0 and 30 degrees. Speaker 1 (8 deg) is closest to Ada (0.990) and also valid
    # for Bea (0.928); Speaker 2 (-18 deg) is valid only for Ada (0.951, Bea 0.669). Taking the
    # best pair first gives 1->Ada and leaves Speaker 2 with nothing. The exact assignment is
    # 1->Bea + 2->Ada (1.879 against 0.990).
    profiles = tmp_path / "p"
    profile(profiles, "Ada Finch", [deg(0)])
    profile(profiles, "Bea Lund", [deg(30)])
    folder = meeting(tmp_path / "m", {"Speaker 1": deg(8), "Speaker 2": deg(-18)})
    assert cosine(deg(8), deg(0)) == pytest.approx(0.990, abs=1e-3)
    assert cosine(deg(-18), deg(30)) == pytest.approx(0.669, abs=1e-3)
    assert run(folder, profiles) == 0
    found = suggestions(folder)
    assert found["Speaker 1"]["participant"] == "Bea Lund"
    assert found["Speaker 2"]["participant"] == "Ada Finch"


def test_null_centroid_is_no_suggestion(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    folder = meeting(
        tmp_path / "m",
        {"Speaker 1": None, "Speaker 2": [1.0, 0.0]},
        means={"Speaker 1": [1.0, 0.0]},
    )
    assert run(folder, profiles) == 0
    found = suggestions(folder)
    assert found["Speaker 1"]["participant"] is None
    assert found["Speaker 1"]["cosine"] is None
    assert found["Speaker 1"]["mean_of_windows_cosine"] is None
    assert found["Speaker 2"]["participant"] == "Zora Quill"


def test_mean_of_windows_cosine_is_recorded_and_never_decides(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    # The centroid matches, the mean of windows does not.
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0]}, means={"Speaker 1": [0.0, 1.0]})
    assert run(folder, profiles) == 0
    entry = suggestions(folder)["Speaker 1"]
    assert entry["participant"] == "Zora Quill"
    assert entry["mean_of_windows_cosine"] == 0.0
    # And the other way round: a perfect mean of windows cannot rescue a poor centroid.
    other = meeting(
        tmp_path / "n", {"Speaker 1": [0.0, 1.0]}, means={"Speaker 1": [1.0, 0.0]}, name="x"
    )
    assert run(other, profiles) == 0
    entry = suggestions(other)["Speaker 1"]
    assert entry["participant"] is None
    assert entry["mean_of_windows_cosine"] == 1.0


def test_profile_vector_is_the_unit_mean_of_its_samples(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    # Samples [1, 0] and [0, 1] average to the unit vector (1, 1)/sqrt 2.
    profile(profiles, "Zora Quill", [[1.0, 0.0], [0.0, 1.0]])
    folder = meeting(tmp_path / "m", {"Speaker 1": [1 / math.sqrt(2)] * 2})
    assert run(folder, profiles) == 0
    entry = suggestions(folder)["Speaker 1"]
    assert entry["participant"] == "Zora Quill"
    assert entry["cosine"] == pytest.approx(1.0)


def test_participant_is_the_stored_name_exactly(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "zORA  Quill", [[1.0, 0.0]])
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0]})
    assert run(folder, profiles) == 0
    assert suggestions(folder)["Speaker 1"]["participant"] == "zORA  Quill"


def test_profile_of_another_model_is_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    profile(profiles, "Bram Tulip", [[1.0, 0.0]], model="other-model")
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0]})
    assert run(folder, profiles) == 0
    assert suggestions(folder)["Speaker 1"]["participant"] == "Zora Quill"
    assert "1 profile(s) skipped" in capsys.readouterr().err


def test_every_profile_of_another_model_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]], model="other-model")
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0]})
    assert run(folder, profiles) == 2
    err = capsys.readouterr().err
    assert "embedding model" in err
    assert "embeddings.json" not in err
    assert not (folder / "suggestions.json").exists()


def test_missing_embeddings_exits_2_with_its_own_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    assert run(folder, tmp_path / "p") == 2
    err = capsys.readouterr().err
    assert "no embeddings.json" in err
    assert "embedding model" not in err
    assert not (folder / "suggestions.json").exists()


def test_unreadable_embeddings_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / EMBEDDINGS_NAME).write_text("{not json", encoding="utf-8")
    assert run(folder, tmp_path / "p") == 2
    assert "cannot read embeddings.json" in capsys.readouterr().err


def test_unknown_meeting_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["suggest", str(tmp_path / "none"), "--profiles-dir", str(tmp_path)]) == 2
    assert "no Meeting folder" in capsys.readouterr().err


@pytest.mark.parametrize("make_dir", [False, True])
def test_no_profiles_is_exit_0_with_no_suggestion_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_dir: bool
) -> None:
    profiles = tmp_path / "p"
    if make_dir:
        profiles.mkdir()
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0], "Speaker 2": None})
    assert run(folder, profiles) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Speaker 1: no suggestion",
        "Speaker 2: no suggestion",
    ]
    assert [e["participant"] for e in suggestions(folder).values()] == [None, None]


def test_writes_only_suggestions_json_with_the_contract_fields(tmp_path: Path) -> None:
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0]])
    folder = meeting(tmp_path / "m", {"Speaker 1": [1.0, 0.0]}, means={"Speaker 1": [1.0, 0.0]})
    before = {p.name for p in folder.iterdir()}
    assert run(folder, profiles) == 0
    assert {p.name for p in folder.iterdir()} - before == {"suggestions.json"}
    assert not list(folder.glob("transcript.*"))
    document = json.loads((folder / "suggestions.json").read_text(encoding="utf-8"))
    assert document["version"] == 1
    assert document["suggestions"] == [
        {
            "label": "Speaker 1",
            "participant": "Zora Quill",
            "cosine": 1.0,
            "mean_of_windows_cosine": 1.0,
            "model": MODEL,
        }
    ]


def test_no_vector_value_is_printed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profiles = tmp_path / "p"
    vec = [0.123456, 0.654321, math.sqrt(1 - 0.123456**2 - 0.654321**2)]
    profile(profiles, "Zora Quill", [vec])
    folder = meeting(tmp_path / "m", {"Speaker 1": vec}, means={"Speaker 1": vec})
    assert run(folder, profiles) == 0
    captured = capsys.readouterr()
    for value in vec:
        assert str(value)[:6] not in captured.out + captured.err
    assert "0.123456" not in captured.out


def test_ties_do_not_depend_on_label_order(tmp_path: Path) -> None:
    # Two Speakers with different centroids but equal cosine to the only profile.
    profiles = tmp_path / "p"
    profile(profiles, "Zora Quill", [[1.0, 0.0, 0.0]])
    a = [0.9, math.sqrt(1 - 0.81), 0.0]
    b = [0.9, 0.0, math.sqrt(1 - 0.81)]
    first = meeting(tmp_path / "a", {"Speaker 1": a, "Speaker 2": b})
    second = meeting(tmp_path / "b", {"Speaker 1": b, "Speaker 2": a})
    assert run(first, profiles) == run(second, profiles) == 0
    named_first = {k for k, v in suggestions(first).items() if v["participant"]}
    named_second = {k for k, v in suggestions(second).items() if v["participant"]}
    # The same voice gets the name in both Meetings, whatever its label.
    assert (named_first == {"Speaker 1"}) == (named_second == {"Speaker 2"})
