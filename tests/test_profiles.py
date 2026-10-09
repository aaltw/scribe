"""Voice profile tests: synthetic Meetings and random vectors under tmp_path, invented names."""

import json
import random
import subprocess
from pathlib import Path

import pytest

from scribe.cli import SUBCOMMANDS, main
from scribe.embeddings import FILE_NAME as EMBEDDINGS_NAME
from scribe.transcript import Segment, Transcript, write_json

MODEL = "test-model@1 (pyannote.audio 0)"
DIMS = 16


def vector(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIMS)]


def make_meeting(
    root: Path,
    name: str = "2026-10-01-demo",
    *,
    names: dict[str, str] | None = None,
    windows: dict[str, int] | None = None,
    model: str = MODEL,
    embeddings: bool = True,
) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    names = {"Speaker 1": "Zora Quill", "Speaker 2": "Bram Tulip"} if names is None else names
    windows = {"Speaker 1": 9, "Speaker 2": 6} if windows is None else windows
    transcript = Transcript(
        segments=(
            Segment(0.0, 4.0, "Speaker 1", "invented one", names.get("Speaker 1")),
            Segment(4.0, 8.0, "Speaker 2", "invented two", names.get("Speaker 2")),
        ),
        speakers=("Speaker 1", "Speaker 2"),
        models={"asr": "test-1"},
    )
    write_json(transcript, folder / "transcript.json")
    if embeddings:
        speakers = {
            label: {
                "model": model,
                "centroid": vector(100 + i),
                "mean_of_windows": vector(i) if count else None,
                "windows": count,
            }
            for i, (label, count) in enumerate(windows.items(), start=1)
        }
        document = {"version": 1, "window_rule": {}, "speakers": speakers}
        (folder / EMBEDDINGS_NAME).write_text(json.dumps(document), encoding="utf-8")
    return folder


def save(folder: Path, who: str, profiles: Path) -> int:
    return main(["profile", "save", str(folder), who, "--profiles-dir", str(profiles)])


def test_profile_is_a_subcommand() -> None:
    assert "profile" in SUBCOMMANDS


def test_default_profiles_dir_is_gitignored() -> None:
    repo = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "check-ignore", "-q", ".scribe/profiles/x.json"], cwd=repo, check=False
    )
    assert result.returncode == 0


def test_save_writes_sample_with_mean_of_windows(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path / "meetings")
    profiles = tmp_path / "profiles"
    assert save(folder, "Zora Quill", profiles) == 0
    profile = json.loads((profiles / "zora-quill.json").read_text(encoding="utf-8"))
    assert profile["name"] == "Zora Quill"
    assert profile["samples"] == [
        {"source": "2026-10-01-demo", "vector": vector(1), "windows": 9, "model": MODEL}
    ]


def test_save_slug_and_match_are_case_insensitive(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path / "meetings")
    profiles = tmp_path / "profiles"
    assert save(folder, "  zORA quill ", profiles) == 0
    assert [p.name for p in profiles.iterdir()] == ["zora-quill.json"]


def test_profile_name_is_the_transcript_spelling_and_later_saves_keep_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles"
    first = make_meeting(tmp_path / "meetings", "2026-10-01-a")
    second = make_meeting(tmp_path / "meetings", "2026-10-02-b", names={"Speaker 1": "ZORA QUILL"})
    assert save(first, "zORA quill", profiles) == 0
    assert save(second, "Zora Quill", profiles) == 0
    capsys.readouterr()
    assert main(["profile", "list", "--profiles-dir", str(profiles)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Zora Quill: 2 sample(s) from 2026-10-01-a, 2026-10-02-b"
    ]


def test_saving_again_from_same_meeting_replaces_the_sample(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path / "meetings")
    profiles = tmp_path / "profiles"
    assert save(folder, "Zora Quill", profiles) == 0
    document = json.loads((folder / EMBEDDINGS_NAME).read_text(encoding="utf-8"))
    document["speakers"]["Speaker 1"]["windows"] = 12
    (folder / EMBEDDINGS_NAME).write_text(json.dumps(document), encoding="utf-8")
    assert save(folder, "Zora Quill", profiles) == 0
    samples = json.loads((profiles / "zora-quill.json").read_text(encoding="utf-8"))["samples"]
    assert [(s["source"], s["windows"]) for s in samples] == [("2026-10-01-demo", 12)]


def test_second_meeting_adds_a_sample(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    first = make_meeting(tmp_path / "meetings", "2026-10-01-a")
    second = make_meeting(tmp_path / "meetings", "2026-10-02-b")
    assert save(first, "Zora Quill", profiles) == 0
    assert save(second, "Zora Quill", profiles) == 0
    samples = json.loads((profiles / "zora-quill.json").read_text(encoding="utf-8"))["samples"]
    assert [s["source"] for s in samples] == ["2026-10-01-a", "2026-10-02-b"]


def test_meeting_name_resolves_under_meetings_dir(tmp_path: Path) -> None:
    make_meeting(tmp_path / "meetings")
    profiles = tmp_path / "profiles"
    code = main(
        [
            "profile",
            "save",
            "2026-10-01-demo",
            "Bram Tulip",
            "--meetings-dir",
            str(tmp_path / "meetings"),
            "--profiles-dir",
            str(profiles),
        ]
    )
    assert code == 0
    assert (profiles / "bram-tulip.json").is_file()


def test_list_prints_name_count_and_sources_never_vectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles"
    first = make_meeting(tmp_path / "meetings", "2026-10-01-a")
    second = make_meeting(tmp_path / "meetings", "2026-10-02-b")
    save(first, "Zora Quill", profiles)
    save(second, "Zora Quill", profiles)
    save(first, "Bram Tulip", profiles)
    capsys.readouterr()
    assert main(["profile", "list", "--profiles-dir", str(profiles)]) == 0
    out = capsys.readouterr()
    assert out.out.splitlines() == [
        "Bram Tulip: 1 sample(s) from 2026-10-01-a",
        "Zora Quill: 2 sample(s) from 2026-10-01-a, 2026-10-02-b",
    ]
    for value in vector(1) + vector(2):
        assert str(value) not in out.out + out.err
        assert f"{value:.4f}" not in out.out + out.err


def test_list_without_profiles_dir_prints_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["profile", "list", "--profiles-dir", str(tmp_path / "none")]) == 0
    assert capsys.readouterr().out == ""


def refused(capsys: pytest.CaptureFixture[str], needle: str) -> None:
    err = capsys.readouterr().err
    assert err.startswith("scribe: error: ")
    assert needle in err


def test_refuses_participant_not_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = make_meeting(tmp_path / "meetings")
    profiles = tmp_path / "profiles"
    assert save(folder, "Nobody Here", profiles) == 2
    refused(capsys, "run scribe name")
    assert not profiles.exists()


def test_refuses_meeting_without_embeddings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path / "meetings", embeddings=False)
    assert save(folder, "Zora Quill", tmp_path / "profiles") == 2
    refused(capsys, "--force")


def test_refuses_fewer_than_four_windows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(tmp_path / "meetings", windows={"Speaker 1": 3, "Speaker 2": 6})
    assert save(folder, "Zora Quill", tmp_path / "profiles") == 2
    refused(capsys, "fewer than the 4")


def test_four_windows_is_enough(tmp_path: Path) -> None:
    folder = make_meeting(tmp_path / "meetings", windows={"Speaker 1": 4, "Speaker 2": 6})
    assert save(folder, "Zora Quill", tmp_path / "profiles") == 0


def test_refuses_null_mean_of_windows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    folder = make_meeting(tmp_path / "meetings", windows={"Speaker 1": 0, "Speaker 2": 6})
    assert save(folder, "Zora Quill", tmp_path / "profiles") == 2
    refused(capsys, "fewer than the 4")


def test_refuses_model_differing_from_existing_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles"
    first = make_meeting(tmp_path / "meetings", "2026-10-01-a")
    other = make_meeting(tmp_path / "meetings", "2026-10-02-b", model="other-model@2")
    assert save(first, "Zora Quill", profiles) == 0
    before = (profiles / "zora-quill.json").read_text(encoding="utf-8")
    assert save(other, "Zora Quill", profiles) == 2
    refused(capsys, "different model")
    assert (profiles / "zora-quill.json").read_text(encoding="utf-8") == before


def test_resaving_the_only_sample_may_change_the_model(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    folder = make_meeting(tmp_path / "meetings")
    assert save(folder, "Zora Quill", profiles) == 0
    document = json.loads((folder / EMBEDDINGS_NAME).read_text(encoding="utf-8"))
    document["speakers"]["Speaker 1"]["model"] = "newer-model@3"
    (folder / EMBEDDINGS_NAME).write_text(json.dumps(document), encoding="utf-8")
    assert save(folder, "Zora Quill", profiles) == 0
    samples = json.loads((profiles / "zora-quill.json").read_text(encoding="utf-8"))["samples"]
    assert [s["model"] for s in samples] == ["newer-model@3"]


def test_refuses_participant_named_on_two_speakers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = make_meeting(
        tmp_path / "meetings", names={"Speaker 1": "Zora Quill", "Speaker 2": "zora quill"}
    )
    profiles = tmp_path / "profiles"
    assert save(folder, "Zora Quill", profiles) == 2
    refused(capsys, "named on 2 Speakers")
    assert not profiles.exists()


def test_refuses_missing_meeting_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(
        ["profile", "save", str(tmp_path / "nope"), "Zora Quill", "--profiles-dir", str(tmp_path)]
    )
    assert code == 2
    refused(capsys, "fix --meetings-dir")


def test_refuses_unreadable_profile_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "zora-quill.json").write_text("not json", encoding="utf-8")
    folder = make_meeting(tmp_path / "meetings")
    assert save(folder, "Zora Quill", profiles) == 2
    refused(capsys, "move it away")
