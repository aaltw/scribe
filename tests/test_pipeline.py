"""`scribe <recording>` end to end with stubbed adapters on a synthetic Recording.

Extract runs for real (ffmpeg on a generated tone); diarize and transcribe are stubs, so these
tests need no asr extra and no models. The Hugging Face cache points at a tmp dir, so the
environment checks behave the same on a Mac with the models cached and in CI.
"""

import json
import math
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scribe import diarize, transcribe
from scribe.cli import main
from scribe.merge import SpeakerSpan, Word
from scribe.pipeline import models_cached, prepare_environment

# Invented lines; the stub diarizer has two Speakers taking turns.
SPANS = [
    SpeakerSpan(0.0, 1.0, "SPEAKER_01"),
    SpeakerSpan(1.0, 2.0, "SPEAKER_00"),
]
WORDS = [
    Word(0.1, 0.4, "Pineapple"),
    Word(0.4, 0.8, "roadmap"),
    Word(0.8, 0.9, "?"),
    Word(1.2, 1.5, "Ja,"),
    Word(1.5, 1.9, "prima"),
]

DIM = 256

ENV_KEYS = ("HF_HUB_OFFLINE", "HF_HOME", "HUGGINGFACE_HUB_CACHE")


class Adapters:
    """Stub diarize and transcribe that record how they were called."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, Path, dict[str, Any]]] = []
        # What the stubs return; a test may replace them before running scribe.
        self.spans = SPANS
        self.words = WORDS
        self.centroids: dict[str, list[float]] = {}
        self.vectors: list[list[float]] = []
        monkeypatch.setattr(diarize, "diarize", self.diarize)
        monkeypatch.setattr(transcribe, "transcribe", self.transcribe)
        monkeypatch.setattr(diarize, "model_version", lambda: "stub-diarizer@1")
        monkeypatch.setattr(transcribe, "model_version", lambda: "stub-asr@1")

    def diarize(self, audio: Path, **kwargs: Any) -> diarize.Diarization:
        self.calls.append(("diarize", audio, kwargs))
        return diarize.Diarization(
            spans=self.spans, centroids=self.centroids, duration=60.0, embed=self.embed
        )

    def embed(self, starts: Sequence[float], length: float) -> list[list[float]]:
        self.calls.append(("embed", Path(), {"starts": list(starts), "length": length}))
        return self.vectors[: len(starts)]

    def transcribe(self, audio: Path, **kwargs: Any) -> list[Word]:
        self.calls.append(("transcribe", audio, kwargs))
        return self.words

    @property
    def order(self) -> list[str]:
        return [stage for stage, _, _ in self.calls]

    def kwargs(self, stage: str) -> dict[str, Any]:
        return next(kwargs for name, _, kwargs in self.calls if name == stage)


@pytest.fixture(autouse=True)
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty Hugging Face cache, and none of the variables the pipeline sets."""
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "")  # record the original so teardown restores it
        monkeypatch.delenv(key)
    cache = tmp_path / "hf-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    return cache


@pytest.fixture
def adapters(monkeypatch: pytest.MonkeyPatch) -> Adapters:
    return Adapters(monkeypatch)


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    path = tmp_path / "2026-10-01 Pineapple Sync.m4a"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=2:sample_rate=48000", "-c:a", "aac", str(path)],
        check=True,
    )  # fmt: skip
    return path


def scribe(*args: str | Path) -> int:
    return main([str(a) for a in args])


def test_writes_transcript_md_and_json(
    tmp_path: Path, recording: Path, adapters: Adapters, capsys: pytest.CaptureFixture[str]
) -> None:
    meetings = tmp_path / "meetings"

    assert scribe(recording, "--meetings-dir", meetings) == 0

    folder = meetings / "2026-10-01-pineapple-sync"
    out, err = capsys.readouterr()
    assert out.strip() == str(folder)
    assert (folder / "audio.wav").is_file()
    assert adapters.calls[0][1] == folder / "audio.wav"
    document = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    assert document["speakers"] == ["Speaker 1", "Speaker 2"]
    assert document["models"] == {"diarize": "stub-diarizer@1", "transcribe": "stub-asr@1"}
    assert [(s["speaker"], s["text"]) for s in document["segments"]] == [
        ("Speaker 1", "Pineapple roadmap?"),
        ("Speaker 2", "Ja, prima"),
    ]
    assert (folder / "transcript.md").read_text(encoding="utf-8") == (
        "**Speaker 1** [00:00]\nPineapple roadmap?\n\n**Speaker 2** [00:01]\nJa, prima\n"
    )
    assert "2 Speakers detected, 2 in the Transcript, 2 Segments" in err
    for stage in ("extract", "diarize", "embed", "transcribe", "merge", "write", "total"):
        assert f"  {stage} " in err


def random_vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


def test_writes_embeddings_json_keyed_by_transcript_label(
    tmp_path: Path, recording: Path, adapters: Adapters, capsys: pytest.CaptureFixture[str]
) -> None:
    rng = random.Random(78)
    # SPEAKER_02 talks first and gets a 3 s window but its row was all zero (dropped by the
    # diarizer); SPEAKER_00 has a row but no 3 s turn; SPEAKER_01 has spans but no words, so
    # merge drops it and it must not reach the file.
    adapters.spans = [
        SpeakerSpan(0.0, 4.0, "SPEAKER_02"),
        SpeakerSpan(4.0, 5.0, "SPEAKER_00"),
        SpeakerSpan(5.0, 12.0, "SPEAKER_01"),
    ]
    adapters.words = [Word(0.5, 1.0, "Pineapple"), Word(4.2, 4.6, "Ja")]
    adapters.centroids = {"SPEAKER_00": random_vector(rng), "SPEAKER_01": random_vector(rng)}
    adapters.vectors = [random_vector(rng) for _ in range(3)]
    meetings = tmp_path / "meetings"

    assert scribe(recording, "--meetings-dir", meetings) == 0

    out, err = capsys.readouterr()
    folder = meetings / "2026-10-01-pineapple-sync"
    # Windows: one in SPEAKER_02's turn, two in SPEAKER_01's, none in SPEAKER_00's 1 s turn.
    assert adapters.kwargs("embed") == {"starts": [0.25, 5.25, 8.25], "length": 3.0}
    document = json.loads((folder / "embeddings.json").read_text(encoding="utf-8"))
    assert set(document) == {"version", "window_rule", "speakers"}
    assert document["version"] == 1
    assert document["window_rule"] == {"window_s": 3.0, "trim_s": 0.25}
    assert list(document["speakers"]) == ["Speaker 1", "Speaker 2"]
    first, second = document["speakers"]["Speaker 1"], document["speakers"]["Speaker 2"]
    for entry in (first, second):
        assert set(entry) == {"model", "centroid", "mean_of_windows", "windows"}
        assert entry["model"] == "stub-diarizer@1"
    assert first["centroid"] is None
    assert first["windows"] == 1
    assert math.isclose(math.hypot(*first["mean_of_windows"]), 1.0)
    assert len(first["mean_of_windows"]) == DIM
    assert second["mean_of_windows"] is None
    assert second["windows"] == 0
    assert math.isclose(math.hypot(*second["centroid"]), 1.0)

    # No vector value, raw or written, reaches stdout or stderr.
    values = [x for v in (*adapters.centroids.values(), *adapters.vectors) for x in v]
    values += first["mean_of_windows"] + second["centroid"]
    for value in values:
        for text in (repr(value), f"{value:.4f}"):
            assert text not in out
            assert text not in err


def test_without_speakers_no_hint_reaches_the_diarizer(
    tmp_path: Path, recording: Path, adapters: Adapters
) -> None:
    assert scribe(recording, "--meetings-dir", tmp_path / "meetings") == 0
    assert adapters.kwargs("diarize") == {"speakers": None}


def test_speakers_3_reaches_the_diarizer_as_3(
    tmp_path: Path, recording: Path, adapters: Adapters
) -> None:
    assert scribe(recording, "--speakers", "3", "--meetings-dir", tmp_path / "meetings") == 0
    assert adapters.kwargs("diarize") == {"speakers": 3}


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_speakers_must_be_a_positive_whole_number(
    tmp_path: Path, recording: Path, adapters: Adapters, value: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        scribe(recording, "--speakers", value, "--meetings-dir", tmp_path / "meetings")
    assert exc.value.code == 2
    assert adapters.calls == []


def test_diarize_runs_before_transcribe(
    tmp_path: Path, recording: Path, adapters: Adapters
) -> None:
    # torch (imported by diarize) must load before parakeet_mlx, for the MPS fallback.
    scribe(recording, "--meetings-dir", tmp_path / "meetings")
    assert adapters.order == ["diarize", "embed", "transcribe"]


def test_explicit_run_and_leading_options_are_the_same_command(
    tmp_path: Path, recording: Path, adapters: Adapters
) -> None:
    meetings = tmp_path / "meetings"
    assert scribe("run", recording, "--name", "a", "--meetings-dir", meetings) == 0
    assert scribe("--speakers", "2", "--name", "b", recording, "--meetings-dir", meetings) == 0
    assert (meetings / "2026-10-01-a" / "transcript.json").is_file()
    assert (meetings / "2026-10-01-b" / "transcript.json").is_file()
    hints = [kwargs for stage, _, kwargs in adapters.calls if stage == "diarize"]
    assert hints == [{"speakers": None}, {"speakers": 2}]


def test_rerun_is_refused_before_any_work_and_force_overwrites(
    tmp_path: Path, recording: Path, adapters: Adapters, capsys: pytest.CaptureFixture[str]
) -> None:
    meetings = tmp_path / "meetings"
    assert scribe(recording, "--meetings-dir", meetings) == 0
    json_path = meetings / "2026-10-01-pineapple-sync" / "transcript.json"
    json_path.write_text("named earlier", encoding="utf-8")
    adapters.calls.clear()
    capsys.readouterr()

    assert scribe(recording, "--meetings-dir", meetings) == 1

    assert "already has transcript.json and transcript.md" in capsys.readouterr().err
    assert adapters.calls == []
    assert json_path.read_text(encoding="utf-8") == "named earlier"

    assert scribe(recording, "--force", "--meetings-dir", meetings) == 0
    assert json.loads(json_path.read_text(encoding="utf-8"))["version"] == 1


def test_missing_recording_fails_with_readable_message(
    tmp_path: Path, adapters: Adapters, capsys: pytest.CaptureFixture[str]
) -> None:
    assert scribe(tmp_path / "nope.mp4", "--meetings-dir", tmp_path / "meetings") == 1
    assert "Recording not found" in capsys.readouterr().err
    assert adapters.calls == []


def test_no_arguments_prints_usage_and_fails(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage: scribe" in capsys.readouterr().err


def fill_cache(cache: Path, skip: str | None = None) -> None:
    snapshots = {
        "models--pyannote--speaker-diarization-community-1": (
            diarize.MODEL_REVISION,
            ["config.yaml", "embedding/pytorch_model.bin", "segmentation/pytorch_model.bin",
             "plda/plda.npz", "plda/xvec_transform.npz"],
        ),
        "models--mlx-community--parakeet-tdt-0.6b-v3": (
            transcribe.MODEL_REVISION,
            ["config.json", "model.safetensors"],
        ),
    }  # fmt: skip
    for repo, (revision, files) in snapshots.items():
        for name in files:
            if name == skip:
                continue
            path = cache / repo / "snapshots" / revision / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")


def test_online_when_nothing_is_cached() -> None:
    env = {"HF_HUB_CACHE": "/nonexistent"}
    prepare_environment(env)
    assert "HF_HUB_OFFLINE" not in env


def test_offline_when_both_snapshots_are_cached(clean_env: Path) -> None:
    fill_cache(clean_env)
    env = {"HF_HUB_CACHE": str(clean_env)}
    prepare_environment(env)
    assert env["HF_HUB_OFFLINE"] == "1"


@pytest.mark.parametrize("missing", ["plda/xvec_transform.npz", "model.safetensors"])
def test_online_when_a_snapshot_file_is_missing(clean_env: Path, missing: str) -> None:
    fill_cache(clean_env, skip=missing)
    env = {"HF_HUB_CACHE": str(clean_env)}
    prepare_environment(env)
    assert "HF_HUB_OFFLINE" not in env


def test_dangling_snapshot_link_counts_as_missing(clean_env: Path) -> None:
    fill_cache(clean_env)
    snapshot = clean_env / "models--mlx-community--parakeet-tdt-0.6b-v3" / "snapshots"
    weights = snapshot / transcribe.MODEL_REVISION / "model.safetensors"
    weights.unlink()
    weights.symlink_to(clean_env / "blobs" / "gone")
    assert not models_cached({"HF_HUB_CACHE": str(clean_env)})


def test_explicit_offline_setting_is_kept(clean_env: Path) -> None:
    fill_cache(clean_env)
    env = {"HF_HUB_CACHE": str(clean_env), "HF_HUB_OFFLINE": "0"}
    prepare_environment(env)
    assert env["HF_HUB_OFFLINE"] == "0"


def test_cache_follows_hf_home_then_xdg(tmp_path: Path) -> None:
    fill_cache(tmp_path / "hf" / "hub")
    assert models_cached({"HF_HOME": str(tmp_path / "hf")})
    fill_cache(tmp_path / "xdg" / "huggingface" / "hub")
    assert models_cached({"XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert not models_cached({"XDG_CACHE_HOME": str(tmp_path / "elsewhere")})
