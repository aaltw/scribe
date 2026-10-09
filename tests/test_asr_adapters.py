"""Both adapters on a synthetic 10 s clip of two `say` voices. macOS with the asr extra only.

The models must already be in the Hugging Face cache: HF_HUB_OFFLINE=1 makes a download fail
instead of happening. Run alone, one model job at a time:

    uv run pytest tests/test_asr_adapters.py -v --durations=0
"""

import logging
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from itertools import pairwise
from pathlib import Path

import pytest

# huggingface_hub reads this once, when first imported; the adapters import it lazily.
os.environ["HF_HUB_OFFLINE"] = "1"

from scribe.diarize import diarize
from scribe.merge import Word
from scribe.transcribe import transcribe

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or find_spec("parakeet_mlx") is None
    or find_spec("pyannote") is None
    or shutil.which("say") is None
    or shutil.which("ffmpeg") is None,
    reason="needs macOS, the asr extra, say and ffmpeg",
)

CLIP_S = 10.0
# Invented lines, one per voice, with a pause between them.
LINES = [
    ("Samantha", "Good morning, shall we look at the pineapple roadmap today?"),
    ("Daniel", "Yes, the budget for the harbour project is ready."),
]


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("clip")
    parts = []
    for i, (voice, text) in enumerate(LINES):
        part = folder / f"{i}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(part), text], check=True)
        parts.append(part)
    output = folder / "audio.wav"
    graph = (
        "[0:a]aresample=16000,apad=pad_dur=1.0[a];[1:a]aresample=16000[b];"
        f"[a][b]concat=n=2:v=0:a=1,apad=whole_dur={CLIP_S},atrim=0:{CLIP_S}"
    )
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(parts[0]), "-i", str(parts[1]), "-filter_complex", graph,
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)],
        check=True,
    )  # fmt: skip
    return output


def test_diarize_finds_two_speakers_on_mps(clip: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="scribe.diarize"):
        diarization = diarize(clip, speakers=2)
    spans = diarization.spans

    assert "diarize: pyannote on mps" in caplog.text
    assert {s.speaker for s in spans} == {"SPEAKER_00", "SPEAKER_01"}
    assert all(0.0 <= s.start < s.end <= CLIP_S for s in spans)
    assert all(a.end <= b.start for a, b in pairwise(spans))
    # The first voice talks first, the second after the pause.
    assert spans[0].speaker != spans[-1].speaker
    # One 256-dim row per Speaker, and the loaded model embeds a 3 s window into the same space.
    assert set(diarization.centroids) == {"SPEAKER_00", "SPEAKER_01"}
    assert all(len(row) == 256 for row in diarization.centroids.values())
    assert diarization.duration == pytest.approx(CLIP_S, abs=0.01)
    vectors = diarization.embed([0.0, 1.0], 3.0)
    assert [len(v) for v in vectors] == [256, 256]
    assert _huggingface_offline()


def test_transcribe_returns_timed_words(clip: Path) -> None:
    words = transcribe(clip)

    _check_words(words)
    texts = [w.text.lower() for w in words]
    assert "pineapple" in texts
    assert "harbour" in texts or "harbor" in texts
    assert "?" in texts
    assert _huggingface_offline()


def test_transcribe_through_chunk_merge(clip: Path) -> None:
    # A 30-minute Recording always goes through parakeet-mlx's chunk merge; 10 s at the
    # defaults never does, so force small chunks here.
    words = transcribe(clip, chunk_s=4.0, overlap_s=1.0)

    _check_words(words)
    assert "pineapple" in [w.text.lower() for w in words]


def _check_words(words: list[Word]) -> None:
    assert words
    assert all(0.0 <= w.start <= w.end <= CLIP_S for w in words)
    assert all(a.start <= b.start for a, b in pairwise(words))
    assert all(w.text and not any(c.isspace() for c in w.text) for w in words)


def _huggingface_offline() -> bool:
    """The adapters' huggingface_hub, imported by now, saw HF_HUB_OFFLINE=1."""
    return bool(sys.modules["huggingface_hub"].constants.HF_HUB_OFFLINE)
