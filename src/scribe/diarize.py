"""Pipeline step 2: find the Speakers in a 16 kHz mono WAV with pyannote community-1.

Needs the `asr` extra. torch and pyannote are imported inside `diarize`, so this module, the
core package and the CLI import without them.

Public names: `MODEL_ID`, `MODEL_REVISION`, `DiarizeError`, `Diarization`, `diarize`,
`embedding_rows`, `model_version`.

Embeddings: besides the spans, `diarize` returns pyannote's `speaker_embeddings` row per
raw label and an `embed` function that embeds 3 s windows with the same, already loaded
embedding model, so the embed stage needs no second model load. The rows follow
`speaker_diarization.labels()`, not the exclusive labels, and pyannote pads all-zero rows when
it has fewer centroids than labels (pyannote.audio 4.0.7, `speaker_diarization.py`), so `embedding_rows` reads them in that order and drops zero rows.
A label that only speaks in overlap keeps its row but has no exclusive span.

Device: MPS when available, else CPU; the choice is logged. On CPU, diarize alone took 0.72-2.6x
real time on an Apple M2 Pro, so a 30-minute Recording breaks the budget there.
MPS runs need `PYTORCH_ENABLE_MPS_FALLBACK=1` for the ops MPS lacks. torch reads that variable
once, when it is first imported, so `diarize` sets it before importing torch and warns when
torch was already imported without it.

Telemetry: pyannote.audio 4 sends each file's duration and Speaker count to otel.pyannote.ai by
default, and huggingface_hub has its own usage telemetry. Nothing derived from a Recording may
leave the machine (ADR 0001), so `diarize` sets `PYANNOTE_METRICS_ENABLED=false` and
`HF_HUB_DISABLE_TELEMETRY=1` on every call, overriding any earlier value, before it imports
pyannote.audio (which imports huggingface_hub). Both libraries read them at import or per call.

The audio goes to pyannote in memory (`{"waveform", "sample_rate"}`): pyannote 4's torchcodec
file decoding is unproven with the locked torch, torchaudio and torchcodec versions.
"""

import logging
import os
import sys
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from scribe.merge import SpeakerSpan

MODEL_ID = "pyannote/speaker-diarization-community-1"
# The Hugging Face snapshot scribe is pinned to and was measured with.
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
SAMPLE_RATE = 16_000
MPS_FALLBACK = "PYTORCH_ENABLE_MPS_FALLBACK"
# Set on every call, before pyannote.audio and huggingface_hub are imported (module docstring).
TELEMETRY_OFF = {"PYANNOTE_METRICS_ENABLED": "false", "HF_HUB_DISABLE_TELEMETRY": "1"}

logger = logging.getLogger(__name__)


# Window embedding batch size.
_BATCH = 32


class DiarizeError(Exception):
    """Diarization failed for a reason worth showing the user as-is."""


@dataclass(frozen=True)
class Diarization:
    """What one diarize run found, plus the means to embed windows of the same audio."""

    # Exclusive spans in start order, raw labels.
    spans: list[SpeakerSpan]
    # pyannote's speaker_embeddings row per raw label, all-zero rows dropped.
    centroids: dict[str, list[float]]
    # Length of the audio in seconds.
    duration: float
    # embed(starts, length): one embedding per window [start, start + length) in seconds, from
    # the pipeline's own embedding model. Holds the loaded pipeline; drop it when done.
    embed: Callable[[Sequence[float], float], list[list[float]]]


def model_version() -> str:
    """`<model id>@<revision> (pyannote.audio <version>)`, for transcript.json's models."""
    try:
        library = version("pyannote.audio")
    except PackageNotFoundError as error:
        raise DiarizeError(_MISSING_EXTRA) from error
    return f"{MODEL_ID}@{MODEL_REVISION} (pyannote.audio {library})"


def diarize(audio: Path, speakers: int | None = None) -> Diarization:
    """Exclusive Speaker spans of a 16 kHz mono 16-bit WAV, in start order, with embeddings.

    Exclusive means no two spans overlap: where pyannote hears overlapping speech, one Speaker
    keeps the time. Labels are pyannote's raw ones (`SPEAKER_00`, ...). `speakers` is an
    optional hint for the number of Speakers; without it pyannote detects the count.
    """
    if speakers is not None and speakers < 1:
        raise DiarizeError(f"speaker count must be at least 1, got {speakers}")
    if "torch" in sys.modules and os.environ.get(MPS_FALLBACK) != "1":
        logger.warning("torch was imported before %s=1 was set; MPS may lack ops", MPS_FALLBACK)
    os.environ.setdefault(MPS_FALLBACK, "1")
    os.environ.update(TELEMETRY_OFF)
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise DiarizeError(_MISSING_EXTRA) from error

    frames = _read_wav(audio)
    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16).to(torch.float32) / 32768.0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("diarize: pyannote on %s", device)
    pipeline = Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if pipeline is None:
        raise DiarizeError(f"could not load {MODEL_ID}; is the Hugging Face token set?")
    pipeline.to(torch.device(device))

    output: Any = pipeline(
        {"waveform": waveform[None], "sample_rate": SAMPLE_RATE}, num_speakers=speakers
    )
    spans = [
        SpeakerSpan(start=float(turn.start), end=float(turn.end), speaker=str(label))
        for turn, _, label in output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ]

    def embed(starts: Sequence[float], length: float) -> list[list[float]]:
        n = round(length * SAMPLE_RATE)
        pieces = [waveform[round(start * SAMPLE_RATE) :][:n] for start in starts]
        if any(len(piece) != n for piece in pieces):
            raise DiarizeError("a window runs past the end of the audio")
        vectors: list[list[float]] = []
        for i in range(0, len(pieces), _BATCH):
            batch = torch.stack(pieces[i : i + _BATCH])[:, None, :]
            vectors.extend(pipeline._embedding(batch).tolist())
        return vectors

    return Diarization(
        spans=sorted(spans, key=lambda s: (s.start, s.end)),
        centroids=embedding_rows(output),
        duration=len(waveform) / SAMPLE_RATE,
        embed=embed,
    )


def embedding_rows(output: Any) -> dict[str, list[float]]:
    """pyannote's `speaker_embeddings` rows by raw label, read in `speaker_diarization.labels()`
    order, with all-zero (padded) rows dropped. Empty when pyannote gave no embeddings."""
    rows = output.speaker_embeddings
    if rows is None:
        return {}
    labels = [str(label) for label in output.speaker_diarization.labels()]
    table = [[float(x) for x in row] for row in rows]
    if len(table) != len(labels):
        raise DiarizeError(f"pyannote gave {len(table)} embedding rows for {len(labels)} Speakers")
    return {label: row for label, row in zip(labels, table, strict=True) if any(row)}


_MISSING_EXTRA = (
    "diarize needs the asr extra (macOS only): rerun with `uv run --extra asr scribe ...`"
)


def _read_wav(audio: Path) -> bytes:
    """The PCM frames of a 16 kHz mono 16-bit WAV, as `scribe extract` writes it."""
    try:
        with wave.open(str(audio), "rb") as f:
            shape = (f.getframerate(), f.getnchannels(), f.getsampwidth())
            frames = f.readframes(f.getnframes())
    except (OSError, EOFError, wave.Error) as error:
        raise DiarizeError(f"cannot read {audio}: {error}") from error
    if shape != (SAMPLE_RATE, 1, 2):
        rate, channels, width = shape
        raise DiarizeError(
            f"{audio.name} is {rate} Hz, {channels} channel(s), {8 * width}-bit; "
            "expected 16 kHz mono 16-bit (run scribe extract)"
        )
    if not frames:
        raise DiarizeError(f"{audio.name} holds no audio")
    return frames
