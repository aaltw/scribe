"""Pipeline embed step: per-Speaker voice embeddings in `<meeting>/embeddings.json`.

Pure Python, no numpy: CI runs without the asr extra. The diarizer supplies two things per raw
pyannote label, both from the pipeline's own embedding model (WeSpeaker ResNet34, 256
dimensions), and this module turns them into the file that `profile save` and the
suggestion step read:

- the **centroid**: pyannote's `speaker_embeddings` row for the label;
- the **mean of windows**: the mean of the label's unit 3 s window embeddings, made unit again.
  Measurement found it separates two Speakers better than the
  centroid (train gap +0.117 against -0.129).

Window rule: a window lies inside one exclusive turn, never across a
turn boundary. `TRIM_S` is cut off both turn edges, then the trimmed turn is tiled from its
start with non-overlapping `WINDOW_S` windows and the remainder is dropped. A window running
past the end of the audio is dropped too.

Public names: `FILE_NAME`, `VERSION`, `WINDOW_S`, `TRIM_S`, `Window`, `windows`, `unit`,
`mean_of_windows`, `build`, `write`.

File format (version 1), keyed by Transcript label in first-heard order:

    {"version": 1,
     "window_rule": {"window_s": 3.0, "trim_s": 0.25},
     "speakers": {"Speaker 1": {"model": "<diarize.model_version()>",
                                "centroid": [256 floats] | null,
                                "mean_of_windows": [256 floats] | null,
                                "windows": <count>}}}

Both vectors are unit length. `centroid` is null when pyannote gave the label an all-zero
(padded) row; `mean_of_windows` is null when the label has no usable 3 s window, and `windows`
then is 0. A raw label that merge drops (spans but no words) has no Transcript label, so it is
not in the file. Labels are the Transcript's `Speaker N` as the pipeline wrote them; Naming
later does not rewrite this file.

Embeddings are biometric data (ADR 0001): nothing here prints a vector value.
"""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scribe.merge import SpeakerSpan

FILE_NAME = "embeddings.json"
VERSION = 1
WINDOW_S = 3.0
TRIM_S = 0.25


@dataclass(frozen=True)
class Window:
    """One `WINDOW_S` window inside an exclusive turn: start in seconds, raw pyannote label."""

    start: float
    speaker: str


def windows(spans: Sequence[SpeakerSpan], duration: float) -> list[Window]:
    """The windows of every exclusive turn, in span order (module docstring, window rule)."""
    out = []
    for span in spans:
        start, end = span.start + TRIM_S, min(span.end, duration) - TRIM_S
        k = 0
        while start + (k + 1) * WINDOW_S <= end:
            out.append(Window(start=start + k * WINDOW_S, speaker=span.speaker))
            k += 1
    return out


def unit(vector: Sequence[float]) -> list[float] | None:
    """`vector` scaled to length 1; None for a zero or non-finite vector."""
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0 or not math.isfinite(norm):
        return None
    return [x / norm for x in vector]


def mean_of_windows(vectors: Sequence[Sequence[float]]) -> tuple[list[float] | None, int]:
    """The unit mean of the unit window vectors, and how many windows went into it.

    Zero or non-finite window vectors are skipped and not counted.
    """
    units = [u for u in map(unit, vectors) if u is not None]
    if not units:
        return None, 0
    mean = [math.fsum(column) / len(units) for column in zip(*units, strict=True)]
    return unit(mean), len(units)


def build(
    labels: Mapping[str, str],
    centroids: Mapping[str, Sequence[float]],
    cut: Sequence[Window],
    vectors: Sequence[Sequence[float]],
    model: str,
) -> dict[str, Any]:
    """The embeddings.json document.

    `labels` is merge's raw-to-Transcript-label map, `centroids` the diarizer's rows by raw
    label (all-zero rows already dropped), `vectors` one embedding per window of `cut`.
    """
    if len(cut) != len(vectors):
        raise ValueError(f"{len(cut)} windows but {len(vectors)} window embeddings")
    speakers: dict[str, Any] = {}
    for raw, label in labels.items():
        row = centroids.get(raw)
        mean, count = mean_of_windows(
            [v for w, v in zip(cut, vectors, strict=True) if w.speaker == raw]
        )
        speakers[label] = {
            "model": model,
            "centroid": unit(row) if row is not None else None,
            "mean_of_windows": mean,
            "windows": count,
        }
    return {
        "version": VERSION,
        "window_rule": {"window_s": WINDOW_S, "trim_s": TRIM_S},
        "speakers": speakers,
    }


def write(folder: Path, document: Mapping[str, Any]) -> Path:
    """Write `document` as `<folder>/embeddings.json`; returns its path."""
    path = folder / FILE_NAME
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path
