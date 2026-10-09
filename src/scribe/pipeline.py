"""The full pipeline: extract, diarize, embed, transcribe, merge and write one Meeting.

`run` is what `uv run scribe <recording>` calls. It writes `audio.wav`, `transcript.json`,
`transcript.md` and `embeddings.json` into `<meetings_dir>/<date>-<name>/` and returns the wall
time of each stage.

Public names: `PipelineError`, `Result`, `STAGES`, `prepare_environment`, `run`.

Order matters for two libraries, so `run` keeps it fixed:

- Diarize runs before transcribe. `scribe.diarize.diarize` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
  and then imports torch; torch reads the variable once, at first import. parakeet_mlx does not
  import torch, so diarizing first means torch is always imported with the variable set.
- `prepare_environment` runs before anything imports huggingface_hub, which reads
  `HF_HUB_OFFLINE` once, at import.

Embed (`scribe.embeddings`) runs right after diarize, with the embedding model diarize
already loaded, and then drops the pyannote pipeline before Parakeet loads. It cuts the 3 s
windows from the exclusive spans; embeddings.json is written in the write stage, keyed by the
raw-to-label map that `merge_with_labels` returns, so its labels are the Transcript's.

Network: runs work offline once the models are downloaded. With both pinned
snapshots complete in the Hugging Face cache, `prepare_environment` sets `HF_HUB_OFFLINE=1`, so
hf_hub_download and pyannote resolve the files from the cache without asking the Hub. With a
snapshot missing, the run stays online and downloads it once. Telemetry (pyannote's and
huggingface_hub's) is turned off by the adapters themselves on every call, so it is off for the
pipeline and for any direct adapter call alike.

Re-runs: a Meeting folder that already holds transcript.json or transcript.md is refused before
any work, because a new Transcript would drop the Naming done on the old one. `force=True`
overwrites both files, embeddings.json and the audio; the Naming is then lost.
"""

import os
import time
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from scribe import diarize, embeddings, transcribe
from scribe.extract import extract_audio, meeting_folder_name
from scribe.merge import merge_with_labels
from scribe.naming import JSON_NAME, MARKDOWN_NAME
from scribe.transcript import Transcript, parse_json, render_json, render_markdown

STAGES = ("extract", "diarize", "embed", "transcribe", "merge", "write")

# Files each pinned snapshot needs, relative to its snapshot folder.
_SNAPSHOTS = {
    (diarize.MODEL_ID, diarize.MODEL_REVISION): (
        "config.yaml",
        "embedding/pytorch_model.bin",
        "segmentation/pytorch_model.bin",
        "plda/plda.npz",
        "plda/xvec_transform.npz",
    ),
    (transcribe.MODEL_ID, transcribe.MODEL_REVISION): ("config.json", "model.safetensors"),
}


class PipelineError(Exception):
    """The pipeline refused to run, for a reason worth showing the user as-is."""


@dataclass(frozen=True)
class Result:
    """What one run wrote, how many Speakers it found, and how long each stage took."""

    folder: Path
    transcript: Transcript
    # Distinct labels in the diarization. Can exceed len(transcript.speakers): merge drops a
    # Speaker with spans but no words.
    detected_speakers: int
    # Seconds per stage, in STAGES order.
    timings: Mapping[str, float]
    # Seconds from start to end of the run, measured on its own rather than summed.
    total: float


def hub_cache(environ: Mapping[str, str]) -> Path:
    """The Hugging Face hub cache folder, resolved as huggingface_hub's constants do."""
    if "HF_HUB_CACHE" in environ:
        return Path(os.path.expandvars(os.path.expanduser(environ["HF_HUB_CACHE"])))
    if "HUGGINGFACE_HUB_CACHE" in environ:
        return Path(os.path.expandvars(os.path.expanduser(environ["HUGGINGFACE_HUB_CACHE"])))
    default_home = os.path.join(os.path.expanduser("~"), ".cache")
    xdg_cache = environ.get("XDG_CACHE_HOME", default_home)
    home = environ.get("HF_HOME", os.path.join(xdg_cache, "huggingface"))
    return Path(os.path.expandvars(os.path.expanduser(home))) / "hub"


def models_cached(environ: Mapping[str, str]) -> bool:
    """True when every file of both pinned snapshots is in the cache.

    Snapshot entries are symlinks into the cache's blobs; `is_file` follows them, so a
    dangling link counts as missing.
    """
    cache = hub_cache(environ)
    for (model_id, revision), files in _SNAPSHOTS.items():
        snapshot = cache / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        if not all((snapshot / name).is_file() for name in files):
            return False
    return True


def prepare_environment(environ: MutableMapping[str, str] = os.environ) -> None:
    """Go offline when both models are cached.

    Must run before huggingface_hub is imported. An explicit `HF_HUB_OFFLINE` in the
    environment is kept as is.
    """
    if "HF_HUB_OFFLINE" not in environ and models_cached(environ):
        environ["HF_HUB_OFFLINE"] = "1"


def run(
    recording: Path,
    meetings_dir: Path,
    name: str | None = None,
    speakers: int | None = None,
    *,
    force: bool = False,
) -> Result:
    """Run all stages on one Recording; `speakers` reaches the diarizer unchanged.

    Raises `PipelineError` for an existing Transcript without `force`, and the stages' own
    `ExtractError`, `DiarizeError` and `TranscribeError`.
    """
    started = time.perf_counter()
    if recording.is_file():
        folder = meetings_dir / meeting_folder_name(recording, name)
        existing = [f for f in (JSON_NAME, MARKDOWN_NAME) if (folder / f).exists()]
        if existing and not force:
            raise PipelineError(
                f"{folder} already has {' and '.join(existing)}; "
                "pass --force to overwrite it (this drops any Naming), or use another --name"
            )
    prepare_environment()

    timings: dict[str, float] = {}

    def lap(stage: str, since: float) -> float:
        now = time.perf_counter()
        timings[stage] = now - since
        return now

    mark = time.perf_counter()
    audio = extract_audio(recording, meetings_dir, name)
    mark = lap("extract", mark)
    # Diarize first: it imports torch with PYTORCH_ENABLE_MPS_FALLBACK set (module docstring).
    diarization = diarize.diarize(audio, speakers=speakers)
    spans, centroids = diarization.spans, diarization.centroids
    mark = lap("diarize", mark)
    windows = embeddings.windows(spans, diarization.duration)
    vectors = diarization.embed([w.start for w in windows], embeddings.WINDOW_S)
    # Free the pyannote pipeline before Parakeet loads (module docstring).
    del diarization
    mark = lap("embed", mark)
    words = transcribe.transcribe(audio)
    mark = lap("transcribe", mark)
    segments, labels = merge_with_labels(spans, words)
    diarize_model = diarize.model_version()
    merged = Transcript(
        segments=segments,
        speakers=tuple(dict.fromkeys(s.speaker for s in segments)),
        models={"diarize": diarize_model, "transcribe": transcribe.model_version()},
    )
    mark = lap("merge", mark)
    # Render the Markdown from the JSON text, as `scribe name` does, so both files agree on
    # millisecond-rounded times.
    json_text = render_json(merged)
    transcript = parse_json(json_text)
    (audio.parent / JSON_NAME).write_text(json_text, encoding="utf-8", newline="\n")
    (audio.parent / MARKDOWN_NAME).write_text(
        render_markdown(transcript), encoding="utf-8", newline="\n"
    )
    embeddings.write(
        audio.parent, embeddings.build(labels, centroids, windows, vectors, diarize_model)
    )
    lap("write", mark)
    return Result(
        folder=audio.parent,
        transcript=transcript,
        detected_speakers=len({s.speaker for s in spans}),
        timings=timings,
        total=time.perf_counter() - started,
    )
