"""Pipeline step 3: timed words of a 16 kHz mono WAV with Parakeet-TDT-0.6B-v3.

Needs the `asr` extra. parakeet-mlx is imported inside `transcribe`, so this module, the core
package and the CLI import without it. Independent of diarization: the whole file is
transcribed once and merge (step 4) assigns the words to Speakers.

Public names: `MODEL_ID`, `MODEL_REVISION`, `CHUNK_S`, `OVERLAP_S`, `TranscribeError`,
`Token`, `transcribe`, `tokens_to_words`, `model_version`.

Telemetry: `transcribe` sets `HF_HUB_DISABLE_TELEMETRY=1` on every call, before it imports
huggingface_hub, so a direct call sends no usage data either (ADR 0001).

Long audio goes through parakeet-mlx's own chunking: windows of `CHUNK_S` seconds overlapping
by `OVERLAP_S`, with the overlap's tokens merged by parakeet-mlx. The defaults are its CLI's.
parakeet-mlx decodes the file with ffmpeg to float32; bfloat16 audio would break `get_logmel`.

Words from sub-word tokens (`tokens_to_words`): Parakeet emits SentencePiece pieces, where a
piece starting with a space begins a new word (`" H"`, `"ello"`). Pieces are grouped into words
on that leading space, then any trailing pieces made only of closing punctuation
(`.,!?;:%)]}…`) are split off as their own Words, so `start?` becomes `start` and `?`. Merge
joins Words with single spaces but no space before closing punctuation, which rebuilds the
text. A word's start is its first piece's start and its end its last piece's end; zero-length
pieces are kept.
"""

import os
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from scribe.merge import Word

MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
# The Hugging Face snapshot scribe is pinned to and was measured with.
MODEL_REVISION = "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15"
CHUNK_S = 120.0
OVERLAP_S = 15.0

_CLOSING_PUNCTUATION = frozenset(".,!?;:%)]}…")
_MISSING_EXTRA = "transcribe needs the asr extra (macOS only): rerun with `uv run --extra asr scribe ...`"


class TranscribeError(Exception):
    """Transcription failed for a reason worth showing the user as-is."""


class Token(Protocol):
    """One timed sub-word piece, as parakeet-mlx's `AlignedToken` has it."""

    @property
    def text(self) -> str: ...
    @property
    def start(self) -> float: ...
    @property
    def end(self) -> float: ...


def model_version() -> str:
    """`<model id>@<revision> (parakeet-mlx <version>)`, for transcript.json's models."""
    try:
        library = version("parakeet-mlx")
    except PackageNotFoundError as error:
        raise TranscribeError(_MISSING_EXTRA) from error
    return f"{MODEL_ID}@{MODEL_REVISION} (parakeet-mlx {library})"


def transcribe(
    audio: Path, *, chunk_s: float = CHUNK_S, overlap_s: float = OVERLAP_S
) -> list[Word]:
    """Timed Words of a whole audio file, in start order, in seconds from its start."""
    if not audio.is_file():
        raise TranscribeError(f"audio not found: {audio}")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    try:
        from huggingface_hub import hf_hub_download
        from parakeet_mlx import from_pretrained
    except ImportError as error:
        raise TranscribeError(_MISSING_EXTRA) from error

    # parakeet-mlx's from_pretrained takes no revision, so fetch the pinned snapshot's files
    # (from the cache when present) and load from that directory.
    config = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    hf_hub_download(MODEL_ID, "model.safetensors", revision=MODEL_REVISION)
    model = from_pretrained(str(Path(config).parent))
    try:
        result = model.transcribe(audio, chunk_duration=chunk_s, overlap_duration=overlap_s)
    except RuntimeError as error:
        raise TranscribeError(f"cannot transcribe {audio.name}: {error}") from error
    words = tokens_to_words(result.tokens)
    return sorted(words, key=lambda w: (w.start, w.end))


def tokens_to_words(tokens: Iterable[Token]) -> list[Word]:
    """Join sub-word pieces into Words and split trailing closing punctuation off them."""
    groups: list[list[Token]] = []
    for token in tokens:
        if not groups or token.text[:1].isspace():
            groups.append([])
        groups[-1].append(token)

    words: list[Word] = []
    for group in groups:
        cut = len(group)
        while cut > 0 and _is_closing_punctuation(group[cut - 1].text):
            cut -= 1
        body, tail = group[:cut], group[cut:]
        # A bare " " piece can open a word (" ", "4", "2"); the word starts at its first sound.
        while body and not body[0].text.strip():
            body = body[1:]
        text = "".join(t.text for t in body).strip()
        if text:
            words.append(Word(start=body[0].start, end=body[-1].end, text=text))
        words.extend(Word(start=t.start, end=t.end, text=t.text.strip()) for t in tail)
    return words


def _is_closing_punctuation(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and all(c in _CLOSING_PUNCTUATION for c in stripped)
