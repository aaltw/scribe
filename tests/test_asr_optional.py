"""The asr extra stays optional, and sub-word tokens become Words. Runs without the extra."""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from scribe.merge import SpeakerSpan, Word, merge
from scribe.transcribe import tokens_to_words

# A child interpreter where the asr extra's libraries cannot be imported, even on a Mac that has
# them installed: proves the core package, both adapter modules and `scribe --help` need none.
BLOCK_ASR = """
import sys

BLOCKED = ("huggingface_hub", "mlx", "numpy", "parakeet_mlx", "pyannote", "torch", "torchaudio")


class BlockAsr:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, BlockAsr())

import scribe
import scribe.diarize
import scribe.transcribe
from scribe.cli import main

try:
    main(["--help"])
except SystemExit as exit:
    code = exit.code
loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
print(f"exit={code} loaded={loaded}", file=sys.stderr)
"""


def test_core_and_help_work_without_asr_extra() -> None:
    result = subprocess.run(
        [sys.executable, "-c", BLOCK_ASR], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "usage: scribe" in result.stdout
    assert "exit=0 loaded=[]" in result.stderr


def test_adapters_report_missing_extra(tmp_path: Path) -> None:
    script = BLOCK_ASR + (
        "\nfrom pathlib import Path"
        "\nfrom scribe.diarize import DiarizeError, diarize"
        "\nfrom scribe.transcribe import TranscribeError, transcribe"
        f"\naudio = Path({str(tmp_path / 'audio.wav')!r})"
        "\naudio.write_bytes(b'')"
        "\nfor run, error in ((diarize, DiarizeError), (transcribe, TranscribeError)):"
        "\n    try:"
        "\n        run(audio)"
        "\n    except error as e:"
        "\n        print(e)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("uv run --extra asr scribe") == 2


# Records the telemetry switches at the moment an adapter first imports an asr library, then
# blocks that import, so the adapter stops there with its missing-extra error.
TELEMETRY_AT_IMPORT = """
import os
import sys
from pathlib import Path

ASR = ("huggingface_hub", "parakeet_mlx", "pyannote", "torch")
KEYS = ("PYANNOTE_METRICS_ENABLED", "HF_HUB_DISABLE_TELEMETRY")
seen = []


class Record:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ASR:
            seen.append((name, {k: os.environ.get(k) for k in KEYS}))
            raise ImportError(f"blocked: {name}")
        return None


sys.meta_path.insert(0, Record())

from scribe.diarize import DiarizeError, diarize
from scribe.transcribe import TranscribeError, transcribe

audio = Path(sys.argv[1])
for run, error in ((diarize, DiarizeError), (transcribe, TranscribeError)):
    os.environ.update(PYANNOTE_METRICS_ENABLED="true", HF_HUB_DISABLE_TELEMETRY="0")
    seen.clear()
    try:
        run(audio)
    except error:
        pass
    name, switches = seen[0]
    print(run.__name__, name, *(f"{k}={v}" for k, v in switches.items()))
"""


def test_adapters_turn_telemetry_off_before_importing_asr_libraries(tmp_path: Path) -> None:
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    env = {**os.environ, "PYANNOTE_METRICS_ENABLED": "true", "HF_HUB_DISABLE_TELEMETRY": "0"}
    result = subprocess.run(
        [sys.executable, "-c", TELEMETRY_AT_IMPORT, str(audio)],
        capture_output=True, text=True, check=False, env=env,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "diarize torch PYANNOTE_METRICS_ENABLED=false HF_HUB_DISABLE_TELEMETRY=1",
        "transcribe huggingface_hub PYANNOTE_METRICS_ENABLED=true HF_HUB_DISABLE_TELEMETRY=1",
    ]


@dataclass(frozen=True)
class Piece:
    text: str
    start: float
    end: float


def pieces(*items: tuple[str, float, float]) -> list[Piece]:
    return [Piece(text, start, end) for text, start, end in items]


def test_pieces_with_leading_space_start_words() -> None:
    tokens = pieces((" H", 0.0, 0.24), ("ello", 0.24, 0.48), (" ever", 0.48, 0.64),
                    ("y", 0.64, 0.8), ("one", 0.8, 0.96))  # fmt: skip
    assert tokens_to_words(tokens) == [Word(0.0, 0.48, "Hello"), Word(0.48, 0.96, "everyone")]


def test_first_piece_starts_a_word_without_leading_space() -> None:
    assert tokens_to_words(pieces(("ja", 0.0, 0.2), (" prima", 0.3, 0.6))) == [
        Word(0.0, 0.2, "ja"),
        Word(0.3, 0.6, "prima"),
    ]


def test_closing_punctuation_becomes_its_own_word() -> None:
    tokens = pieces((" dem", 2.96, 3.12), ("o", 3.12, 3.36), ("?", 3.36, 3.52),
                    (" Y", 4.8, 5.12), ("es", 5.12, 5.36), (",", 5.36, 5.6),
                    (" begin", 5.92, 6.24), ("...", 6.24, 6.32))  # fmt: skip
    assert tokens_to_words(tokens) == [
        Word(2.96, 3.36, "demo"),
        Word(3.36, 3.52, "?"),
        Word(4.8, 5.36, "Yes"),
        Word(5.36, 5.6, ","),
        Word(5.92, 6.24, "begin"),
        Word(6.24, 6.32, "..."),
    ]


def test_apostrophe_and_inner_punctuation_stay_in_the_word() -> None:
    tokens = pieces((" It", 3.52, 3.68), ("'", 3.68, 3.76), ("s", 3.76, 3.92),
                    (" 3", 4.0, 4.1), (",", 4.1, 4.2), ("5", 4.2, 4.3),
                    (" e", 5.0, 5.1), (".", 5.1, 5.2), ("g", 5.2, 5.3), (".", 5.3, 5.4))  # fmt: skip
    assert tokens_to_words(tokens) == [
        Word(3.52, 3.92, "It's"),
        Word(4.0, 4.3, "3,5"),
        Word(5.0, 5.3, "e.g"),
        Word(5.3, 5.4, "."),
    ]


def test_zero_duration_pieces_are_kept() -> None:
    tokens = pieces((" let", 5.6, 5.76), ("'", 5.76, 5.92), ("s", 5.92, 5.92),
                    (" go", 5.92, 5.92), (".", 5.92, 5.92))  # fmt: skip
    assert tokens_to_words(tokens) == [
        Word(5.6, 5.92, "let's"),
        Word(5.92, 5.92, "go"),
        Word(5.92, 5.92, "."),
    ]


def test_bare_space_piece_opens_a_word_at_its_first_sound() -> None:
    tokens = pieces((" is", 6.88, 7.04), (" ", 7.04, 7.2), ("4", 7.2, 7.44), ("2", 7.44, 7.68),
                    ("%", 7.68, 7.92), (" ", 8.0, 8.1))  # fmt: skip
    assert tokens_to_words(tokens) == [
        Word(6.88, 7.04, "is"),
        Word(7.2, 7.68, "42"),
        Word(7.68, 7.92, "%"),
    ]


def test_punctuation_with_leading_space_is_its_own_word() -> None:
    tokens = pieces((" wait", 0.0, 0.3), (" .", 0.3, 0.4), (" (", 0.5, 0.6), ("ok", 0.6, 0.8),
                    (")", 0.8, 0.9))  # fmt: skip
    assert tokens_to_words(tokens) == [
        Word(0.0, 0.3, "wait"),
        Word(0.3, 0.4, "."),
        Word(0.5, 0.8, "(ok"),
        Word(0.8, 0.9, ")"),
    ]


def test_no_tokens_no_words() -> None:
    assert tokens_to_words([]) == []


@pytest.mark.parametrize(
    "texts",
    [
        [" H", "ello", " ever", "y", "one", ",", " sh", "all", " we", " st", "art", "?"],
        [" It", "'", "s", " re", "ady", ".", " The", " bud", "get", " is", " ", "4", "2", "%"],
        [" Y", "es", ",", " let", "'", "s", " beg", "in", "...", " ok", "!"],
        [" (", "ok", ")", " 3", ",", "5", " e", ".", "g", "."],
    ],
)
def test_merge_rebuilds_the_text_from_words(texts: list[str]) -> None:
    words = tokens_to_words(pieces(*((t, float(i), float(i) + 0.1) for i, t in enumerate(texts))))
    assert all(w.text and not any(c.isspace() for c in w.text) for w in words)
    segments = merge([SpeakerSpan(0.0, 100.0, "SPEAKER_00")], words)
    assert [s.text for s in segments] == [" ".join("".join(texts).split())]
