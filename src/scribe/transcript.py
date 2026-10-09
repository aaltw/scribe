"""Pipeline step 5: the Transcript model and its Markdown and JSON writers (§2.5).

Pure Python, no model dependencies. Merge builds `Segment`s, `scribe name` rewrites Participant
names with `Transcript.name_speakers` and writes both files again, and the pipeline fills in
`Transcript.models`.

Public names: `Segment`, `Transcript`, `TranscriptError`, `format_timestamp`, `render_header`,
`render_markdown`, `render_json`, `parse_json`, `write_markdown`, `write_json`, `read_json`.

Timestamp format (transcript.md): `mm:ss` below one hour, `h:mm:ss` from one hour on, so a
Segment starting at 3725 s reads `[1:02:05]`. Seconds are floored, never rounded up, so a
timestamp never points past the moment the Segment starts. Minutes keep two digits below one
hour (`[05:07]`); the hour has no leading zero because a Meeting is never ten hours long.

transcript.json (version 1), written with 2-space indent, UTF-8, no ASCII escaping, a trailing
newline and this fixed key order:

    {
      "version": 1,
      "models": {"<role>": "<version>", ...},        # keys sorted
      "speakers": ["Speaker 1", "Speaker 2"],        # Speaker labels in first-heard order
      "segments": [
        {"start": 0.0, "end": 4.2, "speaker": "Speaker 1", "participant": null, "text": "..."}
      ]
    }

`start` and `end` are seconds rounded to milliseconds, so a read-then-write round trip is
byte-identical. `participant` is the Participant's name once the Speaker is named, else null.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

SCHEMA_VERSION = 1


class TranscriptError(Exception):
    """A transcript.json that cannot be read, for a reason worth showing the user as-is."""


@dataclass(frozen=True)
class Segment:
    """A stretch of speech by one Speaker, with start and end seconds and its text."""

    start: float
    end: float
    speaker: str
    text: str
    participant: str | None = None

    @property
    def label(self) -> str:
        """Who to show: the Participant once named, else the anonymous Speaker label."""
        return self.participant if self.participant is not None else self.speaker


@dataclass(frozen=True)
class Transcript:
    """The ordered Segments of one Meeting, plus its Speaker labels and model versions."""

    segments: tuple[Segment, ...] = ()
    speakers: tuple[str, ...] = ()
    models: Mapping[str, str] = field(default_factory=dict[str, str])

    def name_speakers(self, names: Mapping[str, str]) -> "Transcript":
        """Return a copy with Participant names set for the given Speaker labels.

        Speakers missing from `names` keep their current Participant, or stay unnamed.
        """
        segments = tuple(
            replace(s, participant=names[s.speaker]) if s.speaker in names else s
            for s in self.segments
        )
        return replace(self, segments=segments)


def format_timestamp(seconds: float) -> str:
    """`mm:ss` below one hour, `h:mm:ss` from one hour on; seconds are floored."""
    total = max(0, math.floor(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_header(segment: Segment) -> str:
    """A Segment's header line in transcript.md, `**<Speaker or Participant>** [mm:ss]`.

    transcript.clean.md uses the same line, so `scribe check-clean` compares against this string.
    """
    return f"**{segment.label}** [{format_timestamp(segment.start)}]"


def render_markdown(transcript: Transcript) -> str:
    """Segments in order, each header line then its text, a blank line between Segments."""
    blocks = [f"{render_header(s)}\n{s.text}\n" for s in transcript.segments]
    return "\n".join(blocks)


def render_json(transcript: Transcript) -> str:
    """Deterministic transcript.json text for a Transcript."""
    document = {
        "version": SCHEMA_VERSION,
        "models": {key: transcript.models[key] for key in sorted(transcript.models)},
        "speakers": list(transcript.speakers),
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "speaker": s.speaker,
                "participant": s.participant,
                "text": s.text,
            }
            for s in transcript.segments
        ],
    }
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def parse_json(text: str) -> Transcript:
    """Load transcript.json text back into the model."""
    try:
        document = json.loads(text)
        if document["version"] != SCHEMA_VERSION:
            raise TranscriptError(f"unsupported transcript.json version {document['version']!r}")
        segments = tuple(
            Segment(
                start=_number(raw["start"]),
                end=_number(raw["end"]),
                speaker=_string(raw["speaker"]),
                text=_string(raw["text"]),
                participant=None if raw["participant"] is None else _string(raw["participant"]),
            )
            for raw in document["segments"]
        )
        speakers = tuple(_string(label) for label in document["speakers"])
        models = {_string(k): _string(v) for k, v in document["models"].items()}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise TranscriptError(f"not a valid transcript.json: {error!r}") from error
    return Transcript(segments=segments, speakers=speakers, models=models)


def write_markdown(transcript: Transcript, path: Path) -> None:
    path.write_text(render_markdown(transcript), encoding="utf-8", newline="\n")


def write_json(transcript: Transcript, path: Path) -> None:
    path.write_text(render_json(transcript), encoding="utf-8", newline="\n")


def read_json(path: Path) -> Transcript:
    return parse_json(path.read_text(encoding="utf-8"))


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    return value
