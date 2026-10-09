"""Speaker stats: talk time, share and Segment count per Speaker, computed in code.

Pure Python, no model dependencies. `scribe stats <meeting>` reads transcript.json, writes
stats.json and prints one line per Speaker. `scribe slack` reads stats.json.

Public names: `SpeakerStats`, `StatsError`, `talk_time`, `compute_stats`, `render_json`,
`format_line`, `stats_meeting`.

stats.json (version 1), 2-space indent, UTF-8, no ASCII escaping, a trailing newline:

    {
      "version": 1,
      "speakers": [
        {"speaker": "Speaker 1", "participant": "Alex", "seconds": 12.5, "share": 0.625,
         "segments": 3}
      ]
    }

`speakers` is in first-heard order (the Transcript's own order). `participant` is null while the
Speaker is unnamed. `seconds` is the sum of end minus start, rounded to 3 decimals. `share` is
the Speaker's fraction of all Speakers' talk time, rounded to 3 decimals; 0 for everyone when the
total is zero.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scribe.transcript import Segment, TranscriptError, format_timestamp, read_json

SCHEMA_VERSION = 1


class StatsError(Exception):
    """A Meeting folder that stats cannot be computed for, worded for Alex."""


@dataclass(frozen=True)
class SpeakerStats:
    speaker: str
    participant: str | None
    seconds: float
    share: float
    segments: int

    @property
    def label(self) -> str:
        """Who to show: the Participant once named, else the anonymous Speaker label."""
        return self.participant if self.participant is not None else self.speaker


def talk_time(segments: Iterable[Segment]) -> float:
    """Sum of end minus start over the Segments, in seconds, unrounded."""
    return sum(s.end - s.start for s in segments)


def compute_stats(
    transcript_segments: Iterable[Segment], speakers: Iterable[str]
) -> list[SpeakerStats]:
    """Stats per Speaker label in the given (first-heard) order."""
    segments = tuple(transcript_segments)
    labels = tuple(speakers)
    by_speaker = {label: [s for s in segments if s.speaker == label] for label in labels}
    times = {label: talk_time(group) for label, group in by_speaker.items()}
    total = sum(times.values())
    result: list[SpeakerStats] = []
    for label in labels:
        group = by_speaker[label]
        participant = next((s.participant for s in group if s.participant is not None), None)
        share = times[label] / total if total > 0 else 0.0
        result.append(
            SpeakerStats(label, participant, round(times[label], 3), round(share, 3), len(group))
        )
    return result


def render_json(stats: Iterable[SpeakerStats]) -> str:
    document = {
        "version": SCHEMA_VERSION,
        "speakers": [
            {
                "speaker": s.speaker,
                "participant": s.participant,
                "seconds": s.seconds,
                "share": s.share,
                "segments": s.segments,
            }
            for s in stats
        ],
    }
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def format_line(stats: SpeakerStats) -> str:
    """`<label>: <mm:ss>, <percent>%, <n> Segments`; talk time floors like timestamps do."""
    return (
        f"{stats.label}: {format_timestamp(stats.seconds)}, "
        f"{stats.share * 100:.1f}%, {stats.segments} Segments"
    )


def stats_meeting(folder: Path) -> list[SpeakerStats]:
    """Write `<folder>/stats.json` from its transcript.json and return the stats."""
    source = folder / "transcript.json"
    if not source.is_file():
        raise StatsError(f"no transcript.json in {folder}")
    try:
        transcript = read_json(source)
    except (OSError, TranscriptError) as error:
        raise StatsError(f"cannot read {source}: {error}") from error
    stats = compute_stats(transcript.segments, transcript.speakers)
    (folder / "stats.json").write_text(render_json(stats), encoding="utf-8", newline="\n")
    return stats
