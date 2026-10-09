"""Print a few representative Segments per Speaker of one Meeting, for /transcribe's Naming.

Usage: uv run python skills/transcribe/sample_lines.py <meeting-folder> [--per-speaker N]

For each Speaker, in first-heard order: its label, Segment count and speaking time, then its N
longest Segments (default 3) in time order, each cut to 160 characters. Output goes to this
Claude Code session only; never paste it into a PR, commit or report (ADR 0001).
"""

import argparse
import sys
from pathlib import Path

from scribe.stats import talk_time
from scribe.transcript import TranscriptError, format_timestamp, read_json

MAX_CHARS = 160


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("meeting", type=Path, help="Meeting folder holding transcript.json")
    parser.add_argument("--per-speaker", type=int, default=3, metavar="N")
    args = parser.parse_args()
    try:
        transcript = read_json(args.meeting / "transcript.json")
    except (OSError, TranscriptError) as error:
        print(f"sample_lines: error: {error}", file=sys.stderr)
        return 1
    for speaker in transcript.speakers:
        segments = [s for s in transcript.segments if s.speaker == speaker]
        seconds = talk_time(segments)
        named = f" (now {segments[0].participant})" if segments and segments[0].participant else ""
        print(f"{speaker}{named}: {len(segments)} Segments, {seconds:.0f} s")
        longest = sorted(segments, key=lambda s: len(s.text), reverse=True)[: args.per_speaker]
        for segment in sorted(longest, key=lambda s: s.start):
            text = (
                segment.text if len(segment.text) <= MAX_CHARS else segment.text[:MAX_CHARS] + "…"
            )
            print(f"  [{format_timestamp(segment.start)}] {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
