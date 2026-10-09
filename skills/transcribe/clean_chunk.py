"""File plumbing for /transcribe's cleanup step: raw chunks in, clean chunks appended, cut, removed.

Usage: uv run --extra asr python skills/transcribe/clean_chunk.py <command> <meeting> ...

- `raw <meeting> FROM COUNT`: say where raw Segments FROM to FROM+COUNT-1 (counted from 1) are
  in transcript.md, as one line: `Segments a-b of N: transcript.md offset X limit L`, for the
  Read tool (lines counted from 1). Past the end: `no Segments from FROM (N in all)`. It only
  names lines, because a shell output filter may cut a long chunk. It fails when transcript.md
  is not exactly what transcript.json renders to.
- `append <meeting>`: append transcript.clean.chunk.md to transcript.clean.md, then remove the
  chunk file.
- `keep <meeting> K`: cut transcript.clean.md after its first K Segments, so a retry rewrites
  from Segment K+1. K = 0 removes the file.
- `delete <meeting>`: remove transcript.clean.md and transcript.clean.chunk.md if present.

Exit 0 on success, 1 with one line on stderr otherwise. Messages name files and counts only,
never Segment text. Output goes to this Claude Code session only; never paste it into a PR,
commit or report (ADR 0001).
"""

import argparse
import sys
from pathlib import Path

from scribe.clean import CLEAN_NAME, split_clean
from scribe.naming import JSON_NAME, MARKDOWN_NAME
from scribe.transcript import Segment, TranscriptError, read_json, render_markdown

CHUNK_NAME = "transcript.clean.chunk.md"


class ChunkError(Exception):
    """A cleanup file step that failed, for a reason worth showing the user as-is."""


def raw(meeting: Path, start: int, count: int) -> None:
    try:
        transcript = read_json(meeting / JSON_NAME)
        markdown = (meeting / MARKDOWN_NAME).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, TranscriptError) as error:
        raise ChunkError(
            f"cannot read {JSON_NAME} or {MARKDOWN_NAME}: {type(error).__name__}"
        ) from error
    if markdown != render_markdown(transcript):
        raise ChunkError(f"{MARKDOWN_NAME} does not match {JSON_NAME}")
    if start < 1 or count < 1:
        raise ChunkError("FROM and COUNT must be 1 or more")
    segments = transcript.segments
    end = min(start - 1 + count, len(segments))
    if start > end:
        print(f"no Segments from {start} ({len(segments)} in all)")
        return
    # render_markdown: header, text lines, then one blank line between Segments.
    first = 1 + sum(_lines(s) + 1 for s in segments[: start - 1])
    length = sum(_lines(s) + 1 for s in segments[start - 1 : end]) - 1
    print(
        f"Segments {start}-{end} of {len(segments)}: {MARKDOWN_NAME} offset {first} limit {length}"
    )


def _lines(segment: Segment) -> int:
    return 1 + len(segment.text.split("\n"))


def append(meeting: Path) -> None:
    chunk_path, clean_path = meeting / CHUNK_NAME, meeting / CLEAN_NAME
    try:
        chunk = chunk_path.read_text(encoding="utf-8")
        existing = clean_path.read_text(encoding="utf-8") if clean_path.exists() else ""
    except (OSError, UnicodeDecodeError) as error:
        raise ChunkError(
            f"cannot read {CHUNK_NAME} or {CLEAN_NAME}: {type(error).__name__}"
        ) from error
    separator = "" if not existing else "\n" if existing.endswith("\n") else "\n\n"
    with clean_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(separator + chunk)
    chunk_path.unlink()


def keep(meeting: Path, kept: int) -> None:
    clean_path = meeting / CLEAN_NAME
    if kept < 0:
        raise ChunkError("K must be 0 or more")
    if kept == 0:
        clean_path.unlink(missing_ok=True)
        return
    try:
        clean = split_clean(clean_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ChunkError(f"cannot read {CLEAN_NAME}: {type(error).__name__}") from error
    if len(clean.segments) <= kept:
        return
    cut = clean.segments[kept].line
    clean_path.write_text("\n".join(clean.lines[:cut]), encoding="utf-8", newline="\n")


def delete(meeting: Path) -> None:
    for name in (CLEAN_NAME, CHUNK_NAME):
        (meeting / name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    raw_parser = sub.add_parser("raw")
    raw_parser.add_argument("meeting", type=Path)
    raw_parser.add_argument("start", type=int)
    raw_parser.add_argument("count", type=int)
    sub.add_parser("append").add_argument("meeting", type=Path)
    keep_parser = sub.add_parser("keep")
    keep_parser.add_argument("meeting", type=Path)
    keep_parser.add_argument("kept", type=int)
    sub.add_parser("delete").add_argument("meeting", type=Path)
    args = parser.parse_args()
    if not args.meeting.is_dir():
        print("clean_chunk: error: no such Meeting folder", file=sys.stderr)
        return 1
    try:
        if args.command == "raw":
            raw(args.meeting, args.start, args.count)
        elif args.command == "append":
            append(args.meeting)
        elif args.command == "keep":
            keep(args.meeting, args.kept)
        else:
            delete(args.meeting)
    except ChunkError as error:
        print(f"clean_chunk: error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"clean_chunk: error: {type(error).__name__} on a cleanup file", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
