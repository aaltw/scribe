"""The Clean Transcript and its structure check, `scribe check-clean`.

transcript.clean.md is Claude's copy of a Meeting's transcript.md with misheard and phonetic
spellings fixed. It keeps the raw layout: per Segment, in order, the header line exactly as
`render_header` writes it from transcript.json (`**<Speaker or Participant>** [mm:ss]`), then the
Segment's text. Text may run over several lines. Blank lines and trailing whitespace are not
part of the format: any count of either is accepted.

A line is taken as a header when, without trailing whitespace, it has the header's shape:
starts with `**`, ends with `** [mm:ss]` or `** [h:mm:ss]`. Its name is never parsed. The whole
line is compared with the string rendered from the matching raw Segment, so names with spaces,
apostrophes, hyphens or brackets cannot be misread.

`check_clean` reports the first difference as one line, `Segment k of N raw / M clean: <field>`,
with k counted from 1 and `<field>` one of:

- `header`: the Speaker or Participant differs, the timestamp matches.
- `timestamp`: the timestamp differs, the Speaker or Participant matches.
- `header and timestamp`: both differ (a dropped, merged or reordered Segment often shows so).
- `empty text`: the header matches but no text follows it.
- `missing`: every clean Segment matches, but the raw Transcript has more (M < N).
- `extra`: every raw Segment matches, but the clean file has more (M > N).
- `text before first header`: the clean file has text before its first header (k is 1).

The line never contains Segment text or names, so it may be pasted into PRs and reports.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from scribe.transcript import Segment, Transcript, render_header

CLEAN_NAME = "transcript.clean.md"

RATIO_LOW = 0.5
RATIO_HIGH = 2.0

_HEADER_SHAPE = re.compile(r"\*\*.+\*\* \[(?:\d+:)?\d{2}:\d{2}\]")


@dataclass(frozen=True)
class CleanSegment:
    """One Segment of a Clean Transcript: its header line's index, the header, and its text."""

    line: int
    header: str
    text: str


@dataclass(frozen=True)
class CleanFile:
    """A Clean Transcript split into Segments, plus its lines as read (each without `\\n`)."""

    lines: tuple[str, ...]
    segments: tuple[CleanSegment, ...]
    preamble: bool


@dataclass(frozen=True)
class CleanCheck:
    """The result of comparing a Clean Transcript with its raw Transcript."""

    raw: int
    clean: int
    mismatch: tuple[int, str] | None
    ratio: float | None

    @property
    def ok(self) -> bool:
        return self.mismatch is None

    def mismatch_line(self) -> str:
        """`Segment k of N raw / M clean: <field>`; only valid when not `ok`."""
        assert self.mismatch is not None
        k, field = self.mismatch
        return f"Segment {k} of {self.raw} raw / {self.clean} clean: {field}"

    def ratio_outside(self) -> bool:
        return self.ratio is not None and not RATIO_LOW <= self.ratio <= RATIO_HIGH


def split_clean(text: str) -> CleanFile:
    """Split Clean Transcript text into Segments at each header-shaped line."""
    lines = tuple(text.split("\n"))
    segments: list[CleanSegment] = []
    preamble = False
    header_at: int | None = None
    body: list[str] = []

    def close() -> None:
        if header_at is not None:
            kept = [line.rstrip() for line in body if line.strip()]
            segments.append(CleanSegment(header_at, lines[header_at].rstrip(), "\n".join(kept)))

    for index, line in enumerate(lines):
        if _HEADER_SHAPE.fullmatch(line.rstrip()):
            close()
            header_at, body = index, []
        elif header_at is None:
            preamble = preamble or bool(line.strip())
        else:
            body.append(line)
    close()
    return CleanFile(lines=lines, segments=tuple(segments), preamble=preamble)


def check_clean(transcript: Transcript, clean_text: str, *, partial: bool = False) -> CleanCheck:
    """Compare a Clean Transcript with its raw Transcript, Segment by Segment.

    With `partial`, a clean file holding a correct prefix of the raw Segments passes, so a
    cleanup written in chunks can be checked after each chunk. The text length ratio covers the
    Segments compared, so a prefix does not warn for being short.
    """
    raw = transcript.segments
    clean = split_clean(clean_text)
    n, m = len(raw), len(clean.segments)

    def result(mismatch: tuple[int, str] | None) -> CleanCheck:
        ratio = _ratio(raw[:m], clean.segments) if mismatch is None else None
        return CleanCheck(raw=n, clean=m, mismatch=mismatch, ratio=ratio)

    if clean.preamble:
        return result((1, "text before first header"))
    for k, (segment, cleaned) in enumerate(zip(raw, clean.segments, strict=False), start=1):
        field = _header_field(segment, cleaned.header)
        if field is not None:
            return result((k, field))
        if not cleaned.text:
            return result((k, "empty text"))
    if m > n:
        return result((n + 1, "extra"))
    if m < n and not partial:
        return result((m + 1, "missing"))
    return result(None)


def rename_clean(before: Transcript, after: Transcript, clean_text: str) -> str | None:
    """Clean Transcript text with headers re-rendered from `after`, else None.

    Only header lines change; every other byte, so each Segment's clean text, stays as written.
    Returns None when the clean file does not pass `check_clean` against `before`, the
    Transcript it was written from, since its Segments then cannot be matched by position.
    """
    if not check_clean(before, clean_text).ok:
        return None
    clean = split_clean(clean_text)
    lines = list(clean.lines)
    for cleaned, segment in zip(clean.segments, after.segments, strict=True):
        ending = "\r" if lines[cleaned.line].endswith("\r") else ""
        lines[cleaned.line] = render_header(segment) + ending
    return "\n".join(lines)


def _header_field(segment: Segment, header: str) -> str | None:
    expected = render_header(segment)
    if header == expected:
        return None
    timestamp = expected[expected.rindex(" [") :]
    if header.endswith(timestamp):
        return "header"
    if header.startswith(expected[: -len(timestamp)] + " ["):
        return "timestamp"
    return "header and timestamp"


def _ratio(raw: Sequence[Segment], clean: Sequence[CleanSegment]) -> float | None:
    raw_length = sum(len(s.text) for s in raw)
    if raw_length == 0:
        return None
    return sum(len(c.text) for c in clean) / raw_length
