"""Slack message: `scribe slack <meeting>` writes slack.md for the user to paste.

Pure Python. Reads summary.md (headings fixed by the /transcribe templates) and stats.json
(version 1, written by `scribe stats`), writes `<folder>/slack.md` in Slack mrkdwn: `*bold*`
section titles, `•` bullets, no `#` headings. Nothing is ever posted.

Public names: `SlackError`, `MissingInputError`, `escape`, `render_slack`, `slack_meeting`.
"""

import json
import re
from pathlib import Path

from scribe.stats import SCHEMA_VERSION, SpeakerStats
from scribe.transcript import format_timestamp

SLACK_NAME = "slack.md"
SECTIONS = ("Overview", "Decisions", "Action items")
FOLDER_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")


class SlackError(Exception):
    """A Meeting folder slack.md cannot be written for, worded for the user."""


class MissingInputError(SlackError):
    """summary.md or stats.json is missing; the message names the command that makes it."""


def escape(text: str) -> str:
    """Slack mrkdwn escaping: only `&`, `<` and `>` (`&` first)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def parse_summary(text: str) -> dict[str, list[str]]:
    """Non-blank lines under each `## <section>` heading of summary.md, in file order."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = sections.setdefault(line[3:].strip(), [])
        elif current is not None and line.strip():
            current.append(line.rstrip())
    missing = [name for name in SECTIONS if name not in sections]
    if missing:
        raise SlackError(
            f"summary.md has no '## {missing[0]}' heading; "
            "run /transcribe to write it again from the template"
        )
    return sections


def render_list(lines: list[str]) -> list[str]:
    """`- item` becomes `• item`; a `- None` / `- None.` line becomes `• None`."""
    result: list[str] = []
    for line in lines:
        item = line[2:].strip() if line.startswith("- ") else line.strip()
        if item in ("None", "None."):
            item = "None"
        result.append(f"• {escape(item)}")
    return result


def load_stats(path: Path) -> list[SpeakerStats]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["version"] != SCHEMA_VERSION:
            raise SlackError(
                f"{path.name} has version {document['version']!r}, expected {SCHEMA_VERSION}; "
                "run scribe stats on the Meeting folder to write it again"
            )
        return [
            SpeakerStats(
                str(s["speaker"]),
                None if s["participant"] is None else str(s["participant"]),
                float(s["seconds"]),
                float(s["share"]),
                int(s["segments"]),
            )
            for s in document["speakers"]
        ]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SlackError(
            f"cannot read {path.name} as Speaker stats; "
            "run scribe stats on the Meeting folder to write it again"
        ) from error


def title_line(folder: Path) -> str:
    """`*<date> <name>*` from the Meeting folder's `<date>-<name>` name."""
    match = FOLDER_PATTERN.match(folder.resolve().name)
    title = f"{match.group(1)} {match.group(2)}" if match else folder.resolve().name
    return f"*{escape(title)}*"


def render_slack(title: str, summary: str, stats: list[SpeakerStats]) -> str:
    sections = parse_summary(summary)
    blocks = [title]
    overview = [escape(line.strip()) for line in sections["Overview"]]
    blocks.append("\n".join(["*Overview*", *overview]))
    for name in SECTIONS[1:]:
        blocks.append("\n".join([f"*{name}*", *render_list(sections[name])]))
    lines = [
        f"• {escape(s.label)}: {format_timestamp(s.seconds)}, {s.share * 100:.1f}%" for s in stats
    ]
    blocks.append("\n".join(["*Speaker stats*", *lines]))
    return "\n\n".join(blocks) + "\n"


def slack_meeting(folder: Path) -> Path:
    """Write `<folder>/slack.md` from its summary.md and stats.json and return its path."""
    summary_path = folder / "summary.md"
    stats_path = folder / "stats.json"
    if not summary_path.is_file():
        raise MissingInputError(
            f"no summary.md in {folder}; the /transcribe skill writes it (run /transcribe on "
            "the Recording, or ask Claude to write the Summary for this Meeting)"
        )
    if not stats_path.is_file():
        raise MissingInputError(f"no stats.json in {folder}; scribe stats {folder} writes it")
    try:
        summary = summary_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SlackError(f"cannot read {summary_path} as UTF-8 text") from error
    output = folder / SLACK_NAME
    output.write_text(
        render_slack(title_line(folder), summary, load_stats(stats_path)),
        encoding="utf-8",
        newline="\n",
    )
    return output
