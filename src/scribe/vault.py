"""`scribe vault-note`: one Obsidian note from a Meeting's Summary and Minutes.

The note is `<vault>/<folder>/<YYYY-MM-DD>-<slug>.md`: this frontmatter, then summary.md, then
minutes.md, each as it stands. The date is the Meeting folder name's first ten characters and the
slug is the rest of the name, slugified, so a default name that carries a time keeps it.

    ---
    type: meeting
    status: completed
    created: <date>
    updated: <date>
    tags: [meeting, scribe]
    participants: [<named Participants, first-heard order>]
    meeting-type: ""
    linked-project: ""
    linked-weekly: ""
    ---
"""

import re
from dataclasses import dataclass
from pathlib import Path

from scribe.extract import slugify
from scribe.transcript import TranscriptError, read_json

SUMMARY_NAME = "summary.md"
MINUTES_NAME = "minutes.md"
TRANSCRIPT_NAME = "transcript.json"
DATE_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:-|$)")
PLAIN_NAME = re.compile(r"^[\w][\w .'’-]*$")


class VaultNoteError(Exception):
    """A note that cannot be built; `code` is the exit code (1 exists, 2 missing input)."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class VaultNote:
    path: Path
    frontmatter: str
    body: str


def note_name(folder: Path) -> tuple[str, str]:
    """(date, file name) for a Meeting folder named `<date>-<slug>`."""
    match = DATE_PREFIX.match(folder.name)
    if match is None:
        raise VaultNoteError(
            f"Meeting folder name {folder.name!r} does not start with YYYY-MM-DD; "
            "rename the folder to <YYYY-MM-DD>-<name>, then run again"
        )
    date = match.group(1)
    slug = slugify(folder.name[match.end() :])
    return date, f"{date}-{slug}.md" if slug else f"{date}.md"


def _yaml_name(name: str) -> str:
    if PLAIN_NAME.match(name) and name == name.strip():
        return name
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_frontmatter(date: str, participants: list[str]) -> str:
    people = ", ".join(_yaml_name(p) for p in participants)
    return "\n".join(
        [
            "---",
            "type: meeting",
            "status: completed",
            f"created: {date}",
            f"updated: {date}",
            "tags: [meeting, scribe]",
            f"participants: [{people}]",
            'meeting-type: ""',
            'linked-project: ""',
            'linked-weekly: ""',
            "---",
        ]
    )


def _read(folder: Path, name: str, hint: str) -> str:
    path = folder / name
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise VaultNoteError(f"no {name} in the Meeting folder; {hint}") from None
    except (OSError, UnicodeDecodeError) as error:
        raise VaultNoteError(f"cannot read {name}: {error}; {hint}") from error


def build_note(folder: Path, vault: Path, rel_folder: str) -> VaultNote:
    """Read the Meeting's files and build its note; writes nothing."""
    date, file_name = note_name(folder)
    summary = _read(folder, SUMMARY_NAME, "run /transcribe to write the Summary and Minutes")
    minutes = _read(folder, MINUTES_NAME, "run /transcribe to write the Summary and Minutes")
    _read(folder, TRANSCRIPT_NAME, "run scribe on the Recording first")
    try:
        transcript = read_json(folder / TRANSCRIPT_NAME)
    except TranscriptError as error:
        raise VaultNoteError(f"{error}; run scribe on the Recording again") from error
    participants = list(
        dict.fromkeys(s.participant for s in transcript.segments if s.participant is not None)
    )
    body = f"{summary.rstrip()}\n\n{minutes.rstrip()}\n"
    target_dir = vault / rel_folder
    return VaultNote(target_dir / file_name, render_frontmatter(date, participants), body)


def write_note(note: VaultNote) -> None:
    if note.path.exists():
        raise VaultNoteError(
            f"{note.path} already exists and was left unchanged; "
            "delete or rename it to write the note again",
            code=1,
        )
    note.path.parent.mkdir(parents=True, exist_ok=True)
    note.path.write_text(f"{note.frontmatter}\n\n{note.body}", encoding="utf-8", newline="\n")
