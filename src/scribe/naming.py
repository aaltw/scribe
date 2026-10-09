"""Pipeline step 6: Naming, rewriting a Meeting's Transcript with Participant names."""

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from scribe.clean import CLEAN_NAME, rename_clean
from scribe.transcript import (
    Transcript,
    TranscriptError,
    parse_json,
    render_json,
    render_markdown,
)

JSON_NAME = "transcript.json"
MARKDOWN_NAME = "transcript.md"


class NamingError(Exception):
    """Naming failed for a reason worth showing Alex as-is."""


@dataclass(frozen=True)
class Named:
    """The renamed Transcript, and whether transcript.clean.md was there and was rewritten."""

    transcript: Transcript
    clean_found: bool = False
    clean_renamed: bool = False

    @property
    def clean_skipped(self) -> bool:
        """A clean file exists but its structure differs from transcript.json, so it was kept."""
        return self.clean_found and not self.clean_renamed


def resolve_meeting(meeting: str, meetings_dir: Path) -> Path:
    """A path to the Meeting folder, else a Meeting folder name under `meetings_dir`."""
    as_path = Path(meeting)
    if as_path.is_dir():
        return as_path
    under_dir = meetings_dir / meeting
    if under_dir.is_dir():
        return under_dir
    raise NamingError(f"no Meeting folder {meeting!r} (also looked in {meetings_dir})")


def parse_pairs(pairs: Iterable[str]) -> dict[str, str]:
    """Turn `"Speaker 1=Alex"` arguments into {label: name}."""
    names: dict[str, str] = {}
    for pair in pairs:
        label, sep, name = pair.partition("=")
        label, name = label.strip(), name.strip()
        if not sep or not label or not name:
            raise NamingError(f'malformed pair {pair!r}: expected "Speaker 1=Name"')
        if label in names:
            raise NamingError(f"Speaker {label!r} is named twice")
        names[label] = name
    return names


def name_meeting(folder: Path, names: dict[str, str]) -> Named:
    """Apply `names` to the Meeting's transcript.json and rewrite transcript.json and .md.

    When transcript.clean.md exists and matches transcript.json as `scribe check-clean` would
    see it, its header lines are re-rendered from the renamed Transcript too and its text is
    kept byte for byte. A clean file that does not match is left as it is (`Named.clean_skipped`),
    so a stale cleanup never blocks Naming.

    All files are written to temp files first, so a failure before the renames leaves the
    Meeting untouched. transcript.md renders from the JSON text as written, so both agree on
    millisecond-rounded times.
    """
    json_path = folder / JSON_NAME
    try:
        original = parse_json(json_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise NamingError(f"cannot read {json_path}: {error.strerror or error}") from error
    except TranscriptError as error:
        raise NamingError(f"{json_path}: {error}") from error

    known = set(original.speakers) | {s.speaker for s in original.segments}
    unknown = [label for label in names if label not in known]
    if unknown:
        raise NamingError(
            f"unknown Speaker {', '.join(repr(u) for u in unknown)}; "
            f"this Meeting has {', '.join(sorted(known)) or 'no Speakers'}"
        )

    json_text = render_json(original.name_speakers(names))
    renamed = parse_json(json_text)
    contents = {json_path: json_text, folder / MARKDOWN_NAME: render_markdown(renamed)}
    clean_path = folder / CLEAN_NAME
    clean_found = clean_path.exists()
    clean_text = _read_clean(clean_path) if clean_found else None
    clean_renamed = None if clean_text is None else rename_clean(original, renamed, clean_text)
    if clean_renamed is not None:
        contents[clean_path] = clean_renamed
    _replace_all(contents)
    return Named(renamed, clean_found=clean_found, clean_renamed=clean_renamed is not None)


def _read_clean(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _replace_all(contents: dict[Path, str]) -> None:
    temps: dict[Path, Path] = {}
    try:
        for target, text in contents.items():
            fd, tmp_name = tempfile.mkstemp(
                dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
            )
            temps[target] = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        for target, tmp in temps.items():
            os.replace(tmp, target)
    except OSError as error:
        raise NamingError(f"cannot write Transcript: {error.strerror or error}") from error
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)
