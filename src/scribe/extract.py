"""Pipeline step 1: pull a Recording's audio out to a 16 kHz mono WAV."""

import re
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

DATED_BASENAME = re.compile(r"^(\d{4}-\d{2}-\d{2})(?![0-9])")


class ExtractError(Exception):
    """Extraction failed for a reason worth showing the user as-is."""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def meeting_folder_name(recording: Path, name: str | None = None) -> str:
    """Folder name `<date>-<name>` for a Recording's Meeting.

    The date comes from the basename when it starts with YYYY-MM-DD (and is then removed from
    the default name), else from the file's mtime.
    """
    stem = recording.stem
    match = DATED_BASENAME.match(stem)
    day: str
    if match and _is_date(match.group(1)):
        day = match.group(1)
        stem = stem[match.end() :]
    else:
        day = (
            datetime.fromtimestamp(recording.stat().st_mtime, tz=UTC)
            .astimezone()
            .date()
            .isoformat()
        )
    slug = slugify(name) if name is not None else slugify(stem)
    return f"{day}-{slug}" if slug else day


def _is_date(text: str) -> bool:
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def has_audio_stream(recording: Path) -> bool:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(recording),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise ExtractError(f"cannot read {recording.name}: {probe.stderr.strip()}")
    return bool(probe.stdout.strip())


def extract_audio(recording: Path, meetings_dir: Path, name: str | None = None) -> Path:
    """Write `<meetings_dir>/<date>-<name>/audio.wav` and return its path."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise ExtractError("ffmpeg not found on PATH; install it (macOS: brew install ffmpeg)")
    if not recording.is_file():
        raise ExtractError(f"Recording not found: {recording}")
    if not has_audio_stream(recording):
        raise ExtractError(f"{recording.name} has no audio stream; nothing to extract")

    folder = meetings_dir / meeting_folder_name(recording, name)
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "audio.wav"
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(recording),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ExtractError(f"ffmpeg failed on {recording.name}: {result.stderr.strip()}")
    return output
