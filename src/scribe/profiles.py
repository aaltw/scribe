"""Voice profiles: a named Participant's saved voice, one file per Participant.

`save_sample` copies one Speaker's `mean_of_windows` from a Meeting's `embeddings.json` into
`<profiles-dir>/<slug>.json`; `list_profiles` summarises the files. The suggestion step
reads these files, so the format is a contract.

File format (version 1), the slug being the case-insensitive Participant name:

    {"version": 1,
     "name": "<Participant spelling from transcript.json, as first saved>",
     "samples": [{"source": "<Meeting folder name>",
                  "vector": [256 floats],
                  "windows": <count, at least MIN_WINDOWS>,
                  "model": "<diarize.model_version() copied from embeddings.json>"}]}

The name is the spelling in the Meeting's transcript.json, not the casing typed at save (the
typed name only has to match it case-insensitively). A later save from another Meeting keeps the
stored name, even when that Meeting spells it differently; `profile list` shows the stored name.

At most one sample per source Meeting: saving again from the same Meeting replaces it. All
samples of a profile share one model; a different model is refused. The vector is the Speaker's
mean of windows, never the centroid (it separates Speakers better in measurement).

A Participant named on more than one Speaker of a Meeting (pyannote split one voice) is
refused: which Speaker is "the" voice is the user's call, and a guess would put a wrong or diluted
vector into a profile that later suggests Namings.

Profiles are biometric data (ADR 0001): nothing here prints a vector value.
"""

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scribe.embeddings import FILE_NAME as EMBEDDINGS_NAME
from scribe.naming import JSON_NAME
from scribe.transcript import TranscriptError, parse_json

VERSION = 1
MIN_WINDOWS = 4
DEFAULT_DIR = Path(".scribe/profiles")


class ProfileError(Exception):
    """A refusal; the message ends with a next step the user can act on."""


@dataclass(frozen=True)
class ProfileSummary:
    name: str
    samples: int
    sources: tuple[str, ...]


def slug(name: str) -> str:
    """Lower-case file stem of a Participant name; runs of non-word characters become `-`."""
    stem = re.sub(r"[\W_]+", "-", name.strip().casefold()).strip("-")
    if not stem:
        raise ProfileError(
            f"cannot make a profile file name from {name!r}; use a name with letters or digits"
        )
    return stem


def save_sample(folder: Path, participant: str, profiles_dir: Path) -> tuple[Path, int]:
    """Save `participant`'s voice from Meeting `folder`; returns the file and its sample count."""
    participant = participant.strip()
    stem = slug(participant)
    label, spelling = _speaker_label(folder, participant)
    entry = _embeddings_entry(folder, label)
    sample: dict[str, Any] = {
        "source": folder.resolve().name,
        "vector": entry["mean_of_windows"],
        "windows": entry["windows"],
        "model": entry["model"],
    }
    path = profiles_dir / f"{stem}.json"
    profile: dict[str, Any] = (
        _read_profile(path) if path.exists() else {"version": VERSION, "name": spelling}
    )
    others = [s for s in profile.setdefault("samples", []) if s["source"] != sample["source"]]
    for other in others:
        if other["model"] != sample["model"]:
            raise ProfileError(
                f"this Meeting's embeddings come from a different model than the saved samples "
                f"of {profile['name']!r}; re-run scribe on this Recording with the model "
                f"the samples came from, or move the profile file away to start it again"
            )
        if len(other["vector"]) != len(sample["vector"]):
            raise ProfileError(
                f"this Meeting's vector has {len(sample['vector'])} dimensions but the saved "
                f"samples of {profile['name']!r} have {len(other['vector'])}; "
                f"check that {EMBEDDINGS_NAME} was not edited"
            )
    profile["samples"] = [*others, sample]
    _write_profile(path, profile)
    return path, len(profile["samples"])


def list_profiles(profiles_dir: Path) -> list[ProfileSummary]:
    """Name, sample count and source names of each profile, sorted by file name."""
    if not profiles_dir.is_dir():
        return []
    summaries = []
    for path in sorted(profiles_dir.glob("*.json")):
        profile = _read_profile(path)
        samples = profile["samples"]
        summaries.append(
            ProfileSummary(profile["name"], len(samples), tuple(s["source"] for s in samples))
        )
    return summaries


def load_profiles(profiles_dir: Path) -> list[dict[str, Any]]:
    """Every profile file's validated contents, sorted by file name; [] without the directory."""
    if not profiles_dir.is_dir():
        return []
    return [_read_profile(path) for path in sorted(profiles_dir.glob("*.json"))]


def format_summary(summary: ProfileSummary) -> str:
    sources = ", ".join(summary.sources) or "-"
    return f"{summary.name}: {summary.samples} sample(s) from {sources}"


def _speaker_label(folder: Path, participant: str) -> tuple[str, str]:
    """The Transcript label of the one Speaker `participant` is named on in this Meeting, and the
    spelling transcript.json gives that name."""
    if not folder.is_dir():
        raise ProfileError(
            f"no Meeting folder {str(folder)!r}; pass the path to a Meeting folder or its name "
            "under --meetings-dir"
        )
    try:
        transcript = parse_json((folder / JSON_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, TranscriptError) as error:
        raise ProfileError(
            f"cannot read {JSON_NAME} as a Transcript in this Meeting folder; "
            "run scribe on the Recording first"
        ) from error
    wanted = participant.casefold()
    named: dict[str, str] = {}
    for segment in transcript.segments:
        if segment.participant is not None:
            named.setdefault(segment.speaker, segment.participant)
    labels = sorted(label for label, name in named.items() if name.strip().casefold() == wanted)
    if not labels:
        known = ", ".join(sorted(set(named.values()))) or "no one yet"
        raise ProfileError(
            f"{participant!r} is not named in this Meeting (named: {known}); "
            f'run scribe name <meeting> "Speaker N={participant}" first, or check the spelling'
        )
    if len(labels) > 1:
        raise ProfileError(
            f"{participant!r} is named on {len(labels)} Speakers ({', '.join(labels)}) in this "
            "Meeting, so scribe cannot tell which voice to save; run scribe name <meeting> to "
            "give the extra Speaker its own name, then save again"
        )
    return labels[0], named[labels[0]]


def _embeddings_entry(folder: Path, label: str) -> dict[str, Any]:
    path = folder / EMBEDDINGS_NAME
    if not path.is_file():
        raise ProfileError(
            f"no {EMBEDDINGS_NAME} in this Meeting folder (a Meeting from before embeddings "
            "were written); re-run scribe on the Recording with --force (that drops the Naming, so "
            "name the Speakers again)"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        entry = document["speakers"][label]
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise ProfileError(
            f"{EMBEDDINGS_NAME} is unreadable or has no entry for {label!r}; re-run scribe on the "
            "Recording with --force (that drops the Naming, so name the Speakers again)"
        ) from error
    vector = entry.get("mean_of_windows") if isinstance(entry, dict) else None
    windows = entry.get("windows") if isinstance(entry, dict) else None
    model = entry.get("model") if isinstance(entry, dict) else None
    if not isinstance(windows, int) or isinstance(windows, bool) or windows < MIN_WINDOWS:
        raise ProfileError(
            f"{label!r} has {windows if isinstance(windows, int) else 'no'} usable 3 s windows "
            f"in this Meeting, fewer than the {MIN_WINDOWS} a profile needs; save from a "
            "Meeting where this Participant speaks longer"
        )
    if not _is_vector(vector) or not isinstance(model, str) or not model:
        raise ProfileError(
            f"{EMBEDDINGS_NAME} holds no usable mean of windows or model for {label!r}; re-run "
            "scribe on the Recording with --force (that drops the Naming, so name the Speakers "
            "again)"
        )
    return {"mean_of_windows": vector, "windows": windows, "model": model}


def _is_vector(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
    )


def _read_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
        if profile["version"] != VERSION or not isinstance(profile["name"], str):
            raise ValueError("version or name")
        for sample in profile["samples"]:
            if not (
                isinstance(sample["source"], str)
                and isinstance(sample["model"], str)
                and _is_vector(sample["vector"])
            ):
                raise ValueError("sample")
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise ProfileError(
            f"cannot read {path} as a Voice profile; move it away and save again"
        ) from error
    assert isinstance(profile, dict)
    return profile


def _write_profile(path: Path, profile: dict[str, Any]) -> None:
    text = json.dumps(profile, indent=1) + "\n"
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except OSError as error:
        raise ProfileError(
            f"cannot write {path}: {error.strerror or error}; pass another --profiles-dir"
        ) from error
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
