"""Suggest a Participant per Speaker from the saved Voice profiles.

`suggest` compares each Speaker's centroid in a Meeting's `embeddings.json` with each profile's
vector and writes `<meeting>/suggestions.json`. A suggestion never names anyone by itself: the
/transcribe skill shows it to the user, who confirms before Naming. The file is a contract.

A profile's vector is the unit mean of its samples' vectors (each sample is a unit mean of
windows, so the profile is a mean of mean-of-windows vectors). A Speaker's vector is its
centroid. Both are unit length, so their cosine is their dot product.

Assignment is one-to-one and maximises the total cosine over the pairs at or above
`SUGGEST_THRESHOLD`, so a below-threshold pair never pushes out a valid one; a Speaker or
profile left over gets no suggestion. The method is an exact dynamic programme over a bitmask of
the smaller side (Speakers or profiles), refused above `MAX_SIDE` on that side, which no
Meeting reaches. Label order and file order never decide: Speakers and profiles are put in an
order taken from their vectors and names, and a tie on total cosine goes to the first option in
that order. Only Speakers with identical centroids fall back to label order, as nothing else
tells them apart.

Profiles whose model differs from the Meeting's are skipped; refusal (`SuggestError`) is only
for a missing or unreadable `embeddings.json`, for profiles that all come from another model,
and for a Meeting past the size cap. The unused diagnostic, the cosine of the Speaker's mean of
windows with the profile, is recorded and never used to decide.

File format (version 1), Speakers in `embeddings.json` order:

    {"version": 1,
     "threshold": 0.8401,
     "suggestions": [{"label": "Speaker 1",
                      "participant": "<stored profile name>" | null,
                      "cosine": <centroid cosine> | null,
                      "mean_of_windows_cosine": <diagnostic> | null,
                      "model": "<diarize.model_version() from embeddings.json>"}]}

For a Speaker with no suggestion, `cosine` is the best centroid cosine against a profile of the
same model, or null when there is none (no profiles, or no centroid); `participant` is null.

Suggestions are biometric data (ADR 0001): nothing here prints a vector value.
"""

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from scribe.embeddings import FILE_NAME as EMBEDDINGS_NAME
from scribe.embeddings import unit
from scribe.profiles import load_profiles

FILE_NAME = "suggestions.json"
VERSION = 1
# Cosine between a Speaker's centroid (pyannote's speaker_embeddings row) and a stored profile.
# Calibrated once on held-out data before it was scored: the top of the
# range that gives the fewest train errors, and itself a train same-Speaker score. Calibrated on
# single-sample profiles from one Recording of two people, so a profile of several samples may
# want it re-checked.
SUGGEST_THRESHOLD = 0.8401
MAX_SIDE = 16
_TIE = 1e-12


class SuggestError(Exception):
    """A refusal; the message ends with a next step the user can act on."""


@dataclass(frozen=True)
class Suggestion:
    label: str
    participant: str | None
    cosine: float | None
    mean_of_windows_cosine: float | None
    model: str


@dataclass(frozen=True)
class _Speaker:
    label: str
    model: str
    centroid: list[float] | None
    mean_of_windows: list[float] | None


@dataclass(frozen=True)
class _Profile:
    name: str
    model: str
    vector: list[float]


def suggest(folder: Path, profiles_dir: Path) -> tuple[list[Suggestion], int]:
    """Write `<folder>/suggestions.json`; returns the suggestions and how many profiles were
    skipped for another model (or for having no usable samples)."""
    speakers = _read_speakers(folder)
    profiles, skipped = _usable_profiles(profiles_dir, {s.model for s in speakers})
    scores = _scores(speakers, profiles)
    assignment = _assign(speakers, profiles, scores)
    suggestions = [
        _suggestion(i, speaker, profiles, scores, assignment) for i, speaker in enumerate(speakers)
    ]
    _write(folder / FILE_NAME, suggestions)
    return suggestions, skipped


def format_line(suggestion: Suggestion) -> str:
    """One text line: the stored profile name and cosine to 3 decimals, never a vector."""
    if suggestion.participant is not None and suggestion.cosine is not None:
        return f"{suggestion.label}: {suggestion.participant} (cosine {suggestion.cosine:.3f})"
    if suggestion.cosine is not None:
        return f"{suggestion.label}: no suggestion (best cosine {suggestion.cosine:.3f})"
    return f"{suggestion.label}: no suggestion"


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Dot product of two unit vectors; None when their lengths differ."""
    if len(a) != len(b):
        return None
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _vector(value: object) -> list[float] | None:
    if (
        isinstance(value, list)
        and value
        and all(
            isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)
            for x in value
        )
    ):
        return [float(x) for x in value]
    return None


def _read_speakers(folder: Path) -> list[_Speaker]:
    if not folder.is_dir():
        raise SuggestError(
            f"no Meeting folder {str(folder)!r}; pass the path to a Meeting folder or its name "
            "under --meetings-dir"
        )
    path = folder / EMBEDDINGS_NAME
    if not path.is_file():
        raise SuggestError(
            f"no {EMBEDDINGS_NAME} in this Meeting folder (a Meeting from before embeddings "
            "were written); re-run scribe on the Recording with --force (that drops the Naming, "
            "so name the Speakers again)"
        )
    unreadable = SuggestError(
        f"cannot read {EMBEDDINGS_NAME} in this Meeting folder; re-run scribe on the Recording "
        "with --force (that drops the Naming, so name the Speakers again)"
    )
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["speakers"]
        if not isinstance(entries, dict):
            raise TypeError("speakers")
        speakers = []
        for label, entry in entries.items():
            model = entry["model"]
            centroid, mean = entry["centroid"], entry["mean_of_windows"]
            if (
                not isinstance(model, str)
                or not model
                or (centroid is not None and _vector(centroid) is None)
                or (mean is not None and _vector(mean) is None)
            ):
                raise ValueError("entry")
            speakers.append(_Speaker(label, model, _vector(centroid), _vector(mean)))
    except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as error:
        raise unreadable from error
    return speakers


def _usable_profiles(profiles_dir: Path, models: set[str]) -> tuple[list[_Profile], int]:
    """Profiles whose samples all come from one of the Meeting's `models`, in an order taken
    from their names and vectors, and how many were skipped. Raises when every profile is."""
    found = load_profiles(profiles_dir)
    usable = []
    for profile in found:
        samples = profile["samples"]
        sample_models = {s["model"] for s in samples}
        if not samples or len(sample_models) != 1 or not sample_models <= models:
            continue
        if len({len(s["vector"]) for s in samples}) != 1:
            continue
        mean = unit(
            [math.fsum(column) / len(samples) for column in zip(*(s["vector"] for s in samples))]
        )
        if mean is not None:
            usable.append(_Profile(profile["name"], sample_models.pop(), mean))
    if found and not usable:
        raise SuggestError(
            f"none of the {len(found)} saved Voice profile(s) comes from this Meeting's "
            "embedding model, so scribe cannot compare them; re-run scribe on the Recording "
            "with the model the profiles came from, or move the profile files away and save "
            "them again from Meetings made with this model"
        )
    usable.sort(key=lambda p: (p.name.casefold(), p.name, p.vector))
    return usable, len(found) - len(usable)


def _scores(
    speakers: Sequence[_Speaker], profiles: Sequence[_Profile]
) -> dict[tuple[int, int], float]:
    """Centroid cosine of every (speaker, profile) pair of the same model and dimension."""
    scores: dict[tuple[int, int], float] = {}
    for i, speaker in enumerate(speakers):
        if speaker.centroid is None:
            continue
        for j, profile in enumerate(profiles):
            if profile.model != speaker.model:
                continue
            score = cosine(speaker.centroid, profile.vector)
            if score is not None:
                scores[i, j] = score
    return scores


def _assign(
    speakers: Sequence[_Speaker],
    profiles: Sequence[_Profile],
    scores: Mapping[tuple[int, int], float],
) -> dict[int, int]:
    """{speaker index: profile index} maximising the total cosine of pairs at or above the
    threshold; ties go to the first option in the order of the vectors, never the label."""
    eligible = {pair: s for pair, s in scores.items() if s >= SUGGEST_THRESHOLD}
    if not eligible:
        return {}
    speaker_order = sorted(
        {i for i, _ in eligible}, key=lambda i: (speakers[i].centroid, speakers[i].label)
    )
    profile_order = sorted({j for _, j in eligible})  # profiles are already in canonical order
    transpose = len(profile_order) > len(speaker_order)
    if transpose:
        rows, cols = profile_order, speaker_order
        gain = {(j, i): s for (i, j), s in eligible.items()}
    else:
        rows, cols = speaker_order, profile_order
        gain = dict(eligible)
    if len(cols) > MAX_SIDE:
        raise SuggestError(
            f"this Meeting has {len(cols)} Speakers and profiles that could match, more than "
            f"the {MAX_SIDE} scribe suggest can assign exactly; save fewer profiles or pass "
            "another --profiles-dir"
        )
    bit = {c: k for k, c in enumerate(cols)}
    options = [[(c, gain[r, c]) for c in cols if (r, c) in gain] for r in rows]

    @cache
    def best(k: int, used: int) -> float:
        if k == len(rows):
            return 0.0
        top = best(k + 1, used)
        for c, s in options[k]:
            if not used >> bit[c] & 1:
                top = max(top, s + best(k + 1, used | 1 << bit[c]))
        return top

    chosen: dict[int, int] = {}
    used = 0
    for k, row in enumerate(rows):
        target = best(k, used)
        for c, s in options[k]:
            if not used >> bit[c] & 1 and s + best(k + 1, used | 1 << bit[c]) >= target - _TIE:
                chosen[row] = c
                used |= 1 << bit[c]
                break
    return {c: r for r, c in chosen.items()} if transpose else chosen


def _suggestion(
    index: int,
    speaker: _Speaker,
    profiles: Sequence[_Profile],
    scores: Mapping[tuple[int, int], float],
    assignment: Mapping[int, int],
) -> Suggestion:
    if index in assignment:
        match: int | None = assignment[index]
    else:
        ranked = [(s, -j) for (i, j), s in scores.items() if i == index]
        match = -max(ranked)[1] if ranked else None
    if match is None:
        return Suggestion(speaker.label, None, None, None, speaker.model)
    diagnostic = (
        cosine(speaker.mean_of_windows, profiles[match].vector)
        if speaker.mean_of_windows is not None
        else None
    )
    participant = profiles[match].name if index in assignment else None
    return Suggestion(speaker.label, participant, scores[index, match], diagnostic, speaker.model)


def _write(path: Path, suggestions: Sequence[Suggestion]) -> None:
    document: dict[str, Any] = {
        "version": VERSION,
        "threshold": SUGGEST_THRESHOLD,
        "suggestions": [
            {
                "label": s.label,
                "participant": s.participant,
                "cosine": s.cosine,
                "mean_of_windows_cosine": s.mean_of_windows_cosine,
                "model": s.model,
            }
            for s in suggestions
        ],
    }
    text = json.dumps(document, indent=1) + "\n"
    tmp: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except OSError as error:
        raise SuggestError(
            f"cannot write {path}: {error.strerror or error}; check the Meeting folder's "
            "permissions"
        ) from error
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
