"""Pipeline step 4: merge diarization spans and word timings into Segments.

Pure Python, no model dependencies. The adapters return `SpeakerSpan`s (pyannote's exclusive
diarization) and `Word`s (Parakeet's word timings); `merge` turns them into the `Segment`s of a
Transcript. The pipeline builds the Transcript's Speaker list from the result with
`tuple(dict.fromkeys(s.speaker for s in segments))`, which is first-heard order.

Public names: `SpeakerSpan`, `Word`, `merge`, `merge_with_labels`.

Assigning a word to a Speaker, in order:

1. The spans that contain the word's midpoint (closed intervals, so touching spans both
   contain a midpoint on their shared edge).
2. If none does, the midpoint sits in a gap between spans. Gap rule: the word still goes to a
   neighbouring span, never dropped. Parakeet times sit on ~80 ms frames and pyannote trims
   speech edges, so a word just outside every span was almost always said by the Speaker of
   the span next to it; dropping it would lose text the user needs, and the text came from audio
   someone spoke.
3. Among the candidates, the largest overlap with `[word.start, word.end]` wins; then the span
   whose nearest edge is closest to the midpoint; then the earlier span. Zero-duration words
   overlap nothing, so in a gap they go to the nearest span edge.

Closing punctuation: a Word made only of closing punctuation (`.,!?;:%)]}…`, the same set as
the join rule below) closes the word before it, so it takes that word's Speaker wherever its
midpoint falls. Parakeet puts punctuation on its own ~80 ms frame, often right at a Speaker
change, and the midpoint rule alone would start the next Segment with ". " and leave the
previous one without its full stop. A mark with no preceding word is assigned like any word.
Real words are unaffected.

Speaker labels: pyannote's raw labels (`SPEAKER_00`, ...) become `Speaker 1`, `Speaker 2`, ...
in order of each Speaker's first assigned word, so a Speaker with spans but no words never
leaves a hole in the numbering. With no spans at all but words present, every word goes to
`Speaker 1`: inventing one Speaker beats losing the text. `merge_with_labels` also returns
this raw-to-label map, so the embed step keys embeddings by the same labels without working
the mapping out again; a raw label with spans but no words is not in it.

Segments: consecutive words of one Speaker form a Segment, however long the pause between them.
A Segment runs from its first word's start to its latest word end. Text is the stripped word
texts joined by single spaces, with no space before a token starting with closing punctuation
(`.,!?;:%)]}…`) and none after an opening bracket (`([{`). Words with empty text are skipped.

Lookup uses bisect on span starts and checks only the neighbouring spans, which assumes the
spans do not overlap; pyannote's exclusive diarization guarantees that.
"""

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass

from scribe.transcript import Segment

_NO_SPACE_BEFORE = tuple(".,!?;:%)]}…")
_NO_SPACE_AFTER = tuple("([{")


@dataclass(frozen=True)
class SpeakerSpan:
    """One span of pyannote's exclusive diarization: seconds from audio start, raw label."""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Word:
    """One timed word from the ASR, in seconds from audio start.

    `text` is one whole word or one punctuation mark; the adapter joins sub-word tokens into
    words before building a `Word`. Surrounding whitespace is ignored.
    """

    start: float
    end: float
    text: str


def merge(spans: Iterable[SpeakerSpan], words: Iterable[Word]) -> tuple[Segment, ...]:
    """Assign each word to a Speaker and group consecutive words of one Speaker into Segments."""
    return merge_with_labels(spans, words)[0]


def merge_with_labels(
    spans: Iterable[SpeakerSpan], words: Iterable[Word]
) -> tuple[tuple[Segment, ...], dict[str, str]]:
    """`merge`'s Segments plus its map from raw pyannote label to Transcript label.

    The map is in first-heard order and holds only raw labels that got a word. With no spans
    it is empty, though the Segments then all go to `Speaker 1`.
    """
    ordered_spans = sorted(spans, key=lambda s: (s.start, s.end))
    starts = [s.start for s in ordered_spans]
    ordered_words = sorted((w for w in words if w.text.strip()), key=lambda w: (w.start, w.end))

    labels: dict[str | None, str] = {}
    runs: list[tuple[str, list[Word]]] = []
    for word in ordered_words:
        if runs and _is_closing_punctuation(word):
            label = runs[-1][0]
        else:
            raw = _assign(word, ordered_spans, starts).speaker if ordered_spans else None
            label = labels.setdefault(raw, f"Speaker {len(labels) + 1}")
        if runs and runs[-1][0] == label:
            runs[-1][1].append(word)
        else:
            runs.append((label, [word]))

    segments = tuple(
        Segment(
            start=run[0].start,
            end=max(w.end for w in run),
            speaker=label,
            text=_join(w.text for w in run),
        )
        for label, run in runs
    )
    return segments, {raw: label for raw, label in labels.items() if raw is not None}


def _assign(word: Word, spans: list[SpeakerSpan], starts: list[float]) -> SpeakerSpan:
    midpoint = (word.start + word.end) / 2
    i = bisect_right(starts, midpoint)
    # With non-overlapping spans, i-1 (and i-2 when touching) can contain the midpoint, and
    # i-1 and i are the neighbours on either side of a gap.
    window = spans[max(0, i - 2) : i + 1]
    containing = [s for s in window if s.start <= midpoint <= s.end]
    candidates = containing or window
    # max keeps the first of equal keys, and the window is in time order: earlier span wins.
    return max(candidates, key=lambda s: (_overlap(s, word), -_distance(s, midpoint)))


def _is_closing_punctuation(word: Word) -> bool:
    return all(c in _NO_SPACE_BEFORE for c in word.text.strip())


def _overlap(span: SpeakerSpan, word: Word) -> float:
    return max(0.0, min(span.end, word.end) - max(span.start, word.start))


def _distance(span: SpeakerSpan, point: float) -> float:
    return max(0.0, span.start - point, point - span.end)


def _join(texts: Iterable[str]) -> str:
    text = ""
    for token in (t.strip() for t in texts):
        if text and not token.startswith(_NO_SPACE_BEFORE) and not text.endswith(_NO_SPACE_AFTER):
            text += " "
        text += token
    return text
