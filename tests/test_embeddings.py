"""Embedding rows from pyannote's output, the 3 s window rule and the embeddings.json document.

Random vectors only, no models: these run in CI without the asr extra.
"""

import math
import random
from types import SimpleNamespace
from typing import Any

import pytest

from scribe.diarize import DiarizeError, embedding_rows
from scribe.embeddings import Window, build, mean_of_windows, unit, windows
from scribe.merge import SpeakerSpan

DIM = 256


def vector(rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


class FakeAnnotation:
    def __init__(self, labels: list[str]) -> None:
        self._labels = labels

    def labels(self) -> list[str]:
        return self._labels


def fake_output(labels: list[str], exclusive: list[str], rows: Any) -> SimpleNamespace:
    return SimpleNamespace(
        speaker_diarization=FakeAnnotation(labels),
        exclusive_speaker_diarization=FakeAnnotation(exclusive),
        speaker_embeddings=rows,
    )


def test_rows_follow_speaker_diarization_labels_and_drop_zero_rows() -> None:
    rng = random.Random(1)
    a, b, d = vector(rng), vector(rng), vector(rng)
    # SPEAKER_01 speaks only in overlap, so the exclusive labels skip it; SPEAKER_02's row is
    # pyannote's zero padding, and it is not the last row. Reading rows by the exclusive labels
    # would give SPEAKER_02 row b and SPEAKER_03 the zero row.
    output = fake_output(
        labels=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
        exclusive=["SPEAKER_00", "SPEAKER_02", "SPEAKER_03"],
        rows=[a, b, [0.0] * DIM, d],
    )

    rows = embedding_rows(output)

    assert rows == {"SPEAKER_00": a, "SPEAKER_01": b, "SPEAKER_03": d}


def test_no_embeddings_gives_no_rows() -> None:
    assert embedding_rows(fake_output(["SPEAKER_00"], ["SPEAKER_00"], None)) == {}


def test_row_count_that_does_not_match_the_labels_is_refused() -> None:
    output = fake_output(["SPEAKER_00", "SPEAKER_01"], ["SPEAKER_00"], [[1.0] * DIM])
    with pytest.raises(DiarizeError, match="1 embedding rows for 2 Speakers"):
        embedding_rows(output)


@pytest.mark.parametrize(
    ("end", "starts"),
    [
        (13.49, []),  # 3.49 s turn: 2.99 s after trimming
        (13.5, [10.25]),  # 3.5 s turn: exactly one window
        (16.49, [10.25]),  # remainder dropped
        (16.5, [10.25, 13.25]),
    ],
)
def test_window_rule_trims_then_tiles_inside_one_turn(end: float, starts: list[float]) -> None:
    cut = windows([SpeakerSpan(10.0, end, "SPEAKER_00")], duration=60.0)
    assert [w.start for w in cut] == pytest.approx(starts)
    assert all(w.speaker == "SPEAKER_00" for w in cut)


def test_windows_never_cross_a_turn_boundary() -> None:
    # Two touching 2 s turns of one Speaker give no window, though together they last 4 s.
    spans = [SpeakerSpan(0.0, 2.0, "SPEAKER_00"), SpeakerSpan(2.0, 4.0, "SPEAKER_00")]
    assert windows(spans, duration=60.0) == []


def test_window_past_the_end_of_the_audio_is_dropped() -> None:
    spans = [SpeakerSpan(0.0, 10.0, "SPEAKER_00")]
    assert [w.start for w in windows(spans, duration=6.0)] == [0.25]


def test_unit_and_mean_of_windows() -> None:
    assert unit([3.0, 4.0]) == [0.6, 0.8]
    assert unit([0.0, 0.0]) is None
    assert unit([math.nan, 1.0]) is None
    # Unit vectors are averaged, so a long vector does not outweigh a short one.
    mean, count = mean_of_windows([[10.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    assert count == 2
    assert mean == pytest.approx([math.sqrt(0.5), math.sqrt(0.5)])
    assert mean_of_windows([]) == (None, 0)


def test_build_keys_by_transcript_label_and_omits_unmapped_raw_labels() -> None:
    rng = random.Random(2)
    centroids = {"SPEAKER_00": vector(rng), "SPEAKER_01": vector(rng), "SPEAKER_09": vector(rng)}
    cut = [Window(0.25, "SPEAKER_01"), Window(3.25, "SPEAKER_01"), Window(9.25, "SPEAKER_09")]
    vectors = [vector(rng) for _ in cut]

    document = build(
        {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}, centroids, cut, vectors, "m@1"
    )

    assert list(document["speakers"]) == ["Speaker 1", "Speaker 2"]
    first = document["speakers"]["Speaker 1"]
    assert first["centroid"] == pytest.approx(unit(centroids["SPEAKER_01"]))
    assert first["windows"] == 2
    assert first["mean_of_windows"] == pytest.approx(mean_of_windows(vectors[:2])[0])
    assert document["speakers"]["Speaker 2"]["windows"] == 0
    assert {e["model"] for e in document["speakers"].values()} == {"m@1"}


def test_build_refuses_a_vector_count_that_does_not_match_the_windows() -> None:
    with pytest.raises(ValueError, match="1 windows but 0 window embeddings"):
        build({"SPEAKER_00": "Speaker 1"}, {}, [Window(0.25, "SPEAKER_00")], [], "m@1")
