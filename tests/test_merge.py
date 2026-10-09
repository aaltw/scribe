"""Merge tests use synthetic spans and words with invented text, never a real Meeting."""

from scribe.merge import SpeakerSpan, Word, merge, merge_with_labels
from scribe.transcript import Segment


def test_words_inside_spans_form_one_segment_per_speaker_turn() -> None:
    spans = [SpeakerSpan(0.0, 2.0, "SPEAKER_00"), SpeakerSpan(2.0, 4.0, "SPEAKER_01")]
    words = [
        Word(0.1, 0.5, "Goedemorgen"),
        Word(0.6, 1.0, "allemaal"),
        Word(2.2, 2.6, "Hello"),
        Word(2.7, 3.1, "there"),
    ]
    assert merge(spans, words) == (
        Segment(0.1, 1.0, "Speaker 1", "Goedemorgen allemaal"),
        Segment(2.2, 3.1, "Speaker 2", "Hello there"),
    )


def test_word_straddling_a_speaker_change_goes_to_the_span_holding_its_midpoint() -> None:
    spans = [SpeakerSpan(0.0, 2.0, "SPEAKER_00"), SpeakerSpan(2.0, 4.0, "SPEAKER_01")]
    # Midpoint 2.15 lies in the second span, though the word starts in the first.
    words = [Word(0.5, 1.0, "eerst"), Word(1.9, 2.4, "daarna")]
    assert merge(spans, words) == (
        Segment(0.5, 1.0, "Speaker 1", "eerst"),
        Segment(1.9, 2.4, "Speaker 2", "daarna"),
    )


def test_zero_duration_word_goes_to_the_span_containing_it() -> None:
    spans = [SpeakerSpan(0.0, 1.0, "SPEAKER_00"), SpeakerSpan(1.0, 2.0, "SPEAKER_01")]
    words = [Word(0.4, 0.4, "ja"), Word(1.52, 1.52, "nee")]
    assert merge(spans, words) == (
        Segment(0.4, 0.4, "Speaker 1", "ja"),
        Segment(1.52, 1.52, "Speaker 2", "nee"),
    )


def test_midpoint_on_touching_edge_goes_to_the_span_with_more_overlap() -> None:
    # A short span (under 0.3 s, as pyannote produces) ends where the next one starts.
    spans = [SpeakerSpan(1.9, 2.0, "SPEAKER_00"), SpeakerSpan(2.0, 3.0, "SPEAKER_01")]
    # Midpoint 2.0 is in both; overlap is 0.1 with the first and 0.2 with the second.
    words = [Word(1.8, 2.2, "precies")]
    assert merge(spans, words) == (Segment(1.8, 2.2, "Speaker 1", "precies"),)
    # Speaker 1 is SPEAKER_01 here: the only Speaker with a word.
    assert merge(spans, [Word(0.0, 0.1, "x"), *words])[1].speaker == "Speaker 2"


def test_exact_tie_goes_to_the_earlier_span() -> None:
    spans = [SpeakerSpan(1.0, 2.0, "SPEAKER_00"), SpeakerSpan(2.0, 3.0, "SPEAKER_01")]
    words = [Word(2.5, 2.6, "later"), Word(2.0, 2.0, "grens")]
    assert merge(spans, words) == (
        Segment(2.0, 2.0, "Speaker 1", "grens"),
        Segment(2.5, 2.6, "Speaker 2", "later"),
    )


def test_zero_duration_word_in_a_gap_goes_to_the_nearest_span_edge() -> None:
    spans = [SpeakerSpan(0.0, 1.0, "SPEAKER_00"), SpeakerSpan(2.0, 3.0, "SPEAKER_01")]
    words = [Word(0.5, 0.6, "een"), Word(1.2, 1.2, "twee"), Word(1.8, 1.8, "drie")]
    assert merge(spans, words) == (
        Segment(0.5, 1.2, "Speaker 1", "een twee"),
        Segment(1.8, 1.8, "Speaker 2", "drie"),
    )


def test_word_in_a_gap_goes_to_the_span_it_overlaps() -> None:
    spans = [SpeakerSpan(0.0, 1.0, "SPEAKER_00"), SpeakerSpan(2.0, 3.0, "SPEAKER_01")]
    # Midpoint 1.75 is in the gap; the word overlaps only the second span.
    words = [Word(0.2, 0.4, "hoi"), Word(1.5, 2.0, "welkom")]
    assert merge(spans, words) == (
        Segment(0.2, 0.4, "Speaker 1", "hoi"),
        Segment(1.5, 2.0, "Speaker 2", "welkom"),
    )


def test_words_before_and_after_all_spans_are_kept() -> None:
    spans = [SpeakerSpan(1.0, 2.0, "SPEAKER_00")]
    words = [Word(0.2, 0.3, "vooraf"), Word(1.5, 1.6, "midden"), Word(5.0, 5.1, "achteraf")]
    assert merge(spans, words) == (Segment(0.2, 5.1, "Speaker 1", "vooraf midden achteraf"),)


def test_speakers_are_numbered_by_first_assigned_word() -> None:
    spans = [
        SpeakerSpan(0.0, 1.0, "SPEAKER_01"),  # speaks first, but no words fall here
        SpeakerSpan(1.0, 2.0, "SPEAKER_00"),
        SpeakerSpan(2.0, 3.0, "SPEAKER_02"),
        SpeakerSpan(3.0, 4.0, "SPEAKER_00"),
    ]
    words = [Word(1.2, 1.4, "a"), Word(2.2, 2.4, "b"), Word(3.2, 3.4, "c")]
    assert [s.speaker for s in merge(spans, words)] == ["Speaker 1", "Speaker 2", "Speaker 1"]


def test_unsorted_input_is_ordered_by_time() -> None:
    spans = [SpeakerSpan(2.0, 4.0, "SPEAKER_01"), SpeakerSpan(0.0, 2.0, "SPEAKER_00")]
    words = [Word(2.2, 2.6, "tweede"), Word(0.1, 0.5, "eerste")]
    assert merge(spans, words) == (
        Segment(0.1, 0.5, "Speaker 1", "eerste"),
        Segment(2.2, 2.6, "Speaker 2", "tweede"),
    )


def test_text_joins_words_with_spaces_and_attaches_punctuation() -> None:
    spans = [SpeakerSpan(0.0, 5.0, "SPEAKER_00")]
    tokens = ["Oké", ",", "shall", "we", "(", "kort", ")", "beginnen", "?", " Ja", "!", ""]
    words = [Word(i * 0.3, i * 0.3, t) for i, t in enumerate(tokens)]
    assert merge(spans, words)[0].text == "Oké, shall we (kort) beginnen? Ja!"


def test_empty_diarization_puts_all_words_under_speaker_1() -> None:
    words = [Word(0.1, 0.3, "niemand"), Word(0.4, 0.6, "gevonden")]
    assert merge([], words) == (Segment(0.1, 0.6, "Speaker 1", "niemand gevonden"),)


def test_empty_word_list_gives_no_segments() -> None:
    assert merge([SpeakerSpan(0.0, 1.0, "SPEAKER_00")], []) == ()


def test_empty_diarization_and_empty_word_list_give_no_segments() -> None:
    assert merge([], []) == ()


def test_closing_punctuation_stays_with_the_segment_of_the_word_it_closes() -> None:
    spans = [
        SpeakerSpan(0.0, 2.0, "SPEAKER_00"),
        SpeakerSpan(2.0, 4.0, "SPEAKER_01"),
        SpeakerSpan(4.0, 6.0, "SPEAKER_00"),
    ]
    # The "." and "?" have midpoints (2.04, 4.04) inside the span of whoever speaks next.
    words = [
        Word(0.5, 1.0, "eerste"),
        Word(1.98, 2.1, "."),
        Word(2.2, 2.6, "tweede"),
        Word(3.98, 4.1, "?"),
        Word(4.2, 4.6, "derde"),
    ]
    segments = merge(spans, words)
    assert [s.text for s in segments] == ["eerste.", "tweede?", "derde"]
    assert [s.speaker for s in segments] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    assert not any(s.text.startswith(tuple(".?")) for s in segments)


def test_merge_with_labels_returns_the_raw_to_label_map_merge_used() -> None:
    spans = [
        SpeakerSpan(0.0, 1.0, "SPEAKER_01"),
        SpeakerSpan(1.0, 2.0, "SPEAKER_02"),  # spans but no words: not in the map
        SpeakerSpan(2.0, 3.0, "SPEAKER_00"),
    ]
    words = [Word(0.2, 0.5, "Pineapple"), Word(2.2, 2.5, "harbour")]

    segments, labels = merge_with_labels(spans, words)

    assert labels == {"SPEAKER_01": "Speaker 1", "SPEAKER_00": "Speaker 2"}
    assert list(labels) == ["SPEAKER_01", "SPEAKER_00"]  # first-heard order
    assert segments == merge(spans, words)
    assert {s.speaker for s in segments} == set(labels.values())


def test_merge_with_labels_without_spans_has_an_empty_map() -> None:
    segments, labels = merge_with_labels([], [Word(0.0, 0.5, "Pineapple")])
    assert labels == {}
    assert [s.speaker for s in segments] == ["Speaker 1"]
