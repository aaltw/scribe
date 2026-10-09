# scribe

Turns a meeting recording into a transcript that says who said what.

## Language

**Recording**:
A video or audio file of one meeting, given to scribe as input. Only its audio is used.
_Avoid_: video, file, input

**Meeting**:
The conversation captured in one Recording; scribe's output is grouped per Meeting.

**Speaker**:
One distinct voice found in a Recording, known only by an anonymous label (Speaker 1, Speaker 2) until named.
_Avoid_: user, person, participant

**Participant**:
A real, named person (you, a colleague) that a Speaker turns out to be.
_Avoid_: user, speaker

**Naming**:
The step where the user tells scribe which Participant each Speaker is, after seeing a few of that Speaker's lines.

**Voice profile**:
Stored voice embeddings (not audio samples) of one Participant, kept locally, used to suggest a Naming. Never decides on its own; the user confirms.

**Segment**:
A stretch of speech by one Speaker, with start and end time and its text.
_Avoid_: line, chunk, turn

**Transcript**:
The ordered Segments of one Meeting, written as Markdown for reading and JSON for tools.

**Clean Transcript**:
A copy of a Transcript with misheard and phonetic spellings corrected by Claude, keeping every Segment, Speaker and timestamp. The raw Transcript stays as heard.
_Avoid_: fixed transcript, edited transcript

**Minutes**:
Structured notes of a Meeting per topic: what was discussed, decided and agreed. Longer than a Summary, shorter than a Transcript.
_Avoid_: notes, report

**Speaker stats**:
Per Speaker or Participant in one Meeting: talk time, share of talk time and Segment count. Computed from the Transcript, not judged by a model.

**Summary**:
A short digest of a Meeting (decisions, action items). Derived from the Clean Transcript (or the raw one), never from the audio.

## Relationships

- A **Recording** captures exactly one **Meeting**.
- A **Meeting** has one or more **Speakers**; each **Speaker** is named as at most one **Participant**.
- A **Transcript** is a sequence of **Segments**; each **Segment** belongs to exactly one **Speaker**.
- A **Voice profile** belongs to one **Participant** and only ever suggests a **Naming**.

## Example dialogue

> **User:** "Run scribe on today's recording."
> **Claude:** "Found 2 Speakers. Speaker 1 says 'shall we start with the demo'; Speaker 2 says 'ja, prima'. Who are they?"
> **User:** "Speaker 1 is me." — Naming done; the Transcript now shows the user instead of Speaker 1.
