# 0001: Recordings are processed only on this machine

## Status

Accepted.

## Context

Meeting Recordings hold other people's voices and words, which is personal data. Voice profiles and Speaker embeddings identify a person by voice, which makes them biometric data.

## Decision

The pipeline (extract, diarize, embed, transcribe, merge, write) runs entirely on the local machine. Model weights are downloaded once from Hugging Face; after that, runs work offline.

- **No telemetry.** pyannote.audio 4 enables telemetry by default (file duration, requested speaker counts, model init events). scribe disables it, along with Hugging Face Hub telemetry, in code before the models are imported. A socket-guarded test asserts that a synthetic run makes zero outbound connections.
- **One accepted recipient: Claude Code.** The optional `/transcribe` skill shows Transcript lines inside a Claude Code session and has Claude write the Clean Transcript, Summary and Minutes. That text therefore reaches Anthropic, under the account the user runs Claude Code with. The `scribe` CLI itself sends nothing anywhere.
- **User-chosen destinations.** `scribe vault-note` writes a note into a local Obsidian vault, and only after an explicit yes for each Meeting. `scribe slack` writes `slack.md` for the user to paste; scribe never posts anything.
- **Never in git.** `meetings/` (audio, Transcripts, embeddings) and `.scribe/` (config, Voice profiles) are gitignored. A pre-commit hook and a CI step refuse those paths, audio/video extensions and `.npy`/`.npz` files.

## Consequences

Cloud ASR services (OpenAI, Azure Speech, AssemblyAI, Deepgram) would be quicker to integrate and often more accurate. They are not used. Runs are bound to a machine strong enough for local models; an Apple Silicon Mac with MPS is the supported target.
