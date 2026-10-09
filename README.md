# scribe

Local, private meeting transcription for macOS. scribe turns a meeting recording into a
transcript that says **who said what**, and runs entirely on your Mac: no cloud speech service
and no telemetry.

- **Speaker diarization** with [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) on Apple Silicon (MPS)
- **Multilingual transcription** with NVIDIA [Parakeet-TDT-0.6B-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) through [parakeet-mlx](https://github.com/senstella/parakeet-mlx), with word timestamps. Mixed Dutch and English works.
- **Naming**: replace `Speaker 1` with a real name in one command
- **Voice profiles**: save a person's voice once, and scribe suggests their name in later meetings. You always confirm the suggestion.
- **Speaker stats**: talk time, share and turn count per person
- **Optional [Claude Code](https://claude.com/claude-code) skill** (`/transcribe`): runs the whole flow in a conversation, then writes a cleaned-up transcript, a summary with action items, and minutes

## Requirements

- macOS on Apple Silicon (an M-series chip; the models run on MPS)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- [ffmpeg](https://ffmpeg.org/): `brew install ffmpeg`
- A Hugging Face account. Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1), then log in once with `hf auth login`. After the first download, runs work offline.

## Install

```bash
git clone https://github.com/aaltw/scribe.git
cd scribe
uv sync --extra asr
```

The `asr` extra holds the model dependencies (torch, pyannote.audio, parakeet-mlx). Without it,
the non-model commands (naming, stats, notes) still work.

## Usage

```bash
# Transcribe a recording (video or audio). Output goes to ./meetings/<date>-<name>/,
# the date taken from the file name if it starts with one, else from its modified time
uv run scribe ~/Movies/standup.mp4 --speakers 2 --name standup

# Name the speakers (a Meeting is its folder name under ./meetings, or a path)
uv run scribe name 2026-01-05-standup "Speaker 1=Alex" "Speaker 2=Sam"

# Talk time per speaker
uv run scribe stats 2026-01-05-standup

# Save Alex's voice, so later meetings suggest the name
uv run scribe profile save 2026-01-05-standup Alex
uv run scribe suggest 2026-01-12-standup
```

Each meeting folder holds:

| File | Contents |
|---|---|
| `audio.wav` | 16 kHz mono audio extracted from the recording |
| `transcript.md` | Segments in order: `**Speaker** [mm:ss]` followed by the text |
| `transcript.json` | The same, with start/end seconds, labels, names and model versions |
| `embeddings.json` | Per-speaker voice embeddings, used by Voice profiles |
| `stats.json` | Talk time, share and segment count per speaker |

All commands:

| Command | What it does |
|---|---|
| `scribe <recording>` | Full pipeline: extract, diarize, embed, transcribe, merge, write |
| `scribe extract <recording>` | Audio only |
| `scribe name <meeting> "Speaker N=Name" ...` | Name speakers (also updates a clean transcript) |
| `scribe stats <meeting>` | Speaker stats |
| `scribe profile save <meeting> <name>` / `profile list` | Manage Voice profiles in `./.scribe/profiles/` |
| `scribe suggest <meeting>` | Suggest a name per speaker from the saved profiles |
| `scribe check-clean <meeting>` | Check that a cleaned transcript still matches the raw one |
| `scribe vault-note <meeting>` | Write the summary and minutes as an Obsidian note |
| `scribe slack <meeting>` | Write `slack.md`, a paste-ready summary with speaker stats |
| `scribe config set\|show` | Local settings (vault path and folder) |

Run `uv run scribe <command> --help` for the options.

## Claude Code skill

The `/transcribe` skill walks through the full flow in a Claude Code session:
1. runs the pipeline and shows a few lines per speaker;
2. suggests names from your Voice profiles and asks who is who;
3. records the naming and shows speaker stats;
4. optionally writes a clean transcript, a summary and minutes;
5. offers a vault note and a Slack-ready post.

- **Inside a clone:** open Claude Code in the repository, and `/transcribe <recording>` is available as a project skill.
- **Anywhere else:** install it as a plugin:

  ```
  /plugin marketplace add aaltw/scribe
  /plugin install scribe@scribe
  ```

  The skill runs scribe from the plugin's own copy of this repository, and writes `meetings/` and `.scribe/` in your current directory.

## Privacy

- **Local pipeline.** The pipeline sends nothing off the machine. Model telemetry (pyannote, Hugging Face Hub) is switched off in code, and a test checks that a run makes no network connections. See [ADR 0001](docs/adr/0001-local-only-processing.md).
- **The Claude Code skill.** With `/transcribe`, transcript lines are shown in your Claude Code session, so they reach Anthropic under your Claude account. The CLI on its own does not.
- **Git guards.** Recordings, transcripts, embeddings and Voice profiles are biometric or personal data. `meetings/` and `.scribe/` are gitignored, and a pre-commit hook and a CI check refuse them. Enable the hook with `git config core.hooksPath .githooks`.

## How it works

The pipeline runs in order:
1. **Extract:** ffmpeg pulls the audio out to 16 kHz mono WAV.
2. **Diarize:** pyannote finds the speakers and their speaking turns.
3. **Embed:** one voice embedding per speaker.
4. **Transcribe:** Parakeet produces timed words.
5. **Merge:** each word goes to the speaker whose turn contains the word's midpoint; consecutive words of one speaker form a segment.
6. **Write:** Markdown and JSON.

On an M2 Pro, a 30-minute recording takes a few minutes on an otherwise idle machine. The terms
used throughout are defined in [CONTEXT.md](CONTEXT.md), and design decisions are in
[docs/adr/](docs/adr/).

## Development

```bash
uv sync --extra asr        # or plain `uv sync` on Linux: tests run without the models
uv run ruff check .
uv run mypy
uv run pytest
```

Tests use synthetic audio and invented text only. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE). The models have their own licenses: Parakeet-TDT-0.6B-v3 is CC-BY-4.0, and
pyannote community-1 requires accepting its terms on Hugging Face.
