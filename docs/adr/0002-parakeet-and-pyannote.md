# 0002: Parakeet for transcription, pyannote for diarization

## Status

Accepted.

## Context

scribe needs multilingual speech-to-text with word timestamps (Dutch and English, often mixed within one Meeting), plus speaker diarization, running locally on Apple Silicon.

## Decision

- **Transcription:** NVIDIA Parakeet-TDT-0.6B-v3 through [parakeet-mlx](https://github.com/senstella/parakeet-mlx). It covers Dutch and English with word timestamps and runs natively on Apple Silicon.
- **Diarization and embeddings:** [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) on MPS. Its terms must be accepted once on Hugging Face, and a read token is needed for the first download. After that it runs offline. Its Speaker embeddings also feed the Voice profiles.

Rejected:
- Ollama: it serves LLMs only and has no speech-to-text models.
- NVIDIA's streaming Nemotron ASR: built for low latency and documented for Linux.
- NVIDIA Sortformer: English-biased and CUDA-oriented.

whisper.cpp large-v3 remains the fallback if mixed-language accuracy proves weak.

## Consequences

pyannote on CPU is 16-31x slower than on MPS, which misses the run-time budget, so MPS is required (`PYTORCH_ENABLE_MPS_FALLBACK=1` is set). The model dependencies live in an optional `asr` extra, limited to macOS, so lint and tests run on Linux CI without models.
