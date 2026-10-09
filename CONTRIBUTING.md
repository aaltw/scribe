# Contributing

Issues and pull requests are welcome.

## Setup

```bash
uv sync --extra asr                 # macOS; plain `uv sync` is enough for the tests
git config core.hooksPath .githooks # blocks committing recordings, transcripts and profiles
```

## Before you open a pull request

```bash
uv run ruff check .
uv run mypy          # bare: pyproject.toml lists the files, the skill helpers included
uv run pytest
```

- **No real meeting data.** Tests use synthetic audio, invented text and invented names only.
  Never commit anything from `meetings/` or `.scribe/`, and never paste transcript text into an
  issue or PR.
- **Error messages give the next step.** Every refusal says what went wrong and what to run
  next. The `/transcribe` skill relies on this.
- **Skill text is tested.** Tests in `tests/test_skill_*.py` match sentences in
  `skills/transcribe/SKILL.md` after collapsing whitespace. If you change a sentence there,
  update its test.
- **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:` and so on).

Terms such as Recording, Meeting, Speaker, Participant, Naming and Segment are defined in
[CONTEXT.md](CONTEXT.md). Use them as defined there.
