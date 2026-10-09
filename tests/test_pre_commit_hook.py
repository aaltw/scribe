"""The pre-commit hook refuses personal data. Synthetic files only, in a temp repo."""

import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / ".githooks" / "pre-commit"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "commit.gpgsign", "false")
    hooks = tmp_path / ".githooks"
    hooks.mkdir()
    shutil.copy(HOOK, hooks / "pre-commit")
    git(tmp_path, "config", "core.hooksPath", ".githooks")
    return tmp_path


def commit(repo: Path, relpath: str) -> subprocess.CompletedProcess[str]:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic")
    git(repo, "add", "-f", relpath)
    return git(repo, "commit", "-q", "-m", "test: add file")


@pytest.mark.parametrize(
    "relpath",
    [
        "sample.wav",
        "SAMPLE.WAV",
        "a/b.mp3",
        "c.m4a",
        "d.flac",
        "e.ogg",
        "f.mp4",
        "g.mov",
        "h.mkv",
        "meetings/x.md",
        "meetings/2026-10-01-demo/transcript.json",
        ".scribe/profiles/demo.json",
        ".scribe/embeddings.bin",
        "emb.npy",
        "EMB.NPZ",
        "a/b/emb.npz",
    ],
)
def test_refuses_personal_data(repo: Path, relpath: str) -> None:
    result = commit(repo, relpath)
    assert result.returncode != 0
    assert relpath in result.stderr


def test_allows_lookalike_paths(repo: Path) -> None:
    result = commit(repo, "docs/scribe/notes.md")
    assert result.returncode == 0, result.stderr


def test_allows_normal_file(repo: Path) -> None:
    result = commit(repo, "notes.md")
    assert result.returncode == 0, result.stderr
    assert "notes.md" in git(repo, "ls-files").stdout
