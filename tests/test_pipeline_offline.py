"""`scribe <recording>` with the real models and no network. macOS with the asr extra only.

The pipeline runs in a child interpreter where every IP socket connect raises, started without
HF_HUB_OFFLINE and with both telemetry switches preset to on, so the run itself must go offline
and turn telemetry off. Needs both pinned snapshots in the Hugging Face cache. Run alone, one model job at
a time:

    uv run pytest tests/test_pipeline_offline.py -v
"""

import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from scribe.pipeline import models_cached

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin"
    or find_spec("parakeet_mlx") is None
    or find_spec("pyannote") is None
    or shutil.which("say") is None
    or shutil.which("ffmpeg") is None
    or not models_cached(os.environ),
    reason="needs macOS, the asr extra, say, ffmpeg and both models cached",
)

# Invented lines, one per voice, with a pause between them.
LINES = [
    ("Samantha", "Good morning, shall we look at the pineapple roadmap today?"),
    ("Daniel", "Yes, the budget for the harbour project is ready."),
]

NO_NETWORK = """
import atexit
import socket
import sys

attempts = []
connect = socket.socket.connect


def guarded(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        attempts.append(repr(address))
        raise OSError(f"network blocked by test: {address!r}")
    return connect(self, address)


socket.socket.connect = guarded
socket.socket.connect_ex = lambda self, address: guarded(self, address) or 0
# Registered first, so it runs last: after any exporter flushes at exit.
atexit.register(lambda: print(f"attempts={attempts}", file=sys.stderr))

from scribe.cli import main

sys.exit(main(sys.argv[1:]))
"""


@pytest.fixture
def recording(tmp_path: Path) -> Path:
    parts = []
    for i, (voice, text) in enumerate(LINES):
        part = tmp_path / f"{i}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(part), text], check=True)
        parts.append(part)
    output = tmp_path / "2026-10-02 offline check.m4a"
    graph = "[0:a]apad=pad_dur=1.0[a];[a][1:a]concat=n=2:v=0:a=1"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(parts[0]), "-i", str(parts[1]), "-filter_complex", graph,
         "-c:a", "aac", str(output)],
        check=True,
    )  # fmt: skip
    return output


def test_pipeline_runs_without_network(tmp_path: Path, recording: Path) -> None:
    env = dict(os.environ)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        env.pop(key, None)
    env.update(PYANNOTE_METRICS_ENABLED="true", HF_HUB_DISABLE_TELEMETRY="0")
    meetings = tmp_path / "meetings"

    result = subprocess.run(
        [sys.executable, "-c", NO_NETWORK, str(recording), "--speakers", "2",
         "--meetings-dir", str(meetings)],
        capture_output=True, text=True, check=False, env=env,
    )  # fmt: skip

    assert result.returncode == 0, result.stderr
    assert "attempts=[]" in result.stderr
    assert "torch was imported before" not in result.stderr
    folder = meetings / "2026-10-02-offline-check"
    assert result.stdout.strip() == str(folder)
    document = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    assert document["speakers"] == ["Speaker 1", "Speaker 2"]
    assert "pineapple" in " ".join(s["text"] for s in document["segments"]).lower()


def test_guard_blocks_ip_connects() -> None:
    # The guard itself works: an IP connect in the child fails and is recorded.
    script = NO_NETWORK.split("from scribe.cli")[0] + (
        "\ntry:\n    socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
        "except OSError as e:\n    print(e)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert "network blocked by test" in result.stdout
    assert "attempts=[]" not in result.stderr
    assert "127.0.0.1" in result.stderr
