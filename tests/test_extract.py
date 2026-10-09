"""Extract tests use synthetic audio generated with ffmpeg in a tmp dir, never a real Recording."""

import os
import subprocess
from pathlib import Path

import pytest

from scribe.cli import main
from scribe.extract import meeting_folder_name


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def probe(path: Path, entry: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", entry,
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )  # fmt: skip
    return out.stdout.strip()


def test_extract_writes_16khz_mono_wav_of_same_duration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "2026-10-01 11-19-53.mp4"
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
           "-ac", "2", "-c:a", "aac", str(recording))  # fmt: skip
    meetings = tmp_path / "meetings"

    assert main(["extract", str(recording), "--meetings-dir", str(meetings)]) == 0

    wav = meetings / "2026-10-01-11-19-53" / "audio.wav"
    assert capsys.readouterr().out.strip() == str(wav)
    assert probe(wav, "stream=sample_rate") == "16000"
    assert probe(wav, "stream=channels") == "1"
    assert probe(wav, "stream=codec_name") == "pcm_s16le"
    assert float(probe(wav, "stream=duration")) == pytest.approx(2.0, abs=0.2)


def test_name_flag_overrides_default_name(tmp_path: Path) -> None:
    recording = tmp_path / "2026-10-01 standup.wav"
    ffmpeg("-f", "lavfi", "-i", "sine=duration=1", str(recording))
    meetings = tmp_path / "meetings"

    main(["extract", str(recording), "--name", "Team Sync", "--meetings-dir", str(meetings)])

    assert (meetings / "2026-10-01-team-sync" / "audio.wav").is_file()


def test_folder_name_from_dated_basename(tmp_path: Path) -> None:
    recording = tmp_path / "2026-10-01 11-19-53.mp4"
    recording.touch()
    assert meeting_folder_name(recording) == "2026-10-01-11-19-53"

    named = tmp_path / "2026-10-01 Weekly Sync.mp4"
    named.touch()
    assert meeting_folder_name(named) == "2026-10-01-weekly-sync"


def test_folder_name_from_mtime_for_undated_basename(tmp_path: Path) -> None:
    recording = tmp_path / "Weekly Sync (final).mp4"
    recording.touch()
    # 2026-03-04 12:00 UTC is 2026-03-04 in every timezone within a half-day of UTC
    stamp = 1772625600
    os.utime(recording, (stamp, stamp))
    assert meeting_folder_name(recording) == "2026-03-04-weekly-sync-final"


def test_no_audio_stream_fails_with_readable_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "silent.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-an", str(recording))
    meetings = tmp_path / "meetings"

    assert main(["extract", str(recording), "--meetings-dir", str(meetings)]) != 0

    assert "no audio stream" in capsys.readouterr().err
    assert not meetings.exists()


def test_missing_ffmpeg_fails_with_readable_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert main(["extract", str(tmp_path / "x.mp4")]) != 0
    assert "ffmpeg not found" in capsys.readouterr().err
