"""Local settings in `.scribe/config.json` (gitignored), read by `scribe vault-note`.

The path is `--config`, else the `SCRIBE_CONFIG` environment variable, else
`.scribe/config.json` under the current directory. The file is a JSON object with the keys in
`KEYS`; a missing file is an empty config.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path

CONFIG_ENV = "SCRIBE_CONFIG"
DEFAULT_PATH = Path(".scribe") / "config.json"
DEFAULT_FOLDER = "Meetings"
KEYS = ("vault", "folder")


class ConfigError(Exception):
    """A config that cannot be read or changed, worded for the user with a next step."""


def config_path(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    from_env = os.environ.get(CONFIG_ENV)
    return Path(from_env) if from_env else DEFAULT_PATH


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ConfigError(
            f"cannot read {path} as JSON ({error}); fix or delete it, then run again"
        ) from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path} is not a JSON object; fix or delete it, then run again")
    values = {str(k): v for k, v in document.items() if k in KEYS and isinstance(v, str)}
    return values


def check_folder(rel: str) -> None:
    """Refuse a vault folder that is absolute or climbs out of the vault with `..`."""
    if Path(rel).is_absolute() or ".." in Path(rel).parts:
        raise ConfigError(
            f"folder {rel!r} must be relative to the vault, without '..'; "
            "run scribe config set folder <rel> or pass --folder REL"
        )


def set_value(path: Path, key: str, value: str) -> None:
    if key not in KEYS:
        raise ConfigError(f"unknown key {key!r}; use one of: {', '.join(KEYS)}")
    if key == "folder":
        check_folder(value)
    values = load(path)
    values[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8", newline="\n")


def show_lines(values: Mapping[str, str]) -> list[str]:
    lines: list[str] = []
    for key in KEYS:
        if key in values:
            lines.append(f"{key}: {values[key]}")
        elif key == "folder":
            lines.append(f"{key}: {DEFAULT_FOLDER} (default)")
        else:
            lines.append(f"{key}: (not set)")
    return lines
