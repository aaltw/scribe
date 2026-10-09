"""File plumbing for /transcribe's Summary and Minutes step: remove, count lines, check headings.

Usage: uv run --extra asr python skills/transcribe/outputs.py <command> <meeting> ...

- `delete <meeting>`: remove summary.md and minutes.md if present, so the Write tool writes
  them fresh.
- `lines <meeting> <file>`: print `<file>: N lines`, so Claude can Read the file in ranges.
- `check <meeting>`: check summary.md and minutes.md against summary.template.md and
  minutes.template.md beside this file. The templates' heading lines are the contract `scribe vault-note` and `scribe slack` parse:
  - The output's heading lines (lines starting with `#`) are the template's, in order, byte for
    byte. A template heading with a `<placeholder>` matches any heading with the same text before
    the placeholder and some text after it. From the first such heading to the end, the
    template's headings form a block that repeats one or more times (minutes.md: one block per
    topic).
  - The first non-blank line after the first heading starts with `Source: `.
  - Every heading has some text before the next heading, and no template placeholder line is
    left as is.
  - In summary.md, every line under `## Action items` is `- None.` or ends in
    `(owner: <name>)`; an action item without a clear owner reads `(owner: unclear)`.

Exit 0 on success (`check` prints `summary.md and minutes.md follow the templates`), 1 with one
line per problem on stderr otherwise. Messages name files, line numbers and template headings
only, never Meeting text. Output goes to this Claude Code session only; never paste it into a PR,
commit or report (ADR 0001).
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUTS = {"summary.md": HERE / "summary.template.md", "minutes.md": HERE / "minutes.template.md"}
PLACEHOLDER = re.compile(r"<[^<>]+>")
OWNER = re.compile(r"^- \S.* \(owner: [^()]*\S[^()]*\)$")
ACTION_ITEMS = "## Action items"
NONE = "- None."


class OutputsError(Exception):
    """A file step that failed, for a reason worth showing the user as-is."""


def delete(meeting: Path) -> None:
    for name in OUTPUTS:
        try:
            (meeting / name).unlink(missing_ok=True)
        except OSError as error:
            raise OutputsError(f"cannot remove {name}: {type(error).__name__}") from error


def lines(meeting: Path, name: str) -> None:
    try:
        text = (meeting / name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise OutputsError(f"cannot read {name}: {type(error).__name__}") from error
    print(f"{name}: {len(text.splitlines())} lines")


def _headings(text: str) -> list[tuple[int, str]]:
    return [(n, line) for n, line in enumerate(text.splitlines(), 1) if line.startswith("#")]


def _level(heading: str) -> int:
    return len(heading) - len(heading.lstrip("#"))


def _matches(template: str, heading: str) -> bool:
    placeholder = PLACEHOLDER.search(template)
    if placeholder is None:
        return heading == template
    prefix = template[: placeholder.start()]
    return heading.startswith(prefix) and heading[len(prefix) :].strip() != ""


def problems(name: str, text: str, template: str) -> list[str]:
    """Every way `text` (the written file) breaks `template`'s contract, one line each."""
    found: list[str] = []
    expected = [line for _, line in _headings(template)]
    repeat = next((i for i, h in enumerate(expected) if PLACEHOLDER.search(h)), len(expected))
    written = _headings(text)
    fixed, block = expected[:repeat], expected[repeat:]
    for i, (number, heading) in enumerate(written):
        if i < len(fixed):
            want = fixed[i]
        elif block:
            want = block[(i - len(fixed)) % len(block)]
        else:
            found.append(f"{name} line {number}: a heading the template does not have")
            continue
        if not _matches(want, heading):
            found.append(f"{name} line {number}: expected heading {want!r}")
    extra = len(written) - len(fixed)
    if extra < len(block) or (block and extra % len(block)):
        found.append(f"{name}: headings missing at the end, expected {expected!r} in order")

    body = text.splitlines()
    first = written[0][0] if written else 0
    after = next((line for line in body[first:] if line.strip()), "")
    if not after.startswith("Source: "):
        found.append(f"{name} line {first + 1}: expected a 'Source: ' line after the title")
    bounds = [n for n, _ in written] + [len(body) + 1]
    levels = [_level(h) for _, h in written] + [0]
    for i, ((number, heading), end) in enumerate(zip(written, bounds[1:], strict=True)):
        section = [line for line in body[number : end - 1] if line.strip()]
        # A heading followed straight by a deeper one (## Topic, then ### Discussed) needs no text.
        if not section and levels[i + 1] <= levels[i]:
            found.append(f"{name} line {number}: no text under this heading")
        if heading == ACTION_ITEMS:
            for offset, line in enumerate(body[number : end - 1], number + 1):
                if line.strip() and line != NONE and not OWNER.match(line):
                    found.append(
                        f"{name} line {offset}: action item must end in '(owner: <name>)' "
                        "or '(owner: unclear)', or be '- None.'"
                    )
    leftovers = {line for line in template.splitlines() if PLACEHOLDER.search(line)}
    for number, line in enumerate(body, 1):
        if line in leftovers:
            found.append(f"{name} line {number}: template placeholder left as is")
    return found


def check(meeting: Path) -> None:
    found: list[str] = []
    for name, template_path in OUTPUTS.items():
        try:
            template = template_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise OutputsError(
                f"cannot read {template_path.name}: {type(error).__name__}"
            ) from error
        try:
            text = (meeting / name).read_text(encoding="utf-8")
        except FileNotFoundError:
            found.append(f"{name}: not written")
            continue
        except (OSError, UnicodeDecodeError) as error:
            raise OutputsError(f"cannot read {name}: {type(error).__name__}") from error
        found.extend(problems(name, text, template))
    if found:
        raise OutputsError("\n".join(found))
    print("summary.md and minutes.md follow the templates")


def main() -> int:
    parser = argparse.ArgumentParser(prog="outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("delete", "check"):
        sub.add_parser(command).add_argument("meeting", type=Path)
    count = sub.add_parser("lines")
    count.add_argument("meeting", type=Path)
    count.add_argument("file")
    args = parser.parse_args()
    meeting: Path = args.meeting
    try:
        if not meeting.is_dir():
            raise OutputsError("no such Meeting folder")
        if args.command == "delete":
            delete(meeting)
        elif args.command == "lines":
            lines(meeting, args.file)
        else:
            check(meeting)
    except OutputsError as error:
        for line in str(error).splitlines():
            print(f"outputs: error: {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
