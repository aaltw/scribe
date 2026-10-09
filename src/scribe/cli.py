"""Console entry point for scribe.

Shape: `scribe <recording> [options]` runs the full pipeline. It is the `run` subcommand
written without its name: when the first argument is not a subcommand (`run`, `extract`,
`name`, `stats`, `check-clean`, `vault-note`, `config`, `slack`, `profile`, `suggest`) or `-h`/`--help`, `main` puts `run` in front. A Recording
whose path is literally a subcommand's name needs the explicit form, `scribe run <recording>`.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from scribe import config
from scribe.clean import CLEAN_NAME, check_clean
from scribe.diarize import DiarizeError
from scribe.extract import ExtractError, extract_audio
from scribe.naming import JSON_NAME, NamingError, name_meeting, parse_pairs, resolve_meeting
from scribe.pipeline import STAGES, PipelineError, Result, run
from scribe.profiles import DEFAULT_DIR, ProfileError, format_summary, list_profiles, save_sample
from scribe.slack import MissingInputError, SlackError, slack_meeting
from scribe.stats import StatsError, format_line, stats_meeting
from scribe.suggest import SuggestError, suggest
from scribe.suggest import format_line as format_suggestion
from scribe.transcribe import TranscribeError
from scribe.transcript import TranscriptError, parse_json
from scribe.vault import VaultNoteError, build_note, write_note

SUBCOMMANDS = (
    "run",
    "extract",
    "name",
    "stats",
    "check-clean",
    "vault-note",
    "config",
    "slack",
    "profile",
    "suggest",
)
USAGE = """scribe [-h] <recording> [--speakers N] [--name NAME] [--meetings-dir DIR] [--force]
       scribe {run,extract,name,stats,check-clean,vault-note,config,slack,profile,suggest} ..."""
DESCRIPTION = """Turn a meeting Recording into a Transcript saying who said what.

scribe <recording> runs the full pipeline (extract, diarize, embed, transcribe, merge, write)
into <meetings-dir>/<date>-<name>/ and prints each stage's wall time. It is short for
scribe run <recording>; see scribe run --help for its options."""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribe",
        usage=USAGE,
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="{run,extract,name,stats,check-clean,vault-note,config,slack,profile,suggest}",
    )
    pipeline = sub.add_parser(
        "run",
        prog="scribe [run]",
        help="the full pipeline (the default: scribe <recording> means scribe run <recording>)",
        description="Extract, diarize, embed, transcribe, merge and write one Recording's "
        "Transcript into <meetings-dir>/<date>-<name>/ (audio.wav, transcript.json, "
        "transcript.md, embeddings.json), then print each stage's wall time. A Meeting folder that already has a Transcript is "
        "refused, so a re-run cannot drop its Naming; --force overwrites it.",
    )
    pipeline.add_argument("recording", type=Path, help="video or audio file")
    pipeline.add_argument(
        "--speakers",
        type=positive_int,
        metavar="N",
        help="number of Speakers, as a hint to the diarizer (default: detected)",
    )
    pipeline.add_argument("--name", help="Meeting name (default: slug of the Recording's basename)")
    pipeline.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        metavar="DIR",
        help="where Meeting folders live (default: ./meetings)",
    )
    pipeline.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing Transcript in the Meeting folder, dropping its Naming",
    )
    extract = sub.add_parser("extract", help="extract a Recording's audio to 16 kHz mono WAV")
    extract.add_argument("recording", type=Path, help="video or audio file")
    extract.add_argument("--name", help="Meeting name (default: slug of the Recording's basename)")
    extract.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    name = sub.add_parser(
        "name",
        help="name Speakers in a Meeting's Transcript",
        description="Rewrite transcript.json and transcript.md with Participant names. "
        "Unnamed Speakers keep their label; naming again replaces an earlier name. When "
        "transcript.clean.md exists and matches transcript.json, its headers are renamed too "
        "and its text is kept; when it does not match, it is left as it is with a warning.",
    )
    name.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    name.add_argument("pairs", nargs="+", metavar="LABEL=NAME", help='e.g. "Speaker 1=Alex"')
    name.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    stats = sub.add_parser(
        "stats",
        help="Speaker stats: talk time, share and Segment count",
        description="Write stats.json in the Meeting folder from its transcript.json and print "
        "each Speaker's talk time, share of talk time and Segment count.",
    )
    stats.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    stats.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    check = sub.add_parser(
        "check-clean",
        help="check that transcript.clean.md keeps every Segment of transcript.json",
        description="Compare transcript.clean.md with transcript.json: same Segment count and "
        "order, and each header line (Speaker or Participant, timestamp) exactly as rendered "
        "from the json. Prints 'k of N' and exits 0 on a match. On a mismatch prints one line, "
        "'Segment k of N raw / M clean: <field>', and exits 1. Exits 2 when a file is missing "
        "or unreadable. Never prints Segment text.",
    )
    check.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    check.add_argument(
        "--partial",
        action="store_true",
        help="accept a clean file holding a correct prefix of the Segments (a chunked cleanup)",
    )
    check.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    vault = sub.add_parser(
        "vault-note",
        help="write a Meeting's Summary and Minutes as one note in the Obsidian vault",
        description="Write <vault>/<folder>/<YYYY-MM-DD>-<slug>.md: frontmatter, then summary.md, "
        "then minutes.md. Exits 1 when the note exists (file unchanged), 2 when summary.md, "
        "minutes.md or transcript.json is missing or no vault is configured.",
    )
    vault.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    vault.add_argument("--vault", type=Path, metavar="DIR", help="vault root (else config)")
    vault.add_argument("--folder", metavar="REL", help="folder in the vault (else config)")
    vault.add_argument(
        "--dry-run", action="store_true", help="print the target path and frontmatter only"
    )
    vault.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    add_config_option(vault)
    cfg = sub.add_parser(
        "config",
        help="show or set local settings (vault, folder)",
        description="`config set <key> <value>` writes one setting, `config show` prints them. "
        f"Keys: {', '.join(config.KEYS)}.",
    )
    add_config_option(cfg)
    cfg_sub = cfg.add_subparsers(dest="config_command", required=True)
    cfg_set = cfg_sub.add_parser("set", help="set one key")
    cfg_set.add_argument("key", help=f"one of: {', '.join(config.KEYS)}")
    cfg_set.add_argument("value")
    cfg_sub.add_parser("show", help="print the settings")
    slack = sub.add_parser(
        "slack",
        help="write slack.md, a Slack message for you to paste",
        description="Write slack.md in the Meeting folder from its summary.md and stats.json: a "
        "title line, the Summary's Overview, Decisions and Action items, then Speaker stats, in "
        "Slack mrkdwn. Exits 2 when summary.md or stats.json is missing. Never posts anything.",
    )
    slack.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    slack.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    profile = sub.add_parser(
        "profile",
        help="save and list Voice profiles",
        description="`profile save <meeting> <Participant>` stores that Participant's voice from "
        "a Meeting (a mean of window embeddings, no audio) in <profiles-dir>/<name>.json; "
        "`profile list` prints each profile's name, sample count and source Meetings, never "
        "vectors. Exits 2 on a refusal, with a next step.",
    )
    profile_sub = profile.add_subparsers(dest="profile_command", required=True)
    save = profile_sub.add_parser("save", help="save a Participant's voice from a Meeting")
    save.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    save.add_argument("participant", help="a Participant named in that Meeting")
    save.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    add_profiles_dir_option(save)
    list_ = profile_sub.add_parser("list", help="list profiles: name, samples, source Meetings")
    add_profiles_dir_option(list_)
    suggest_ = sub.add_parser(
        "suggest",
        help="suggest a Participant for each Speaker from the saved Voice profiles",
        description="Compare each Speaker's centroid in embeddings.json with the Voice profiles "
        "and write suggestions.json in the Meeting folder: a one-to-one assignment by total "
        "cosine, only at or above the threshold. Prints one line per Speaker, never vectors, and "
        "never writes a Transcript. Exits 2 when embeddings.json is missing or no profile comes "
        "from the Meeting's model; no profiles at all is exit 0 with no suggestions.",
    )
    suggest_.add_argument(
        "meeting",
        help="path to the Meeting folder, or a Meeting folder name under --meetings-dir",
    )
    suggest_.add_argument(
        "--meetings-dir",
        type=Path,
        default=Path("meetings"),
        help="where Meeting folders live (default: ./meetings)",
    )
    add_profiles_dir_option(suggest_)
    return parser


def add_profiles_dir_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=DEFAULT_DIR,
        metavar="DIR",
        help=f"where Voice profiles live (default: {DEFAULT_DIR})",
    )


def add_config_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        metavar="FILE",
        help=f"config file (default: ${config.CONFIG_ENV}, else {config.DEFAULT_PATH})",
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in (*SUBCOMMANDS, "-h", "--help"):
        arguments.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    if args.command == "run":
        try:
            result = run(
                args.recording, args.meetings_dir, args.name, args.speakers, force=args.force
            )
        except (PipelineError, ExtractError, DiarizeError, TranscribeError) as err:
            print(f"scribe: error: {err}", file=sys.stderr)
            return 1
        print_summary(result)
        print(result.folder)
    elif args.command == "extract":
        try:
            output = extract_audio(args.recording, args.meetings_dir, args.name)
        except ExtractError as err:
            print(f"scribe: error: {err}", file=sys.stderr)
            return 1
        print(output)
    elif args.command == "name":
        try:
            names = parse_pairs(args.pairs)
            named = name_meeting(resolve_meeting(args.meeting, args.meetings_dir), names)
        except NamingError as err:
            print(f"scribe: error: {err}", file=sys.stderr)
            return 1
        if named.clean_skipped:
            print(
                f"scribe: warning: {CLEAN_NAME} does not match {JSON_NAME}, so its headers "
                f"were not renamed; run scribe check-clean {args.meeting} to see why",
                file=sys.stderr,
            )
    elif args.command == "stats":
        try:
            folder = resolve_meeting(args.meeting, args.meetings_dir)
            for line in map(format_line, stats_meeting(folder)):
                print(line)
        except (NamingError, StatsError) as err:
            print(f"scribe: error: {err}", file=sys.stderr)
            return 1
    elif args.command == "profile":
        return run_profile(args)
    elif args.command == "suggest":
        return run_suggest(args)
    elif args.command == "slack":
        return run_slack(args.meeting, args.meetings_dir)
    elif args.command == "check-clean":
        return run_check_clean(args.meeting, args.meetings_dir, partial=args.partial)
    elif args.command == "vault-note":
        return run_vault_note(args)
    elif args.command == "config":
        return run_config(args)
    return 0


def run_vault_note(args: argparse.Namespace) -> int:
    """`scribe vault-note`: 0 written, 1 note exists, 2 missing input or no vault."""
    try:
        settings = config.load(config.config_path(args.config))
        vault_text = str(args.vault) if args.vault is not None else settings.get("vault")
        if not vault_text:
            raise VaultNoteError(
                "no vault configured; run scribe config set vault <path> or pass --vault DIR"
            )
        vault = Path(vault_text).expanduser()
        if not vault.is_dir():
            raise VaultNoteError(
                f"vault {vault} is not a directory; run scribe config set vault <path> "
                "or pass --vault DIR"
            )
        rel = args.folder or settings.get("folder") or config.DEFAULT_FOLDER
        config.check_folder(rel)
        note = build_note(resolve_meeting(args.meeting, args.meetings_dir), vault, rel)
        if args.dry_run:
            print(note.path)
            print(note.frontmatter)
            return 0
        write_note(note)
    except config.ConfigError as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    except NamingError as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    except VaultNoteError as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return err.code
    print(note.path)
    return 0


def run_profile(args: argparse.Namespace) -> int:
    """`scribe profile save|list`: 0 on success, 2 on a refusal. Never prints a vector."""
    try:
        if args.profile_command == "save":
            try:
                folder = resolve_meeting(args.meeting, args.meetings_dir)
            except NamingError as err:
                raise ProfileError(
                    f"{err}; pass the path to a Meeting folder or fix --meetings-dir"
                ) from err
            path, count = save_sample(folder, args.participant, args.profiles_dir)
            print(f"{path} ({count} sample(s))")
        else:
            for summary in list_profiles(args.profiles_dir):
                print(format_summary(summary))
    except (NamingError, ProfileError) as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    return 0


def run_suggest(args: argparse.Namespace) -> int:
    """`scribe suggest`: 0 on success (also with no profiles), 2 on a refusal. Never prints a
    vector."""
    try:
        try:
            folder = resolve_meeting(args.meeting, args.meetings_dir)
        except NamingError as err:
            raise SuggestError(
                f"{err}; pass the path to a Meeting folder or fix --meetings-dir"
            ) from err
        suggestions, skipped = suggest(folder, args.profiles_dir)
    except (ProfileError, SuggestError) as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    if skipped:
        print(
            f"scribe: note: {skipped} profile(s) skipped (another model or no samples)",
            file=sys.stderr,
        )
    for suggestion in suggestions:
        print(format_suggestion(suggestion))
    return 0


def run_config(args: argparse.Namespace) -> int:
    path = config.config_path(args.config)
    try:
        if args.config_command == "set":
            config.set_value(path, args.key, args.value)
        else:
            for line in config.show_lines(config.load(path)):
                print(line)
    except config.ConfigError as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    return 0


def run_slack(meeting: str, meetings_dir: Path) -> int:
    """`scribe slack`: 0 on success, 2 when summary.md or stats.json is missing, else 1."""
    try:
        print(slack_meeting(resolve_meeting(meeting, meetings_dir)))
    except MissingInputError as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 2
    except (NamingError, SlackError) as err:
        print(f"scribe: error: {err}", file=sys.stderr)
        return 1
    return 0


def run_check_clean(meeting: str, meetings_dir: Path, *, partial: bool) -> int:
    """`scribe check-clean`: 0 on a match, 1 on a mismatch, 2 when it cannot check.

    Messages name files and counts only, never Segment text or the Meeting's name.
    """
    try:
        folder = resolve_meeting(meeting, meetings_dir)
    except NamingError:
        print("scribe: error: no such Meeting folder", file=sys.stderr)
        return 2
    try:
        transcript = parse_json((folder / JSON_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, TranscriptError):
        print(f"scribe: error: cannot read {JSON_NAME} as a Transcript", file=sys.stderr)
        return 2
    clean_path = folder / CLEAN_NAME
    if not clean_path.is_file():
        print(f"scribe: error: no {CLEAN_NAME} in the Meeting folder", file=sys.stderr)
        return 2
    try:
        clean_text = clean_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        print(f"scribe: error: cannot read {CLEAN_NAME} as UTF-8 text", file=sys.stderr)
        return 2
    result = check_clean(transcript, clean_text, partial=partial)
    if not result.ok:
        print(result.mismatch_line())
        return 1
    if result.ratio_outside():
        print(
            f"scribe: warning: clean/raw text length ratio {result.ratio:.2f} is outside 0.5-2.0",
            file=sys.stderr,
        )
    print(f"{result.clean} of {result.raw}")
    return 0


def print_summary(result: Result) -> None:
    """Counts and per-stage wall times on stderr; stdout keeps only the Meeting folder."""
    transcript = result.transcript
    print(
        f"scribe: {result.detected_speakers} Speakers detected, "
        f"{len(transcript.speakers)} in the Transcript, {len(transcript.segments)} Segments",
        file=sys.stderr,
    )
    for stage in STAGES:
        print(f"  {stage:<10} {format_duration(result.timings[stage])}", file=sys.stderr)
    print(f"  {'total':<10} {format_duration(result.total)}", file=sys.stderr)


def format_duration(seconds: float) -> str:
    """`8.3 s` below a minute, else `12m05.3s (725.3 s)`."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m{rest:04.1f}s ({seconds:.1f} s)"
