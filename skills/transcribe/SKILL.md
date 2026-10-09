---
name: transcribe
description: Turn a meeting Recording into a Transcript that says who said what, do the Naming with the user, show Speaker stats, on request write a Clean Transcript, and write the Summary and Minutes. Use when the user runs /transcribe <recording>, or asks to transcribe, run scribe on, or name the Speakers of a Recording or Meeting.
argument-hint: <recording> [--name "<meeting name>"]
---

# /transcribe <recording> [--name "<meeting name>"]

Runs scribe's pipeline on one Recording, shows the user a few lines per Speaker, asks who each
Speaker is, records the Naming, shows Speaker stats, on request writes a Clean Transcript, and
writes the Summary and Minutes, offers the Summary and Minutes as a vault note and writes
slack.md. Terms as in `CONTEXT.md`: Recording, Meeting, Speaker,
Participant, Naming, Transcript, Segment, Clean Transcript, Summary, Minutes, Speaker stats.

## Where scribe runs

`<scribe>` below is the scribe checkout: the directory two levels above this skill's base
directory (the one holding `pyproject.toml`). In every command in this file, `uv run` means
`uv run --project "<scribe>"`, and a path starting with `skills/transcribe/` is relative to
`<scribe>`. Run every command from the user's current working directory: Meeting folders go to
`./meetings/` and local config and Voice profiles to `./.scribe/` there. The first run builds
the environment with the `asr` extra, which downloads the models once.

## Privacy (ADR 0001)

Transcript lines may be shown in this Claude Code session; that is the one accepted recipient.
Never write Transcript text, quotes or Participant names anywhere else: no file in git, PR,
issue, commit message or message to another session. Run the pipeline
only through `uv run --extra asr scribe`; never call the adapters or models directly, because only that path
runs offline with telemetry off.

embeddings.json, suggestions.json and the Voice profiles in `.scribe/profiles/` are biometric
data (ADR 0001). Never Read or print embeddings.json or a profile file, because they hold
vectors; suggestions.json holds only names and cosines, so step 5 may Read it.

## Steps

Ask every question below as plain chat text and end your turn to wait for the answer; do not
use a question tool. Run only the `uv run --extra asr` commands named here.

Read transcript.md and transcript.clean.md only with the Read tool, never through Bash output
(`cat`, `sed`, a script that prints them): a shell output filter may cut long output,
so Claude would work from a Transcript with lines missing.

1. **Recording.** Take the path from the argument; if none, ask for it. Do not check it with
   other commands: `uv run --extra asr scribe` reports a missing or unreadable Recording itself.
   If the argument also has `--name "<name>"`, pass it on to step 3 as is.

2. **Speaker hint.** Ask exactly: "How many Speakers? (Enter = detect)". If the user gives a number
   N, pass `--speakers N`; on Enter, empty, "detect" or anything not a number, pass nothing.
   Detection alone has found 3 Speakers in a 2-person Meeting and 1 in a two-voice clip, so
   the hint matters.

3. **Run.** From the current working directory:

   ```bash
   uv run --extra asr scribe "<recording>" [--speakers N] [--name "<name>"]
   ```

   `--name` only when step 1 had one. It takes minutes for a long Recording (about 2.5 min for
   30 min of audio); use a long Bash timeout (600000 ms). stdout is only the Meeting folder
   path; stderr carries the Speaker and Segment counts and the stage timings. Tell the user the
   counts in one line.

   If it exits 1 with `already has transcript...; pass --force`, the Meeting folder holds an
   earlier Transcript. Tell the user: "This Meeting already has a Transcript. Choose one:
   (1) run under another Meeting name, in a new Meeting folder; nothing is dropped;
   (2) re-run with --force, which overwrites it and drops its Naming;
   (3) redo just the Naming on the existing Transcript."

   - (1): ask for the Meeting name, then run
     `uv run --extra asr scribe "<recording>" [--speakers N] --name "<name>"`.
   - (2): only on a clear yes to overwriting, run the step 3 command with `--force` added.
   - (3): go to step 4 with the folder that starts the error line (`<folder> already has ...`).

   **Repeat command.** From here on, "the repeat command" is `/transcribe "<recording>"` with
   the Recording path of this run, plus `--name "<name>"` when this run passed `--name` (from
   the argument or from choice (1)). Running it and choosing (3) reaches this same Meeting
   folder again (its date comes from the Recording, not from today); every "try again later"
   below names it in full, with the path and name filled in, followed by "(Meeting folder
   <meeting>)", so the user can repeat it exactly.

   Anything else, stop. Any other error: show its `scribe: error:` line and tell the user: "Check
   the Recording or file that line names, then run /transcribe again." Then stop.

   Every command here passes `--extra asr`, so it installs the extra when a plain `uv sync`
   removed it. Never run a plain `uv sync` in this checkout.

4. **Show lines.** With the Meeting folder from stdout:

   ```bash
   uv run --extra asr python skills/transcribe/sample_lines.py "<meeting>"
   ```

   It prints each Speaker's Segment count, speaking time and its 3 longest Segments with
   timestamps. Show that to the user as is. If it exits 1 (`sample_lines: error:`, transcript.json
   unreadable), show the line and tell the user: "The Transcript in this Meeting folder cannot be
   read. Run <the repeat command> and choose (2) to rebuild it with --force." Then stop.

5. **Naming.** First, on every path into this step, look for suggestions from the saved Voice
   profiles:

   ```bash
   uv run --extra asr scribe suggest "<meeting>"
   ```

   Only when it exits 0, Read `<meeting>/suggestions.json` with the Read tool (it holds names
   and cosines, no vectors). A Speaker has a suggestion only when its entry's `participant` is
   not null; decide on `participant` alone, never on `cosine`, because a Speaker with no
   suggestion still carries its best cosine. A Speaker label from step 4 that is missing from
   the file has no suggestion. When suggest exits non-zero, no Speaker has a suggestion, even
   if an older suggestions.json is still in the folder: do not Read it.

   When suggest ends with no suggestion for any Speaker, tell the user the one sentence below for
   its outcome, then ask the Naming question as usual:

   - Exit 0 and no Speaker has a suggestion: "No saved Voice profile matched these Speakers
     (or none is saved yet), so there are no suggestions; you can save Voice profiles after
     the Naming."
   - Exit 2, `no embeddings.json in this Meeting folder`, or `cannot read embeddings.json`:
     "This Meeting has no usable voice embeddings (an older Meeting), so there are no
     suggestions; to get them later, run <the repeat command> and choose (2), which rebuilds it
     with --force and drops the Naming."
   - Exit 2, `none of the ... saved Voice profile(s) comes from this Meeting's embedding model`:
     "The saved Voice profiles come from another embedding model, so scribe cannot compare them
     and there are no suggestions; to get suggestions again, move the files in
     .scribe/profiles/ away and save new profiles after this Naming."
   - Exit 2, `cannot read ... as a Voice profile`: "A Voice profile file is broken, so there
     are no suggestions; move the file the error line names out of .scribe/profiles/ and save
     that Participant again after the Naming."
   - Exit 2, `more than the 16 scribe suggest can assign exactly`: "There are too many Speakers
     and profiles to match exactly, so there are no suggestions; move the profiles you no
     longer need out of .scribe/profiles/."
   - Exit 2, `no Meeting folder`: "scribe suggest could not find the Meeting folder, so there
     are no suggestions; if step 6 cannot find it either, run <the repeat command> again."
   - Exit 2, `cannot write`: "suggestions.json could not be written, so there are no
     suggestions; check that the Meeting folder is writable."
   - Any other exit: show its last stderr line and say "scribe suggest did not run, so there
     are no suggestions; run `uv run --extra asr scribe suggest "<meeting>"` yourself to see
     why."

   Only when suggest exits 0 and stderr has `scribe: note: N profile(s) skipped`, also tell
   The user: "N saved Voice profile(s) come from another embedding model and were not compared;
   save them again from a Meeting made with this model."

   Then ask: "Who is each Speaker? Give a name per Speaker; 'skip' or 'unknown' keeps the
   label." Only when some Speakers have a suggestion, put one line per such Speaker above the
   question, "Speaker N looks like X (0.93)", with X the `participant` value exactly as stored
   and its `cosine` to 2 decimals, and add to the question: "'yes' accepts a suggestion."
   Accept any form ("1 is me, 2 is Sam", "Speaker 2=Sam", one name per line). "me" or
   "I" means the user. If an answer is ambiguous (a name given twice, a Speaker that does not
   exist), ask once more about that Speaker only. If every Speaker is skipped, skip the command.

   A suggestion is accepted only by an explicit yes for that Speaker ("yes", "y", "1 yes",
   "yes to all"); then the `participant` value, in its stored spelling, is the name for step
   6. When the user types a name for a Speaker, that name in their spelling is the name for step 6,
   even when it differs from the suggestion only in case. Enter, an empty answer, or an answer
   that leaves out a Speaker with a suggestion never accepts it: ask once more about that
   Speaker only, and when that answer is again empty or not a yes or a name, the Speaker keeps
   its label. A suggestion never reaches `scribe name` unless the user accepted it.

6. **Record the Naming.** One pair per named Speaker, each pair quoted:

   ```bash
   uv run --extra asr scribe name "<meeting>" "Speaker 1=the user" "Speaker 2=Sam"
   ```

   It rewrites both files; Speakers left out keep their label. Naming again later replaces a
   name, so a mistake is fixed by running it again.

   If it exits 1, show its `scribe: error:` line. For a malformed pair or an unknown Speaker,
   tell the user which Speaker labels exist (from step 4), ask step 5 again for that Speaker only,
   and run the command again. For a transcript.json it cannot read, tell the user as in step 4
   (run the repeat command and choose (2), --force) and stop. If it warns that
   transcript.clean.md does not match, tell the user: "An earlier Clean Transcript no longer
   matches; the cleanup next replaces it."

   **Voice profiles.** Only after the last `scribe name` of this step exits 0, and only for the
   Participants named in this run (the names in the pairs passed to `scribe name` in this run,
   not names an earlier Naming of this Meeting gave), ask exactly "Save a Voice profile for
   <name>? (y/N)" per Participant, one question at a time, with <name> as passed to `scribe
   name`. When every Speaker was skipped, or `scribe name` never exited 0, ask nothing and go
   to step 7. Only an explicit yes ("y", "yes") saves; no, "n", Enter, empty or anything else
   skips that Participant. On yes, run:

   ```bash
   uv run --extra asr scribe profile save "<meeting>" "<name>"
   ```

   Exit 0 prints `<profile file> (<n> sample(s))`: tell the user "Voice profile saved: <profile
   file> (<n> sample(s))." and keep the line for step 11. On exit 2, show the `scribe: error:`
   line, tell the user the next step for it below, and go on to the next Participant's question, or
   to step 7 after the last one; nothing was saved for that Participant:

   - `is not named in this Meeting`: "scribe has not recorded that name in this Meeting. Check
     the spelling against the `named:` list in the line, then run
     `uv run --extra asr scribe profile save "<meeting>" "<name>"` with that spelling."
   - `is named on N Speakers`: "That name is on more than one Speaker, so scribe cannot tell
     which voice to save. To save it later, run <the repeat command>, choose (3), give the
     extra Speaker its own name, and answer yes to this question again."
   - `no embeddings.json in this Meeting folder`, `is unreadable or has no entry for`, or
     `holds no usable mean of windows or model`: "This Meeting has no usable voice embeddings
     for that Speaker, so no profile can be saved from it; save one from another Meeting, or
     run <the repeat command> and choose (2), which rebuilds them with --force and drops the
     Naming."
   - `fewer than the 4 a profile needs`: "That Participant speaks too briefly in this Meeting
     for a Voice profile; save one from a Meeting where they speak longer."
   - `different model than the saved samples`: "The saved Voice profile for that name comes
     from another embedding model; move its file in .scribe/profiles/ away, then run
     `uv run --extra asr scribe profile save "<meeting>" "<name>"` to start it again."
   - `dimensions but the saved samples`: "This Meeting's embeddings do not fit the saved
     profile; run <the repeat command> and choose (2) to rebuild them, then save again."
   - `cannot read ... as a Voice profile`: "The saved Voice profile file is broken; move the
     file the error line names away, then run
     `uv run --extra asr scribe profile save "<meeting>" "<name>"` again."
   - `cannot make a profile file name`: "The name has no letters or digits, so it cannot name a
     profile file; run <the repeat command>, choose (3), and give a name with letters."
   - `cannot write`: "The profile file could not be written; check that .scribe/profiles/ is
     writable, then run `uv run --extra asr scribe profile save "<meeting>" "<name>"` again."
   - `no Meeting folder`: "The Meeting folder is gone or moved. Run <the repeat command>
     again."
   - `cannot read transcript.json as a Transcript`: tell the user as in step 4 (run the repeat
     command and choose (2), --force).
   - Any other exit: show its last stderr line and say "No profile was saved; run
     `uv run --extra asr scribe profile save "<meeting>" "<name>"` yourself to see why."

   If the harness denies the `scribe suggest` of step 5 or a `profile save` here, follow
   **A denied write** at the end of this file.

7. **Speaker stats.** Once the Naming is recorded (also when every Speaker was skipped), run:

   ```bash
   uv run --extra asr scribe stats "<meeting>"
   ```

   It writes `<meeting>/stats.json` and prints one line per Speaker,
   `<name or label>: <mm:ss>, <percent>%, <n> Segments`. Show the lines to the user as a table
   (Speaker, talk time, share, Segments). Run it after the last `scribe name` of step 6, and
   again after any `scribe name` later in this run, so stats.json always has the current names.
   The table is a convenience: on any failure below, tell the user and go on to step 8 unless the
   step says stop.

   - Exit 1, `scribe: error: no Meeting folder ...`: "The Meeting folder is gone or moved. Run
     <the repeat command> again." Stop.
   - Exit 1, `scribe: error: no transcript.json in ...` or `scribe: error: cannot read ...`: tell
     The user as in step 4 (run the repeat command and choose (2), --force) and stop.
   - Exit 1 with a Python traceback and no `scribe: error:` line: show the traceback's last
     line. If it names `UnicodeDecodeError`, transcript.json is not UTF-8 text: tell the user as in
     step 4 (run the repeat command and choose (2), --force) and stop. Otherwise stats.json could
     not be written: tell the user: "stats.json could not be written. Check that the Meeting folder
     is writable and the disk has space, then run
     `uv run --extra asr scribe stats "<meeting>"`." Go on to step 8.
   - Exit 2 (a usage line): the command was mistyped. Run it once more exactly as above; if it
     exits 2 again, show the usage line, tell the user "scribe stats did not run; run
     `uv run --extra asr scribe stats "<meeting>"` yourself to see why.", and go on to step 8.

8. **Cleanup.** Ask exactly: "Clean up the Transcript? (Enter = yes)". Enter, empty or yes
   means yes; no, skip or anything else means skip this step. On yes, follow
   [Cleanup](#cleanup) below, then go to step 9.

9. **Summary and Minutes.** Follow [Summary and Minutes](#summary-and-minutes) below, then go
   to step 10. The user is not asked first: every run writes both.

10. **Destinations.** Follow [Destinations](#destinations) below, then go to step 11.

11. **Done.** Say where the Transcript is: `<meeting>/transcript.md` (for reading) and
    `<meeting>/transcript.json` (for tools), and which Speakers kept their label. Also name
    `<meeting>/stats.json` if step 7 wrote it, `<meeting>/transcript.clean.md` if step 9 used
    it, and `<meeting>/summary.md` and `<meeting>/minutes.md` if step 9 wrote them, with the
    Source line they carry. Name the vault note path if step 10 wrote it, and
    `<meeting>/slack.md` if step 10 wrote it: tell the user they post it themselves, scribe never posts.
    To change a name later, run the repeat command, choose (3) and name
    again: that run writes stats.json, the Summary and the Minutes again with the new names, and
    runs `uv run --extra asr scribe slack "<meeting>"` again so slack.md carries them too. An
    existing vault note is left unchanged, because `vault-note` refuses to overwrite it, so it
    keeps the old names. Only when this run is a re-Naming (choice (3)) and `vault-note` exited 1
    with `already exists and was left unchanged` in step 10, tell the user: "The vault note still has
    the old names. Delete or rename it, then run
    `uv run --extra asr scribe vault-note "<meeting>"` to write it again." On a first run, or when
    no vault note exists, say nothing about it.
    Only when step 6 saved a Voice profile in this run, name each profile file it printed and
    tell the user: "Deleting .scribe/profiles/<slug>.json (the file named above) undoes that save."
    Only when that save printed more than 1 sample, add: "Deleting it also drops the samples
    from earlier Meetings; save them again from those Meetings if you still want them." When
    no profile was saved in this run, say nothing about profiles.

## Cleanup

`CHUNK_SEGMENTS = 50`: the most Segments Claude writes in one chunk. To tune a run, change this
one number; every command below takes it from here.

transcript.clean.md is a copy of transcript.md with misheard and phonetic spellings fixed.
`scribe check-clean` compares it with transcript.json Segment by Segment (contract: the
docstring of `src/scribe/clean.py`), so a Segment that is dropped, merged, added or moved fails
the check. `H` below is `skills/transcribe/clean_chunk.py`, always run as
`uv run --extra asr python H ...`. transcript.md and transcript.json are only read, never
written, in this step.

**Rules for the clean text.** Hold to all of them for every chunk:

- Copy every header line (`**<name>** [mm:ss]` or `[h:mm:ss]`) byte for byte from the lines of
  transcript.md you Read. Never retype a name or a timestamp from memory.
- Fix misheard and phonetic spellings only: a name, a term or a word the speech recognizer got
  wrong, where the context makes the intended word clear. Keep the wording, word order, filler
  words and language as spoken. No translation, no rephrasing, no summary.
- Keep every Segment, in order, under its own header: none added, dropped, merged, split or
  moved. Every Segment keeps some text, even a lone "uh"; a header with no text fails.
- Never write a text line shaped like a header: a line that starts with `**` and ends with
  `** [mm:ss]` or `** [h:mm:ss]`. check-clean reads such a line as a new header and fails.
  If a Segment's text would form one, keep it inside the sentence so the line does not end
  in the bracketed time.
- Nothing before the first header: no title, note or blank comment line.

**Steps.**

1. Remove any earlier cleanup files: `uv run --extra asr python H delete "<meeting>"`.
2. Set `FROM = 1`. For each chunk:
   1. `uv run --extra asr python H raw "<meeting>" FROM CHUNK_SEGMENTS`. It prints one line,
      `Segments a-b of N: transcript.md offset X limit L`. Read `<meeting>/transcript.md` with
      the Read tool at that offset and limit; those lines are Segments a to b exactly as raw.
      Do not print them through Bash: a shell output filter may cut a long chunk. Once it
      prints `no Segments from ...`, every chunk is written: go to step 3.
   2. Write the clean version of exactly those Segments, following the rules above. If
      `<meeting>/transcript.clean.md` does not exist, Write the chunk to it. Otherwise Write the
      chunk to `<meeting>/transcript.clean.chunk.md` and run
      `uv run --extra asr python H append "<meeting>"`.
   3. Check the prefix: `uv run --extra asr scribe check-clean "<meeting>" --partial`. Exit 0
      prints `M of N`; set `FROM = M + 1` and do the next chunk.
3. Full check: `uv run --extra asr scribe check-clean "<meeting>"`. Exit 0 prints `N of N`: tell
   The user "Clean Transcript checked: N of N Segments." and go back to step 9.

**On a check failure (exit 1).** The one stdout line is
`Segment k of N raw / M clean: <field>`. The first time in this cleanup:

1. Run `uv run --extra asr python H keep "<meeting>" K` with `K = k - 1`. It keeps the clean
   Segments before k, which matched, and cuts the rest.
2. Run `raw "<meeting>" k COUNT`, with COUNT reaching the last Segment of the chunk that failed
   (for `missing` at the full check, through N), and Read those lines. If it prints
   `no Segments from ...` (`extra` past N), skip to the check. Rewrite those Segments, keeping the error line in mind: `header` or
   `timestamp` means a header line was not copied exactly; `header and timestamp`, `missing` or
   `extra` means a Segment was dropped, merged, split or added; `empty text` means a Segment lost
   its text; `text before first header` means something was written above the first header.
   Write or append as in step 2.2, then run the same check again and carry on.

A second exit 1 anywhere in the same cleanup ends the cleanup: run
`uv run --extra asr python H delete "<meeting>"` and tell the user: "The Clean Transcript failed its
check twice (<the error line>), so I deleted it. The Summary and Minutes will use the raw
Transcript. To try again later, run <the repeat command>, choose (3), then answer yes to the
cleanup." Go back to step 9.

**`H` fails** (exit 1, one `clean_chunk: error:` line): no retry. Run `H delete` and tell the user:
"A cleanup file step failed (<the error line>), so I deleted the clean file. The Summary and
Minutes will use the raw Transcript. To try again later, run <the repeat command>, choose (3),
then answer yes to the cleanup." If `H delete` itself fails, add: "Delete
<meeting>/transcript.clean.md and transcript.clean.chunk.md by hand before using the folder."
Go back to step 9.

**Exit 2: check-clean cannot check.** No retry; show the `scribe: error:` line, run `H delete`,
and give the user the next step for the message, then say the Summary and Minutes will use the raw
Transcript:

- `no such Meeting folder`: "The Meeting folder is gone or moved. Run <the repeat command>
  again."
- `cannot read transcript.json as a Transcript`: "The Transcript in this Meeting folder cannot be
  read. Run <the repeat command> and choose (2) to rebuild it with --force."
- `no transcript.clean.md in the Meeting folder`: "The clean file was not written. Run
  <the repeat command>, choose (3), and answer yes to the cleanup to try again."
- `cannot read transcript.clean.md as UTF-8 text`: same next step as the line above.

Go back to step 9.

**Length warning.** If a passing check, partial or full, also prints `clean/raw text length ratio
... is outside 0.5-2.0` on stderr, the clean text is far shorter or longer than the raw. Carry
on. Only when the cleanup ends with the full check passing, tell the user once, right after "Clean
Transcript checked: N of N Segments.": "The clean text differs a lot in length from the raw;
skim it before using it." When the cleanup ends without a clean file (a fallback above), drop
the warning: there is no clean file left to skim.


## Summary and Minutes

summary.md is the Summary (decisions, action items with owner) and minutes.md the Minutes (per
topic: what was discussed, decided and agreed), both written by Claude in the Meeting folder.
`O` below is `skills/transcribe/outputs.py`, always run as
`uv run --extra asr python O ...`. The templates are `summary.template.md` and
`minutes.template.md` beside this file. Their heading lines (the lines starting with `#`) are a
contract: `scribe vault-note` and `scribe slack` find the sections of both files by those exact lines.

**Source.** Run `uv run --extra asr scribe check-clean "<meeting>"` and pick the source file and
its Source line from the result. Copy the Source line exactly:

- Exit 0: the Clean Transcript. Read `transcript.clean.md`; Source line
  `Source: Clean Transcript (transcript.clean.md)`.
- Exit 2, `no transcript.clean.md in the Meeting folder`: the raw Transcript. Read
  `transcript.md`. If step 8 ran a cleanup in this run that ended without a clean file (a
  second check failure, an `H` failure or a check-clean exit 2), the Source line is
  `Source: raw Transcript (transcript.md), Clean Transcript failed its check`; when step 8 was
  answered no, `Source: raw Transcript (transcript.md), cleanup skipped`.
- Exit 1, or exit 2 with `cannot read transcript.clean.md as UTF-8 text` (an older clean file
  that no longer matches): the raw Transcript. Read `transcript.md`; Source line
  `Source: raw Transcript (transcript.md), Clean Transcript failed its check`. Tell the user: "An
  older Clean Transcript does not match the Transcript, so the Summary and Minutes use the raw
  one. To clean it again, run <the repeat command>, choose (3), and answer yes to the cleanup."
- Exit 2, `no such Meeting folder` or `cannot read transcript.json as a Transcript`: give the
  next step for that line from "Exit 2: check-clean cannot check" under Cleanup and stop.

**Read the source.** Run `uv run --extra asr python O lines "<meeting>" <file>`; it prints
`<file>: N lines`. Read the file with the Read tool at offset 1, 501, 1001 and so on, limit 500,
until you have read line N. Read all of it before writing anything.

**Write.**

1. `uv run --extra asr python O delete "<meeting>"` removes an earlier summary.md and minutes.md.
2. Read both templates with the Read tool.
3. Write `<meeting>/summary.md`, then `<meeting>/minutes.md`, holding to these rules:

- Copy every heading line of the template byte for byte, in the template's order. Never
  translate, rename, reorder, add or drop a heading, even when the Meeting was in Dutch; no
  other line in the file starts with `#`. In minutes.md, `## Topic: <topic title>` and its three
  `###` headings repeat once per topic, in the order the topics came up; only the topic title
  after `## Topic: ` is yours.
- The first line after the title is the Source line chosen above.
- Replace every `<...>` placeholder line with content. A section with nothing to report holds
  the one line `- None.`
- Write the text in the language mostly spoken in the Meeting. The headings, the Source
  line, `(owner: <name>)`, `(owner: unclear)` and `- None.` stay in English exactly as written
  here, whatever the Meeting's language: `O check`, `scribe vault-note` and `scribe slack` look for those strings.
- Use only what the Transcript says. Name people exactly as the Transcript's headers do: the
  Participant name, or the Speaker label when the Speaker was not named.
- Summary: `## Overview` is 2 to 5 sentences. `## Decisions` has one line per decision the
  Meeting settled. `## Action items` has one line per task someone took on or was given, as
  `- <action> (owner: <name>)`, with a deadline inside the action text if one was said. When
  the Transcript does not make clear who owns it, write `(owner: unclear)`; never guess an
  owner.
- Minutes: under each topic, `### Discussed` has the points raised and who raised them,
  `### Decided` the decisions on that topic, and `### Agreed` what people agreed to do or
  accept without a formal decision. Every decision in the Summary also appears under the
  `### Decided` of one topic.

**Check.** Run `uv run --extra asr python O check "<meeting>"`. Exit 0 prints
`summary.md and minutes.md follow the templates`: show the user summary.md as written and say
"Summary and Minutes written from <the source>." Then go back to step 10.

On exit 1 of any `O` command above (`delete`, `lines` or `check`; `outputs: error:` lines):

- `no such Meeting folder`: "The Meeting folder is gone or moved. Run <the repeat command>
  again." Stop.
- `cannot remove ...` or `cannot read ...` (a file step failed, not the content): tell the user:
  "summary.md or minutes.md could not be written (<the error line>). Check that the Meeting
  folder is writable, then run <the repeat command> and choose (3)." Go back to step 10.
- Any other line names a file, a line number and what the template wants there. The first
  time, Read that file, fix those lines with the Edit tool (Write the file again for
  `not written`), and run the check again. A second exit 1 keeps both files: show the error
  lines and tell the user: "summary.md or minutes.md does not follow its template, so the vault note and slack.md may not
  find its sections. Read it as notes, or run <the repeat command> and choose (3) to write
  both again." Go back to step 10.

If the Write tool itself fails, tell the user why in one line and add: "Run <the repeat command>
and choose (3) to write the Summary and Minutes again." Go back to step 10.

## Destinations

Runs after the Summary and Minutes step. The user posts nothing through scribe: the vault note is a
file in their vault and slack.md is a file for them to paste. Every command is run from the current
working directory. Both destinations are skipped when summary.md was not written (step 9 stopped before it):
tell the user "No Summary was written, so there is nothing to send. Run <the repeat command> and
choose (3)." and go to step 11.

**Vault note.** Ask exactly: "Save the Summary and Minutes to your vault? (y/N)". Only an
explicit yes ("y", "yes") means yes. No, "n", Enter, empty or anything else skips the vault note:
go straight to **Slack message** below, and do not run `config` or `vault-note`.

On yes:

1. Run `uv run --extra asr scribe config show`. If its `vault:` line says `(not set)`, this is
   the first use. Ask: "Path of your vault?" Then ask: "Folder in the vault? (Enter = Meetings)".
   Run `uv run --extra asr scribe config set vault "<vault path>"`, and when the user gave a folder
   run `uv run --extra asr scribe config set folder "<folder>"`. On Enter set no folder: the
   default applies. If the vault is already set, ask nothing and use it.
2. Run `uv run --extra asr scribe vault-note "<meeting>"`. Exit 0 prints the note's path: tell
   The user "Vault note written: <path>."

**`config` exits 2** (`scribe: error:` line on stderr; nothing was written). Show the line, then:

- `folder ... must be relative to the vault, without '..'`: "The folder must be a path inside
  the vault, such as Meetings. Give me another folder." Ask for the folder again
  and run `config set folder` again; if the user gives none, skip the vault note.
- `unknown key`: the command was mistyped. Run it once more exactly as above; if it exits 2
  again, tell the user "scribe config did not run; run `uv run --extra asr scribe config show` yourself
  to see why." and skip the vault note.
- `cannot read ... as JSON` or `... is not a JSON object`: "The scribe config file is broken.
  Fix or delete the file named in the error line, then run <the repeat command> and choose (3)."
  Skip the vault note.

**`vault-note` exits non-zero** (`scribe: error:` line on stderr). Show the line, then:

- Exit 1, `already exists and was left unchanged`: the note is already in the vault and was not
  touched. Tell the user: "A vault note for this Meeting already exists and was left unchanged. Delete
  or rename it if you want a new one, then run `uv run --extra asr scribe vault-note "<meeting>"`."
- Exit 2, `no vault configured` or `is not a directory`: "The vault path is not set or does not
  exist. Give me the vault path again." Run `config set vault` as in step 1 with the new path,
  then run `vault-note` once more. A second exit 2 skips the vault note.
- Exit 2, `must be relative to the vault, without '..'`: the config file holds a bad folder.
  Ask for the folder as for `config set folder` above, set it and run `vault-note` once more.
- Exit 2, `no summary.md`, `no minutes.md` or `no transcript.json`, or `cannot read` one of them:
  "The vault note needs summary.md, minutes.md and transcript.json in the Meeting folder. Run
  <the repeat command> and choose (3) to write them again."
- Exit 2, `does not start with YYYY-MM-DD`: "The Meeting folder name must start with
  YYYY-MM-DD. Rename the folder to <YYYY-MM-DD>-<name>, then run
  `uv run --extra asr scribe vault-note "<meeting>"`."
- Exit 2, `no Meeting folder` or any other `scribe: error:` line: "The Meeting folder is gone or
  moved. Run <the repeat command> again."

**Slack message.** Always, whatever the user answered about the vault:

```bash
uv run --extra asr scribe slack "<meeting>"
```

Exit 0 prints the path of `<meeting>/slack.md`. Tell the user: "slack.md is at <path>. Open it and
post it in Slack yourself; scribe never posts." Exit non-zero (`scribe: error:` line): show it and
give the user the next step for it below, then go on to step 11:

- Exit 2, `no stats.json`: run `uv run --extra asr scribe stats "<meeting>"`, then the slack
  command once more. If either fails, tell the user: "slack.md was not written. Run
  `uv run --extra asr scribe stats "<meeting>"`, then `uv run --extra asr scribe slack "<meeting>"`."
- Exit 2, `no summary.md`: "slack.md was not written. Run <the repeat command> and choose (3) to
  write the Summary again."
- Exit 1, `no '## ... ' heading`: "summary.md does not follow its template, so slack.md was not
  written. Run <the repeat command> and choose (3) to write the Summary again."
- Exit 1, `stats.json` unreadable or of another version: run
  `uv run --extra asr scribe stats "<meeting>"` and the slack command once more; if it fails
  again, tell the user "slack.md was not written. Run
  `uv run --extra asr scribe stats "<meeting>"`, then `uv run --extra asr scribe slack "<meeting>"`."
- Exit 1, `cannot read summary.md`: "summary.md cannot be read. Run <the repeat command> and
  choose (3) to write it again."
- Any other line: "slack.md was not written (<the error line>). Run <the repeat command> and
  choose (3)."

**A denied write.** The harness's auto-mode classifier may block a vault-note or slack command,
or the write of slack.md or the vault note, because the file holds Meeting content. That is a
denial, not an exit code. Do not retry, rerun in another way, or write the file by another
route. Tell the user: "only you can approve it in this session, or run the printed command yourself
with !" and print the exact command that was denied, for example
`! uv run --extra asr scribe vault-note "<meeting>"`, then go on to the next destination, or to
step 11.

The same holds for the `scribe suggest` command of step 5 and a `scribe profile save` of step
6, which write suggestions.json and a Voice profile file: tell the user the same sentence and print
the denied command, for example `! uv run --extra asr scribe profile save "<meeting>" "<name>"`.
Only when the denied command is `scribe suggest`, go on with the Naming question with no
suggestions. Only when it is a `profile save`, nothing was saved for that Participant: go on to
the next Participant's save question, or to step 7 after the last one.
