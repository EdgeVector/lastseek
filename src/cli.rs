//! Argument checking for the `lastseek` CLI.
//!
//! # Why this module exists
//!
//! The CLI used to read its flags by scanning the argument vector for the names
//! it cared about (`args.iter().position(|a| a == "--k")`). Anything it did not
//! care about was never looked at, so an unrecognized flag was consumed in
//! silence and the command ran as if it had not been typed.
//!
//! That produced two failures, measured live on 2026-08-17:
//!
//! 1. **Silent wrong results.** `--limit 50` — the spelling brain and kanban
//!    both use — returned the default 20 and exited 0. A typo of `--min-score`
//!    disabled the score floor and reported success. A caller could not tell an
//!    honored flag from an ignored one.
//! 2. **Unintended mutations.** `lastseek drain --help` did not print help. The
//!    flag was discarded and `drain` **ran**, with its default `--max-files
//!    2000`, against the live inbox — 1,234 batch files applied to the primary
//!    index and the watermark advanced, exit 0. It happened twice in three
//!    runs of the scheduled owner job, the second time to a run that had read
//!    the warning and was deliberately avoiding it, because `--help` does not
//!    read as a flag you choose to type. It reads as how you find out which
//!    flags exist.
//!
//! So flag names are now checked as strictly as flag values already were, and
//! `--help` is answered before any side effect. The shape of the error follows
//! `lastgit`, the sibling tool in the same stack, which already gets this right:
//! name the flag, name the verb, name the alternative.
//!
//! # Scope
//!
//! This is a checker, not a parser. The verbs still read their own values
//! through `flag_value` / `parse_flag`; `check` runs first and refuses anything
//! those readers would have ignored. Keeping it a separate pass means a verb
//! cannot forget to opt in — an unlisted verb is itself an error.

use crate::LastSeekError;

/// What a verb accepts.
///
/// `value_flags` take the next argument as their value; `bool_flags` stand
/// alone. `positionals` is how many bare arguments may follow the verb — one
/// for `query <text>` and `search <text>`, zero for everything else.
pub struct VerbSpec {
    pub verb: &'static str,
    pub usage: &'static str,
    pub value_flags: &'static [&'static str],
    pub bool_flags: &'static [&'static str],
    pub positionals: usize,
}

/// `--home` is read by the older verbs (the store directory is resolved
/// before dispatch), so it is appended to those specs rather than repeated
/// in all of them. `index`, `search`, and `status --app` do not take it.
const UNIVERSAL_VALUE_FLAGS: &[&str] = &["--home"];

const SPECS: &[VerbSpec] = &[
    VerbSpec {
        verb: "index",
        usage: "\
lastseek index --app NAME --schema S [--schema S]... [--wait]

Names the schemas for one app and builds that app's index. An app does not
name a directory. The index is $LASTSEEK_HOME/apps/NAME, or
~/.lastseek/apps/NAME when LASTSEEK_HOME is unset. LASTDB_APP_DATA does not
move it. NAME is 1-64 characters of [a-z0-9_-].

When the published index is exactly this schema set and does not need a
rebuild, this prints {\"state\":\"ready\"} and builds nothing. When a build is
already running, it prints {\"state\":\"building\"}. Otherwise it starts the
build in a new process group and prints {\"state\":\"building\"} without
waiting. Progress goes to <dir>/index.log.

  --wait         run the build in the foreground and exit non-zero if it fails
  --schema S     repeatable. The whole set this app wants indexed.
",
        value_flags: &["--app", "--schema"],
        bool_flags: &["--wait"],
        positionals: 0,
    },
    VerbSpec {
        verb: "search",
        usage: "\
lastseek search --app NAME <text> [--schema S]... [--k N] [--min-score F] [--max-changes N]

Answers from the app's index. When that index is not ready, prints
{\"ok\":false,\"code\":\"index_not_ready\",\"state\":S} and exits 3. Search
does not start a first build.

A ready index is caught up from the LastDB changes list first, at most
--max-changes records (default 200). The catch-up is skipped when another
process holds the update lock, and a search still answers when LastDB cannot
be reached. A gap or a replaced changes list starts a background rebuild and
answers from the index that is already published. \"stale\" is true in each
of those cases.

  --schema S       repeatable. Narrows to schemas this index holds. A schema
                   it does not hold is an error (exit 2).
  --k N            number of results (default 20)
  --min-score F    drop hits below this score
  --max-changes N  changes applied before answering (default 200)
",
        value_flags: &["--app", "--schema", "--k", "--min-score", "--max-changes"],
        bool_flags: &[],
        positionals: 1,
    },
    VerbSpec {
        verb: "status",
        usage: "\
lastseek status [--home DIR] [--table PATH] [--coverage]

`lastseek status --app NAME` reports one app's index and does not take these
flags. See `lastseek index --help`.

  --coverage        add per-schema coverage. Reads the node's schema catalog
                    WITH row counts, which is expensive — opt-in on purpose.
  --coverage-floor F  minimum vectors/source ratio for healthy (default 0.95)
  --source-counts PATH  declared live-record counts
  --lastdb-socket PATH  LastDB Unix socket

Reports `freshness_lag_ms` — how far behind the node the applied corpus is,
derived from the watermark alone. `--queue` adds the inbox backlog, which
costs a directory listing.
",
        value_flags: &[
            "--table",
            "--coverage-floor",
            "--source-counts",
            "--lastdb-socket",
            "--inbox",
        ],
        bool_flags: &["--coverage", "--queue"],
        positionals: 0,
    },
    VerbSpec {
        verb: "doctor",
        usage: "\
lastseek doctor [--home DIR] [--table PATH] [--coverage]

Currently an alias for `status`. Same flags.
",
        value_flags: &[
            "--table",
            "--coverage-floor",
            "--source-counts",
            "--lastdb-socket",
            "--inbox",
        ],
        bool_flags: &["--coverage", "--queue"],
        positionals: 0,
    },
    VerbSpec {
        verb: "query",
        usage: "\
lastseek query <text> [--home DIR] [--k N] [--schema S]... [--min-score F] [--caller C]

Use --inbox DIR to select the same LastDB feed as drain. Otherwise the store uses its saved source inbox.

Answers record keys and scores. The index retains no record text, so a
response never carries a body: point-fetch the record from LastDB by key.

  --k N          number of results (default 20). NOT --limit.
  --schema S     repeatable. Accepts a readable name (Card), an identity hash,
                 or a registry name. A name that resolves to nothing is an
                 ERROR, not an empty result — those are different answers.
  --min-score F  drop hits below this score
  --caller C     stable operator label for the process that issued the query
  --exact        REFUSED (exit 1): substring matching needs retained text.
",
        value_flags: &["--table", "--k", "--schema", "--min-score", "--caller", "--inbox"],
        // `--exact` stays a recognised flag so the refusal names the reason
        // instead of the generic unknown-flag message.
        bool_flags: &["--exact"],
        positionals: 1,
    },
    VerbSpec {
        verb: "add",
        usage: "\
lastseek add [--home DIR]

Reads Records as JSONL on stdin, embeds them, appends to the pending log.
",
        value_flags: &[],
        bool_flags: &[],
        positionals: 0,
    },
    VerbSpec {
        verb: "compact",
        usage: "\
lastseek compact [--home DIR]

Folds the pending log into the slab and publishes a new generation.
",
        value_flags: &[],
        bool_flags: &[],
        positionals: 0,
    },
    VerbSpec {
        verb: "rebuild",
        usage: "\
lastseek rebuild [--home DIR] [--lastdb-socket PATH] [--schema NAME]...

Build a search index of the named schemas from live LastDB records, then
follow the changes extension. The app that wants search names its schemas;
LastSeek indexes nothing else, and embeds every field of a named schema. Use a
separate --home for each app's index.
",
        value_flags: &["--lastdb-socket", "--schema", "--table"],
        bool_flags: &[],
        positionals: 0,
    },
    VerbSpec {
        verb: "drain",
        usage: "\
lastseek drain [--home DIR] [--lastdb-socket PATH] [--inbox DIR] [--max-files N] [--consume]

Applies the host's IndexChangeBatch feed. BOUNDED: --max-files defaults to
2000 and `remaining` reports what the cap left behind, because an unbounded
drain over a large backlog wrote 549.80 GB in twenty minutes and took the
machine down on 2026-08-07.

  --consume      DELETE each batch file after applying it.

After an online rebuild, drain follows POST /v1/x/changes and --max-files caps
the number of change events. The socket is $LASTDB_SOCKET_PATH when LASTDB_APP
is lastseek or unset. Another app's socket path is not used. A changed list
id or a gap rebuilds the index. File consumption only applies to inbox stores.
",
        value_flags: &["--table", "--lastdb-socket", "--inbox", "--max-files"],
        bool_flags: &["--consume"],
        positionals: 0,
    },
    VerbSpec {
        verb: "schema-table",
        usage: "\
lastseek schema-table [--from PATH|URL] [--table PATH]

Refreshes the Schema Service resolution table. Schema names come from the
Schema Service directly, never proxied through LastDB.
",
        value_flags: &["--from", "--table"],
        bool_flags: &[],
        positionals: 0,
    },
    VerbSpec {
        verb: "coverage-declare",
        usage: "\
lastseek coverage-declare --schema S --live-records N [--source TEXT] [--at TS]

Declares how many LIVE records a schema has. This is the only denominator
coverage divides by, because it is the only count the owning app would
recognise as its own.
",
        value_flags: &[
            "--table",
            "--schema",
            "--live-records",
            "--source",
            "--at",
            "--source-counts",
        ],
        bool_flags: &[],
        positionals: 0,
    },
];

/// `status --app` is the app index, not the older store report.
///
/// `--home` is not accepted. An app does not name a directory.
const APP_STATUS: VerbSpec = VerbSpec {
    verb: "status",
    usage: "\
lastseek status --app NAME

Prints one JSON object and exits 0 for every state:

  absent            nothing built or building
  building          a build holds the build lock
  ready             the published index does not need a rebuild
  failed            the last build failed, or a building record has no lock holder
  rebuild_required  the changes list needs a new build

A schema nobody named is not in this index. LASTDB_APP_DATA does not move
the directory. See `lastseek index --help`.
",
    value_flags: &["--app"],
    bool_flags: &[],
    positionals: 0,
};

pub fn spec_for(verb: &str) -> Option<&'static VerbSpec> {
    SPECS.iter().find(|s| s.verb == verb)
}

/// The app commands do not take `--home`. An app names itself, and LastSeek
/// owns the directory under `$LASTSEEK_HOME/apps` or `~/.lastseek/apps`.
fn allows_home(spec: &VerbSpec) -> bool {
    spec.verb != "index" && spec.verb != "search" && !spec.value_flags.contains(&"--app")
}

/// Every flag a verb accepts, for error messages. Sorted so the list a user
/// sees does not depend on declaration order.
fn known_flags(spec: &VerbSpec) -> Vec<String> {
    let mut all: Vec<String> = spec
        .value_flags
        .iter()
        .chain(spec.bool_flags.iter())
        .map(|s| s.to_string())
        .collect();
    if allows_home(spec) {
        all.extend(UNIVERSAL_VALUE_FLAGS.iter().map(|s| s.to_string()));
    }
    all.push("--help".to_string());
    all.sort();
    all.dedup();
    all
}

/// The outcome of checking an argument vector.
#[derive(Debug)]
pub enum Invocation {
    /// `--help` / `-h` was asked for. Print this and exit 0, executing nothing.
    Help(&'static str),
    /// The arguments are valid for this verb. Proceed.
    Run,
}

/// True for a token that is meant as a flag rather than a value.
///
/// A lone `-` and anything that parses as a negative number are values —
/// `--min-score -0.5` must not be read as a flag named `-0.5`. In practice
/// values are skipped by position before reaching here; this guards the case
/// where one appears where a flag was expected.
fn looks_like_flag(tok: &str) -> bool {
    if !tok.starts_with('-') || tok == "-" || tok == "--" {
        return false;
    }
    tok.parse::<f64>().is_err()
}

/// Check `args` (the full argument vector, verb first) against the verb's spec.
///
/// Runs BEFORE any side effect. Returns `Help` when the user asked for usage,
/// `Run` when every flag is recognized and every value-taking flag has a value,
/// and an error naming the offending flag otherwise.
pub fn check(args: &[String]) -> Result<Invocation, LastSeekError> {
    let Some(verb) = args.first() else {
        return Ok(Invocation::Run);
    };
    // `status --app` is a different command from the older store report.
    // Choosing the spec here keeps `--coverage` and `--home` off it, and
    // keeps `--app` off the older report.
    let app_status = verb == "status" && args.iter().any(|arg| arg == "--app");
    let Some(spec) = (if app_status {
        Some(&APP_STATUS)
    } else {
        spec_for(verb)
    }) else {
        // Unknown verbs are handled by the dispatcher's own error, which
        // prints the full usage. Nothing to check.
        return Ok(Invocation::Run);
    };
    let home = allows_home(spec);

    let mut positionals = 0usize;
    let mut i = 1usize;
    while i < args.len() {
        let tok = args[i].as_str();

        // Answered before anything else, and before any side effect: `--help`
        // is how a caller finds out what the other flags are, so it must never
        // be the thing that runs the command.
        if tok == "--help" || tok == "-h" {
            return Ok(Invocation::Help(spec.usage));
        }

        if !looks_like_flag(tok) {
            positionals += 1;
            if positionals > spec.positionals {
                return Err(LastSeekError::UnexpectedArgument {
                    arg: tok.to_string(),
                    verb: spec.verb.to_string(),
                });
            }
            i += 1;
            continue;
        }

        if spec.value_flags.contains(&tok) || (home && UNIVERSAL_VALUE_FLAGS.contains(&tok)) {
            if i + 1 >= args.len() {
                return Err(LastSeekError::MissingFlagValue {
                    flag: tok.to_string(),
                    verb: spec.verb.to_string(),
                });
            }
            // Skip the value without inspecting it, so a value that happens to
            // start with `-` is not mistaken for the next flag.
            i += 2;
            continue;
        }

        if spec.bool_flags.contains(&tok) {
            i += 1;
            continue;
        }

        return Err(LastSeekError::UnknownFlag {
            flag: tok.to_string(),
            verb: spec.verb.to_string(),
            known: known_flags(spec),
        });
    }

    Ok(Invocation::Run)
}
