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
/// for `query <text>`, zero for everything else.
pub struct VerbSpec {
    pub verb: &'static str,
    pub usage: &'static str,
    pub value_flags: &'static [&'static str],
    pub bool_flags: &'static [&'static str],
    pub positionals: usize,
}

/// `--home` is read by every verb (the store directory is resolved before
/// dispatch), so it is appended to each spec rather than repeated in all of
/// them.
const UNIVERSAL_VALUE_FLAGS: &[&str] = &["--home"];

const SPECS: &[VerbSpec] = &[
    VerbSpec {
        verb: "status",
        usage: "\
lastseek status [--home DIR] [--table PATH] [--coverage]

  --coverage        add per-schema coverage. Reads the node's schema catalog
                    WITH row counts, which is expensive — opt-in on purpose.
  --coverage-floor F  minimum vectors/source ratio for healthy (default 0.95)
  --source-counts PATH  declared live-record counts
  --catalog PATH|URL    catalog fixture instead of the live socket
  --lastdb-socket PATH  LastDB Unix socket

Reports `freshness_lag_ms` — how far behind the node the applied corpus is,
derived from the watermark alone. `--queue` adds the inbox backlog, which
costs a directory listing.
",
        value_flags: &[
            "--table",
            "--coverage-floor",
            "--source-counts",
            "--catalog",
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
            "--catalog",
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
        value_flags: &["--table", "--k", "--schema", "--min-score", "--caller"],
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
        verb: "drain",
        usage: "\
lastseek drain [--home DIR] [--inbox DIR] [--max-files N] [--consume]

Applies the host's IndexChangeBatch feed. BOUNDED: --max-files defaults to
2000 and `remaining` reports what the cap left behind, because an unbounded
drain over a large backlog wrote 549.80 GB in twenty minutes and took the
machine down on 2026-08-07.

  --consume      DELETE each batch file after applying it.
",
        value_flags: &["--table", "--inbox", "--max-files"],
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

pub fn spec_for(verb: &str) -> Option<&'static VerbSpec> {
    SPECS.iter().find(|s| s.verb == verb)
}

/// Every flag a verb accepts, for error messages. Sorted so the list a user
/// sees does not depend on declaration order.
fn known_flags(spec: &VerbSpec) -> Vec<String> {
    let mut all: Vec<String> = spec
        .value_flags
        .iter()
        .chain(spec.bool_flags.iter())
        .chain(UNIVERSAL_VALUE_FLAGS.iter())
        .map(|s| s.to_string())
        .collect();
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
    let Some(spec) = spec_for(verb) else {
        // Unknown verbs are handled by the dispatcher's own error, which
        // prints the full usage. Nothing to check.
        return Ok(Invocation::Run);
    };

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

        if spec.value_flags.contains(&tok) || UNIVERSAL_VALUE_FLAGS.contains(&tok) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn help_on_a_mutating_verb_is_help_and_never_the_mutation() {
        // The regression this module exists for. `drain --help` applied 1,234
        // batch files to the primary index on 2026-08-17.
        for v in [
            args(&["drain", "--help"]),
            args(&["drain", "-h"]),
            args(&["drain", "--consume", "--help"]),
            args(&["compact", "--help"]),
            args(&["add", "--help"]),
        ] {
            match check(&v).unwrap() {
                Invocation::Help(usage) => assert!(!usage.is_empty()),
                Invocation::Run => panic!("{v:?} would have executed the verb"),
            }
        }
    }

    #[test]
    fn help_is_answered_before_a_later_argument_can_fail() {
        // Asking for help while holding a wrong flag must print help, not
        // scold — otherwise the way out of the error is unreachable.
        match check(&args(&["query", "--help", "--limit", "50"])).unwrap() {
            Invocation::Help(_) => {}
            Invocation::Run => panic!("expected help"),
        }
    }

    #[test]
    fn the_flag_brain_and_kanban_spell_by_muscle_memory_is_refused_by_name() {
        // `--limit` is what both consumers call this flag. It used to return
        // the default 20 and exit 0.
        let err = check(&args(&["query", "text", "--limit", "50"])).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--limit"), "{msg}");
        assert!(msg.contains("query"), "{msg}");
        assert!(msg.contains("--k"), "names the flag that does work: {msg}");
    }

    #[test]
    fn a_typo_of_a_real_flag_is_refused_rather_than_silently_disabling_it() {
        let err = check(&args(&["query", "text", "--min-scor", "0.9"])).unwrap_err();
        assert!(err.to_string().contains("--min-scor"));
    }

    #[test]
    fn every_verb_rejects_an_invented_flag() {
        for verb in [
            "status",
            "doctor",
            "query",
            "add",
            "compact",
            "drain",
            "schema-table",
            "coverage-declare",
        ] {
            let mut v = args(&[verb, "--totally-bogus", "zzz"]);
            if verb == "query" {
                v.insert(1, "text".to_string());
            }
            assert!(check(&v).is_err(), "{verb} accepted an unrecognized flag");
        }
    }

    #[test]
    fn the_flags_each_verb_documents_are_accepted() {
        let ok = [
            args(&["status", "--home", "/tmp/x", "--coverage"]),
            args(&["status", "--coverage-floor", "0.9", "--queue"]),
            args(&["query", "some text", "--k", "5", "--schema", "Card"]),
            args(&["query", "t", "--schema", "A", "--schema", "B", "--exact"]),
            args(&[
                "drain",
                "--inbox",
                "/tmp/i",
                "--max-files",
                "10",
                "--consume",
            ]),
            args(&["schema-table", "--from", "/tmp/a.json", "--table", "/tmp/t"]),
            args(&[
                "coverage-declare",
                "--schema",
                "Card",
                "--live-records",
                "9",
            ]),
            args(&["compact", "--home", "/tmp/x"]),
        ];
        for v in ok {
            match check(&v) {
                Ok(Invocation::Run) => {}
                Ok(Invocation::Help(_)) => panic!("{v:?} should run, not help"),
                Err(e) => panic!("{v:?} rejected: {e}"),
            }
        }
    }

    #[test]
    fn a_negative_value_is_a_value_and_not_an_unknown_flag() {
        match check(&args(&["query", "t", "--min-score", "-0.5"])).unwrap() {
            Invocation::Run => {}
            Invocation::Help(_) => panic!("expected run"),
        }
    }

    #[test]
    fn a_value_flag_with_no_value_is_named_rather_than_ignored() {
        let err = check(&args(&["drain", "--max-files"])).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--max-files"), "{msg}");
    }

    #[test]
    fn a_stray_positional_is_refused_where_the_verb_takes_none() {
        assert!(check(&args(&["drain", "oops"])).is_err());
        // query legitimately takes exactly one.
        assert!(check(&args(&["query", "one"])).is_ok());
        assert!(check(&args(&["query", "one", "two"])).is_err());
    }
}
