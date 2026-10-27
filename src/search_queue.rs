//! Refuse a partial LastDB feed after optional Search delivery stops.

use crate::LastSeekError;
use std::path::{Path, PathBuf};

pub const REBUILD_REQUIRED: &str = ".rebuild-required";

pub fn inbox(args: &[String]) -> PathBuf {
    args.windows(2)
        .find(|pair| pair[0] == "--inbox")
        .map(|pair| PathBuf::from(&pair[1]))
        .unwrap_or_else(crate::ingest::default_inbox)
}

pub fn required(inbox: &Path) -> Result<bool, LastSeekError> {
    Ok(inbox.join(REBUILD_REQUIRED).try_exists()?)
}

pub fn ensure_current(inbox: &Path) -> Result<(), LastSeekError> {
    if required(inbox)? {
        return Err(LastSeekError::SearchRebuildRequired);
    }
    Ok(())
}

pub fn check_command(args: &[String], command: &str) -> Result<(), LastSeekError> {
    if matches!(command, "query" | "drain") {
        ensure_current(&inbox(args))?;
    }
    Ok(())
}

pub fn reported_state<'a>(state: &'a str, args: &[String]) -> Result<&'a str, LastSeekError> {
    Ok(if required(&inbox(args))? {
        "rebuild_required"
    } else {
        state
    })
}
