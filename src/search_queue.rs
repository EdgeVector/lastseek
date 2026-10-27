//! Refuse a partial LastDB feed after optional Search delivery stops.

use crate::LastSeekError;
use std::path::{Path, PathBuf};

pub const REBUILD_REQUIRED: &str = ".rebuild-required";

pub fn inbox(args: &[String], home: &Path) -> Result<PathBuf, LastSeekError> {
    let selected = args
        .windows(2)
        .find(|pair| pair[0] == "--inbox")
        .map(|pair| PathBuf::from(&pair[1]));
    let source = match selected {
        Some(path) => path,
        None => match std::fs::read(home.join("search-source-inbox.json")) {
            Ok(bytes) => serde_json::from_slice::<PathBuf>(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                crate::ingest::default_inbox()
            }
            Err(error) => return Err(error.into()),
        },
    };
    Ok(std::path::absolute(source)?)
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

pub fn check_command(args: &[String], command: &str, home: &Path) -> Result<(), LastSeekError> {
    if matches!(command, "query" | "drain") {
        let source = inbox(args, home)?;
        ensure_current(&source)?;
        if command == "drain" {
            crate::store::create_private_dir_all(home)?;
            crate::store::write_private_atomic(
                &home.join("search-source-inbox.json"),
                &serde_json::to_vec(&source)?,
            )?;
        }
    }
    Ok(())
}

pub fn reported_state<'a>(
    state: &'a str,
    args: &[String],
    home: &Path,
) -> Result<&'a str, LastSeekError> {
    Ok(if required(&inbox(args, home)?)? {
        "rebuild_required"
    } else {
        state
    })
}
