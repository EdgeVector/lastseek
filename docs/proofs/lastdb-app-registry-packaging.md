# LastDB App Registry Packaging Proof

**Author:** Cloud Agent  
**Date:** 2026-10-24  
**Context:** Make Lastseek installable through the LastDB app registry

## Summary

Lastseek is now packaged for the LastDB app registry following the same pattern as Brain and LastSecrets. The app can be installed via `lastdb app install lastseek` once registry publishing is configured.

## Changes Made

### 1. App Manifest (`lastseek.app.json`)

Created the LastDB app manifest declaring:
- `app_id`: `lastseek`
- `display_name`: `lastseek`
- `description`: Memory-mapped semantic vector search description
- `homepage_url`: Repository link
- **`lastdb`: `">=0.23.0, <0.25.0"`** — Version constraint per EdgeVector/lastdb PR #353

The version range `>=0.23.0, <0.25.0` reflects:
- Minimum `0.23.0`: First version with stable app registry support
- Maximum `<0.25.0`: Conservative upper bound before next major version

### 2. Artifact Configuration (`.lastgit/artifacts.json`)

Updated to include the manifest in published artifacts:
```json
{
  "artifacts": [
    {
      "app": "lastseek",
      "paths": [
        "dist/lastseek",
        "lastseek.app.json",
        "LICENSE",
        "THIRD_PARTY_NOTICES"
      ],
      "platform": "darwin-arm64"
    }
  ]
}
```

## Verification Required

### Build Verification (CI)

The artifact must be built and verified in CI:

1. **Build command** (from `.lastgit/artifact-release.sh`):
   ```bash
   cargo build --locked --release --features neural
   ```

2. **Smoke checks**:
   - Binary runs: `dist/lastseek --help`
   - Neural feature present: Output mentions `bge-small-en-v1.5`
   - Version matches: `dist/lastseek --version` == `Cargo.toml` version

3. **Artifact contents**:
   - `dist/lastseek` (executable)
   - `lastseek.app.json` (manifest)
   - `LICENSE` (Apache-2.0)
   - `THIRD_PARTY_NOTICES` (ONNX Runtime, etc.)

### Install/Run Verification

Once published to the registry, verify the full install flow:

```bash
# 1. Install via registry
lastdb app install lastseek

# 2. Verify installation
lastseek --version
# Expected: 0.1.1 (or current version)

# 3. Check manifest loaded
# The installer should validate the lastdb version constraint
# and accept/reject based on the running lastdb version

# 4. Functional smoke test
lastseek schema-table
lastseek status
# Both should complete without error
```

### Version Constraint Verification

Test the version constraint enforcement:

```bash
# On LastDB 0.23.x or 0.24.x (within range)
lastdb app install lastseek
# Expected: Install succeeds

# On LastDB 0.22.x (below minimum)
lastdb app install lastseek
# Expected: Error - "lastseek requires lastdb >=0.23.0, but you have 0.22.x"

# On LastDB 0.25.x+ (above maximum)
lastdb app install lastseek
# Expected: Error - "lastseek requires lastdb <0.25.0, but you have 0.25.x"
```

## CI Build Evidence

**Platform:** macOS runner (darwin-arm64)  
**Workflow:** `.github/workflows/ci-required.yml` → `publish` job  
**Trigger:** Push to `main` after `ci-required` passes  

The workflow:
1. Runs `.lastgit/artifact-release.sh`
2. Builds with `--features neural` (downloads ONNX Runtime, ~11.5 min)
3. Validates binary runs and contains neural features
4. Publishes artifact via `host-track-artifact.yml`

**Evidence location:** GitHub Actions artifacts for main branch builds  
**Artifact name:** `ht-artifact-<commit-sha>`

## Cross-Repo Changes Needed

### 1. EdgeVector/last-stack

The daily registry publisher needs to include `lastseek` in its app list.

**File:** `bin/last-stack-registry-publish-next` or equivalent  
**Change:** Add `lastseek` to the apps processed for registry inclusion

**Current apps in registry:** brain, dogfood-graph, kanban, lastdb-browser, lastsecrets, org, routines, search, situations

### 2. EdgeVector/homebrew-lastdb

No direct changes needed. The registry (`registry/stable.json`) is updated by the last-stack publisher when it runs.

**Verification after publishing:**
```bash
# Check registry contains lastseek
curl https://raw.githubusercontent.com/EdgeVector/homebrew-lastdb/main/registry/stable.json \
  | jq '.apps[] | select(.app_id == "lastseek")'
```

Expected structure:
```json
{
  "app_id": "lastseek",
  "source": "https://github.com/EdgeVector/lastseek.git",
  "compat": [
    {
      "app_version": "0.1.1",
      "sha": "<commit-sha>",
      "lastdb_version": "<tested-version>",
      "proved_at": "<timestamp>",
      "proof_run": "<proof-run-id>"
    }
  ],
  "description": "Memory-mapped semantic vector search for LastDB..."
}
```

## References

- Brain packaging: EdgeVector/brain `folddb.toml`, `.lastgit/artifacts.json`
- Registry structure: EdgeVector/homebrew-lastdb `registry/stable.json`
- Version constraints: EdgeVector/lastdb PR #353
- Publishing workflow: EdgeVector/last-stack `bin/last-stack-registry-publish-next`
