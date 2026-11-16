#!/usr/bin/env python3
"""Pack the built binary and its provenance record for one platform.

CI runs this after `.lastgit/artifact-release.sh`. The two output files become
GitHub Release assets under the tag `app-<commit>`, where
`scripts/registry-prepare.py` finds them on an install.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

MEMBERS = ("dist/lastseek", "lastseek.app.json", "LICENSE", "THIRD_PARTY_NOTICES")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=60).stdout.decode().strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=("darwin-arm64", "linux-x86_64"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "lastdb-app.json").read_text())
    app = manifest["app_id"]
    repo = manifest["source"].removeprefix("https://github.com/").removesuffix(".git")

    files = []
    buffer = io.BytesIO()
    # Fixed metadata, so the same tree packs to the same bytes.
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in MEMBERS:
            content = (root / name).read_bytes()
            mode = 0o755 if name.startswith("dist/") else 0o644
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(content), mode, 0
            tar.addfile(info, io.BytesIO(content))
            files.append({"path": name, "size": len(content), "mode": mode,
                          "sha256": hashlib.sha256(content).hexdigest()})
    archive = gzip.compress(buffer.getvalue(), mtime=0)
    proof = {"app": app, "repo": repo, "platform": args.platform,
             "source_oid": git("-C", str(root), "rev-parse", "HEAD"),
             "tree_oid": git("-C", str(root), "rev-parse", "HEAD^{tree}"),
             "tar_sha256": hashlib.sha256(archive).hexdigest(), "files": files}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{app}-{args.platform}.tar.gz").write_bytes(archive)
    (args.out / f"{app}-{args.platform}.provenance.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(f'Packed {app} {args.platform} at {proof["source_oid"]}: {len(archive)} bytes, sha256 {proof["tar_sha256"]}')


if __name__ == "__main__":
    main()
