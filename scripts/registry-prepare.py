#!/usr/bin/env python3
"""Put the feature-complete binary at dist/lastseek for the app registry install.

The installer clones this repository at the commit its signed listing names and
runs this file in the clone. It needs no GitHub login and no Rust toolchain:

1. Download the package that CI built from this exact commit. It is a public
   GitHub Release asset under the tag `app-<commit>`.
2. Check the package against its provenance record and against this checkout
   (commit, tree, platform, and the size, mode and SHA-256 of every file).
3. Copy `dist/lastseek` out of it.

When no package exists for this commit and platform, build from the locked
source instead. That needs `cargo`, and a cold neural build can exceed the
installer's 600 second step limit.

The package is trusted as far as GitHub Releases of the source repository are:
the listing signs the commit, and CI on `main` writes the release for it.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile

PLATFORMS = {("Darwin", "arm64"): "darwin-arm64", ("Linux", "x86_64"): "linux-x86_64"}
TAR_CAP = 268435456
UNPACK_CAP = 536870912


class Unavailable(Exception):
    """No usable package; the source build is the remaining path."""


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, timeout=60).stdout.decode().strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(url, cap):
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "download"
        result = subprocess.run(
            ["curl", "--disable", "--fail", "--silent", "--location", "--proto", "=https",
             "--max-time", "300", "--max-filesize", str(cap), "--output", str(target), url],
            capture_output=True, timeout=330)
        if result.returncode != 0 or not target.is_file():
            raise Unavailable(f"no download at {url} (curl exit {result.returncode})")
        data = target.read_bytes()
    if len(data) > cap:
        raise Unavailable("package exceeds the download cap")
    return data


def read_package(app, repo, sha, target):
    """Return (provenance bytes, tar bytes) from a local directory or the release."""
    names = (f"{app}-{target}.provenance.json", f"{app}-{target}.tar.gz")
    # The installer clears the environment, so this variable is never set
    # there. CI sets it to prove the package it just built.
    local = os.environ.get("LASTSEEK_PACKAGE_DIR")
    if local:
        try:
            return tuple((Path(local) / name).read_bytes() for name in names)
        except OSError as error:
            raise Unavailable(f"LASTSEEK_PACKAGE_DIR has no package: {error}") from None
    base = f"https://github.com/{repo}/releases/download/app-{sha}"
    return download(f"{base}/{names[0]}", 1048576), download(f"{base}/{names[1]}", TAR_CAP)


def install_package(root, app, repo, sha, target):
    raw_proof, archive = read_package(app, repo, sha, target)
    try:
        proof = json.loads(raw_proof)
        expected = (app, repo, sha, git("rev-parse", "HEAD^{tree}"), target)
        actual = tuple(proof.get(key) for key in ("app", "repo", "source_oid", "tree_oid", "platform"))
        if actual != expected or proof["tar_sha256"] != digest(archive):
            raise Unavailable("package provenance differs from this checkout")
        files = {entry["path"]: entry for entry in proof["files"]}
        if len(files) != len(proof["files"]) or "dist/" + app not in files or len(files) > 64:
            raise Unavailable("package file list is invalid")
        staged = {}
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            members = tar.getmembers()
            if sum(member.size for member in members) > UNPACK_CAP:
                raise Unavailable("package exceeds the unpack cap")
            for member in members:
                path = Path(member.name)
                record = files.get(member.name)
                if (not member.isfile() or path.is_absolute() or ".." in path.parts
                        or member.name in staged or record is None):
                    raise Unavailable("package has an unsafe, repeated or unlisted member")
                content = tar.extractfile(member).read()
                if (len(content) != record["size"] or digest(content) != record["sha256"]
                        or member.mode != record["mode"]):
                    raise Unavailable("package member differs from its provenance")
                staged[member.name] = (content, member.mode)
        if set(staged) != set(files):
            raise Unavailable("package is missing a listed member")
    except (KeyError, TypeError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        raise Unavailable(f"package could not be verified ({type(error).__name__})") from None
    for name, (content, mode) in staged.items():
        if Path(name).parts[0] == "dist":
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)
    print(f"Prepared {app} from the verified {target} package of {sha}")


def build_from_source(root, app, reason):
    print(f"No verified package for this commit ({reason}); building from the locked source")
    if shutil.which("cargo") is None:
        raise SystemExit(
            f"{app}: no prebuilt package is available for this commit and platform, and the source "
            "build needs the Rust toolchain (cargo), which is not on PATH")
    subprocess.run(["cargo", "build", "--locked", "--release", "--features", "neural"], check=True)
    target = root / "dist" / app
    target.parent.mkdir(exist_ok=True)
    built = Path(os.environ.get("CARGO_TARGET_DIR", "target")) / "release" / app
    target.write_bytes((root / built).read_bytes())
    target.chmod(0o755)


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    manifest = json.loads((root / "lastdb-app.json").read_text())
    app = manifest["app_id"]
    repo = manifest["source"].removeprefix("https://github.com/").removesuffix(".git")
    target = PLATFORMS.get((platform.system(), platform.machine()))
    try:
        if target is None:
            raise Unavailable(f"no package is built for {platform.system()} {platform.machine()}")
        install_package(root, app, repo, git("rev-parse", "HEAD"), target)
    except Unavailable as reason:
        build_from_source(root, app, reason)


if __name__ == "__main__":
    main()
