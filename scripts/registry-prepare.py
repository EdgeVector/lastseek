#!/usr/bin/env python3
"""Prefer a verified exact-source CI package, falling back to a locked source build."""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import zipfile


def command(args):
    result = subprocess.run(args, check=True, capture_output=True, timeout=90)
    return result.stdout


def api(path):
    return json.loads(command(["gh", "api", path]))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def package(root, app, repo, sha):
    runs = api(f"repos/{repo}/actions/workflows/ci-required.yml/runs?head_sha={sha}&branch=main&event=push&per_page=100")["workflow_runs"]
    runs = {r["id"] for r in runs if r["head_sha"] == sha and r["head_branch"] == "main" and r["event"] == "push"
            and r["status"] == "completed" and r["conclusion"] == "success"}
    artifacts = api(f"repos/{repo}/actions/artifacts?name=registry-package-{sha}&per_page=100")["artifacts"]
    for artifact in artifacts:
        if artifact["expired"] or artifact["workflow_run"]["id"] not in runs:
            continue
        if artifact["size_in_bytes"] > 268435456:
            raise RuntimeError("CI package exceeds download cap")
        data = command(["gh", "api", f"repos/{repo}/actions/artifacts/{artifact['id']}/zip"])
        if len(data) > 268435456 or artifact.get("digest") != "sha256:" + digest(data):
            raise RuntimeError("CI package digest mismatch")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 536870912:
                raise RuntimeError("CI archive exceeds unpack cap")
            proof = json.loads(archive.read(f"{app}/provenance.json"))
            tar = archive.read(f"{app}/bundle.tar")
        tree = command(["git", "rev-parse", "HEAD^{tree}"]).decode().strip()
        expected = (app, repo, sha, tree, "darwin-arm64")
        actual = tuple(proof.get(k) for k in ("app", "repo", "source_oid", "tree_oid", "platform"))
        if actual != expected or proof["tar_sha256"] != digest(tar):
            raise RuntimeError("CI package provenance mismatch")
        files = {f["path"]: f for f in proof["files"]}
        if len(files) != len(proof["files"]) or len(files) > 20000:
            raise RuntimeError("CI package file list is invalid")
        seen = set()
        with tarfile.open(fileobj=io.BytesIO(tar), mode="r:") as archive:
            members = archive.getmembers()
            if sum(m.size for m in members) > 536870912:
                raise RuntimeError("CI tar exceeds unpack cap")
            for member in members:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts or member.name in seen:
                    raise RuntimeError("Unsafe or duplicate CI package path")
                record = files.get(member.name)
                if record is None:
                    raise RuntimeError("Unexpected CI package member")
                content = archive.extractfile(member).read()
                if len(content) != record["size"] or digest(content) != record["sha256"] or member.mode != record["mode"]:
                    raise RuntimeError("CI package member digest mismatch")
                seen.add(member.name)
                if path.parts[0] == "dist":
                    target = root / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    target.chmod(member.mode)
        if seen != set(files):
            raise RuntimeError("Missing CI package member")
        print(f"Prepared {app} from verified CI package at {sha}")
        return True
    return False


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    manifest = json.loads((root / "lastdb-app.json").read_text())
    app = manifest["app_id"]
    repo = manifest["source"].removeprefix("https://github.com/").removesuffix(".git")
    sha = command(["git", "rev-parse", "HEAD"]).decode().strip()
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            if package(root, app, repo, sha):
                return
        except (OSError, subprocess.SubprocessError, KeyError, ValueError, RuntimeError, zipfile.BadZipFile, tarfile.TarError):
            print("Exact-source CI package unavailable; using the locked source build")
    args = ["cargo", "build", "--locked", "--release"]
    if app == "lastseek":
        args += ["--features", "neural"]
    subprocess.run(args, check=True)
    target = root / "dist" / app
    target.parent.mkdir(exist_ok=True)
    target.write_bytes((root / os.environ.get("CARGO_TARGET_DIR", "target") / "release" / app).read_bytes())
    target.chmod(0o755)


if __name__ == "__main__":
    main()
