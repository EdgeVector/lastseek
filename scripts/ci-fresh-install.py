#!/usr/bin/env python3
"""Run the new-user LastSeek path on a CI runner, in a fresh home, on generated data.

What a new user does, in order:

1. Install the stable LastDB release and start `lastdbd` with no flags.
2. `lastdb app install brain`, `brain init`, write a few records.
3. Install LastSeek. Nothing else: no model download by hand, no rebuild, no
   timer, no service.
4. `brain search`. The first search builds the index. A record written later
   is found by the next search.

`--mode package` installs LastSeek from this checkout with the app's own
install plan (prepare, probe, links), in the environment the installer gives
those steps, using the package the same CI run built. It is the proof a pull
request can give, because a listing names a merged commit.
`--mode registry` runs the real `lastdb app install lastseek --channel C`.

The home is generated and removed. No existing database is read.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import time
import urllib.request

TAP_RAW = "https://raw.githubusercontent.com/EdgeVector/homebrew-lastdb/main/Formula/lastdb.rb"
TAP_RELEASE = "https://github.com/EdgeVector/homebrew-lastdb/releases/download"
LINUX_ASSET = "lastdb-x86_64-unknown-linux-gnu.tar.gz"
RECORDS = {
    "ci-cats": "Felines spend most of the day asleep, often sixteen hours.",
    "ci-bread": "Sourdough loaves need a long, slow fermentation before baking.",
    "ci-probe": "The spacecraft used a gravity assist from Jupiter to reach Saturn.",
}
# No word of a query appears in its record, so only a semantic match finds it.
FIRST_QUERY, FIRST_SLUG = "how long do cats nap", "ci-cats"
LATER_SLUG = "ci-tides"
LATER_RECORD = "The pull of the moon raises the oceans twice a day."
LATER_QUERY = "why does the sea level go up and down"


def fetch(url, cap=64 * 1024 ** 2):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise RuntimeError(f"{url} exceeds its size bound")
    return data


class Run:
    def __init__(self, work, mode, channel, package_dir):
        self.work, self.mode, self.channel, self.package_dir = work, mode, channel, package_dir
        self.home = work / "home"
        self.source = Path(__file__).resolve().parents[1]
        self.facts = []
        self.daemon = None

    def note(self, text):
        print(f"== {text}", flush=True)
        self.facts.append(text)

    def call(self, argv, env=None, body=None, timeout=600, check=True, cwd=None):
        result = subprocess.run([str(a) for a in argv], env=env or self.env, input=body, text=True,
                                capture_output=True, timeout=timeout, cwd=cwd)
        if check and result.returncode != 0:
            print(result.stdout[-4000:], result.stderr[-4000:], sep="\n", flush=True)
            raise RuntimeError(f"{argv[0]} {argv[1] if len(argv) > 1 else ''} exited {result.returncode}")
        return result

    def install_lastdb(self):
        formula = fetch(TAP_RAW).decode()
        version = next(line.split('"')[1] for line in formula.splitlines() if line.startswith("  version "))
        if platform.system() == "Darwin":
            # The public install command, with the runner's own home.
            subprocess.run(["brew", "install", "edgevector/lastdb/lastdb"], check=True, timeout=900)
            prefix = subprocess.run(["brew", "--prefix"], check=True, capture_output=True, text=True).stdout.strip()
            binaries = Path(prefix) / "bin"
        else:
            # Homebrew has no Linux build. Use the same stable release's Linux asset.
            archive = fetch(f"{TAP_RELEASE}/v{version}/{LINUX_ASSET}")
            sums = fetch(f"{TAP_RELEASE}/v{version}/SHA256SUMS.txt").decode()
            if f"{hashlib.sha256(archive).hexdigest()}  {LINUX_ASSET}" not in sums.replace("*", ""):
                raise RuntimeError("the stable Linux asset differs from SHA256SUMS.txt")
            unpacked = self.work / "lastdb"
            unpacked.mkdir()
            (self.work / LINUX_ASSET).write_bytes(archive)
            with tarfile.open(self.work / LINUX_ASSET) as tar:
                tar.extractall(unpacked, filter="data")
            binaries = next(path.parent for path in sorted(unpacked.rglob("lastdbd")) if path.is_file())
        self.home.mkdir(parents=True)
        path = os.pathsep.join((str(self.home / ".local/bin"), str(binaries), os.environ["PATH"]))
        # Everything after this point sees only the generated home.
        self.env = {**{k: v for k, v in os.environ.items() if not k.startswith(("GH_", "GITHUB_TOKEN", "ACTIONS_", "LASTSEEK_", "LASTDB_"))},
                    "HOME": str(self.home), "PATH": path}
        self.note(f"LastDB stable formula version {version}; "
                  f"`lastdb --version`: {self.call(['lastdb', '--version']).stdout.strip()}")

    def start_daemon(self):
        log = (self.work / "lastdbd.log").open("w")
        # The Homebrew service runs `lastdbd` with LASTDB_HOME and no flags.
        self.daemon = subprocess.Popen(["lastdbd"], env={**self.env, "LASTDB_HOME": str(self.home / ".lastdb")},
                                       stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.daemon.poll() is not None:
                raise RuntimeError("lastdbd exited during startup")
            if self.call(["lastdb", "status"], check=False, timeout=30).returncode == 0:
                version = self.socket("/api/version")
                self.note(f"lastdbd serves a fresh home; API {version.get('api_version')}, "
                          f"build {version.get('build') or version.get('version')}")
                return
            time.sleep(1)
        raise RuntimeError("lastdbd did not serve within 90 seconds")

    def socket(self, route):
        out = self.call(["curl", "--silent", "--fail", "--max-time", "20", "--unix-socket",
                         self.home / ".lastdb/data/folddb.sock", "http://localhost" + route]).stdout
        return json.loads(out)

    def install_brain(self):
        self.call(["lastdb", "app", "install", "brain", "--accept-install"], timeout=900)
        self.call(["brain", "init"], timeout=300)
        for slug, text in RECORDS.items():
            self.call(["brain", "put", slug, "--type", "concept"], body=text, timeout=120)
        self.note(f"Brain installed from the stable channel; {len(RECORDS)} generated records written")

    def install_lastseek(self):
        if self.mode == "registry":
            self.call(["lastdb", "app", "install", "lastseek", "--channel", self.channel, "--accept-install"],
                      timeout=1500)
            receipt = json.loads((self.home / ".local/share/lastdb-apps/lastseek/current/lastdb-app-install.json").read_text())
            self.note(f"`lastdb app install lastseek --channel {self.channel}` installed {receipt['sha']} "
                      f"(listing API {receipt['listing']['api_version']})")
            return
        plan = json.loads((self.source / "lastdb-app.json").read_text())["install"]
        data = self.home / ".lastdb"
        # The installer clears the environment and passes these names.
        step_env = {name: self.env[name] for name in ("HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LANG")
                    if name in self.env}
        step_env.update(LASTDB_HOME=str(data), LASTDB_DATA_DIR=str(data / "data"),
                        LASTDB_SOCKET=str(data / "data/folddb.sock"), LASTDB_APP_ID="lastseek",
                        LASTDB_APP_SOURCE_DIR=str(self.source), LASTSEEK_PACKAGE_DIR=str(self.package_dir))
        output = ""
        for phase in ("prepare", "probe"):
            for step in plan[phase]:
                result = self.call(step["argv"], env=step_env, cwd=self.source, timeout=step.get("timeout_s", 30))
                print(result.stdout, result.stderr, sep="", flush=True)
                output += result.stdout
        if "from the verified" not in output:
            raise RuntimeError("prepare did not use the verified package")
        for link in plan["links"]:
            target = self.home / ".local/bin" / link["command"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to(self.source / link["source"])
        sha = self.call(["git", "-C", self.source, "rev-parse", "HEAD"]).stdout.strip()
        self.note(f"LastSeek {sha} installed with its own install plan from the package this run built; "
                  "the model came from the pinned download")

    def status(self):
        result = self.call(["lastseek", "status"], check=False, timeout=60)
        return json.loads(result.stdout) if result.returncode == 0 else {}

    def lastseek_top(self, query):
        result = self.call(["lastseek", "query", query, "--k", "3", "--schema", "brain/Concept"],
                           check=False, timeout=60)
        if result.returncode != 0:
            return None, result.stderr.strip()[-300:]
        rows = json.loads(result.stdout)["results"]
        return (rows[0]["key_hash"] if rows else None), ""

    def brain_finds(self, query, slug):
        return slug in self.call(["brain", "search", query, "--json"], check=False, timeout=120).stdout

    def search(self):
        if self.status().get("committed_rows", 0) or self.status().get("source") == "lastdb-change-feed":
            raise RuntimeError("an index exists before the first search")
        started = time.monotonic()
        found = self.brain_finds(FIRST_QUERY, FIRST_SLUG)
        first = time.monotonic() - started
        deadline = time.monotonic() + 120
        while not found and time.monotonic() < deadline:
            time.sleep(2)
            found = self.brain_finds(FIRST_QUERY, FIRST_SLUG)
        status = self.status()
        if not found or status.get("source") != "lastdb-change-feed" or status.get("query_count", 0) < 1:
            raise RuntimeError(f"Brain search did not answer from a LastSeek index: {json.dumps(status)[:1500]}")
        rows = status.get("committed_rows", 0) + status.get("pending_ops", 0)
        if rows < len(RECORDS):
            raise RuntimeError("the first index holds fewer vectors than the records written")
        top, error = self.lastseek_top(FIRST_QUERY)
        if top != FIRST_SLUG:
            raise RuntimeError(f"LastSeek ranked {top!r} first for the first query {error}")
        self.note(f"First `brain search` ran {first:.1f}s and built the index with no other command; "
                  f"`{FIRST_QUERY}` returns `{FIRST_SLUG}` ({rows} vectors, embedder {status.get('embedder_id')}, "
                  f"{status.get('query_count')} LastSeek queries recorded)")

        self.call(["brain", "put", LATER_SLUG, "--type", "concept"], body=LATER_RECORD, timeout=120)
        top, error = self.lastseek_top(LATER_QUERY)
        if top != LATER_SLUG:
            raise RuntimeError(f"a record written after the build is not found by the next query: {top!r} {error}")
        if not self.brain_finds(LATER_QUERY, LATER_SLUG):
            raise RuntimeError("Brain search does not return the record written after the build")
        status = self.status()
        agents = [path.name for path in (self.home / "Library/LaunchAgents").glob("*lastseek*")] \
            if (self.home / "Library/LaunchAgents").is_dir() else []
        if agents:
            raise RuntimeError(f"a LastSeek timer exists: {agents}")
        self.note(f"A record written after the build is first for `{LATER_QUERY}` on the next query; "
                  f"no LastSeek timer or service exists (applied through {status.get('applied_through')}, "
                  f"recovery {json.dumps(status.get('online_recovery'))})")

    def stop(self):
        if self.daemon and self.daemon.poll() is None:
            self.daemon.terminate()
            try:
                self.daemon.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.daemon.kill()
        if platform.system() == "Darwin":
            # The first app install loads the app updater for this generated home.
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}/com.edgevector.lastdb-app-registry"],
                           capture_output=True, check=False)
        shutil.rmtree(self.home, ignore_errors=True)
        self.facts.append(f"Daemon stopped; generated home removed: {not self.home.exists()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="empty directory on the runner")
    parser.add_argument("--mode", choices=("package", "registry"), required=True)
    parser.add_argument("--channel", choices=("next", "stable"), default="next")
    parser.add_argument("--package-dir", type=Path, help="package built by this run (package mode)")
    args = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("this runs on a GitHub Actions runner only; it installs LastDB and starts a daemon")
    if args.mode == "package" and args.package_dir is None:
        raise SystemExit("--mode package needs --package-dir")
    args.work.mkdir(parents=True, exist_ok=True)
    run = Run(args.work.resolve(), args.mode, args.channel, args.package_dir and args.package_dir.resolve())
    failure = None
    try:
        run.install_lastdb()
        run.start_daemon()
        run.install_brain()
        run.install_lastseek()
        run.search()
    except Exception as error:  # the summary must record a failed run too
        failure = error
        log = args.work / "lastdbd.log"
        if log.is_file():
            print(log.read_text()[-3000:], flush=True)
    finally:
        run.stop()
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        result = "FAILED: " + str(failure) if failure else "passed"
        summary.write(f"### LastSeek fresh install ({platform.system()} {platform.machine()}, {args.mode}): {result}\n\n")
        summary.writelines(f"- {fact}\n" for fact in run.facts)
        summary.write("\n")
    if failure:
        raise SystemExit(f"fresh install failed: {failure}")


if __name__ == "__main__":
    main()
