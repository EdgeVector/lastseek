#!/usr/bin/env python3
"""Run the LastSeek app commands against a running LastDB node.

The node is a fresh generated home that the `lastdb-node` CI action started.
This script declares two schemas, writes records, and then runs
`lastseek index`, `lastseek status` and `lastseek search`. It fails when the
first index does not return at once with `building`, when status does not
reach `ready`, when a search misses an indexed record, when a deleted record
still answers, when a record of a schema nobody named appears, or when a
second index of the same schemas rebuilds anything.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

NAMED, UNNAMED = "SeekNote", "SeekPrivate"
APP = "harbour"
FIELDS = ["title", "body"]
INDEX_RETURN_S = 30
READY_DEADLINE_S = 600
FAILED_GRACE_S = 15


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost", timeout=60)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(60)
        self.sock.connect(self.path)


def fail(message: str) -> None:
    raise SystemExit(f"lastseek node check failed: {message}")


def call(node: str, path: str, body: dict) -> dict:
    connection = UnixConnection(node)
    try:
        connection.request("POST", path, json.dumps(body),
                           {"Content-Type": "application/json", "X-LastDB-Client": "lastseek-ci"})
        reply = connection.getresponse()
        raw = reply.read()
    finally:
        connection.close()
    if reply.status != 200:
        fail(f"{path} answered HTTP {reply.status}: {raw[:400]!r}")
    return json.loads(raw)


def declare(node: str, name: str) -> None:
    call(node, "/api/schemas/seed-system", {"isolated_copy": True, "schema": {
        "source": "system_seed", "name": name, "descriptive_name": name, "schema_type": "Hash",
        "record_keys": {"primary": {"kind": "hash"}, "alternates": {}},
        "fields": FIELDS, "field_types": dict.fromkeys(FIELDS, "String"),
        "field_descriptions": {field: f"LastSeek CI {field}" for field in FIELDS},
        "field_versions": dict.fromkeys(FIELDS, 1),
    }})


def record(schema: str, key: str, title: str, body: str) -> dict:
    return {"schema": schema, "key": {"hash": key}, "record": {"title": title, "body": body}}


def write(node: str, *mutations: dict) -> None:
    reply = call(node, "/v1/write", {"mutations": list(mutations)})
    if reply.get("written") != len(mutations):
        fail(f"a write of {len(mutations)} records reported {reply!r}")


def revision(node: str, key: str) -> str:
    result = call(node, "/v1/read", {"reads": [{"schema": NAMED, "key": {"hash": key}}]})["results"][0]
    records = result.get("records") or []
    if len(records) != 1:
        fail(f"the read of {key} returned {result!r}")
    return records[0]["revision"]


class LastSeek:
    def __init__(self, program: Path, home: Path, node: str, model: Path):
        self.program = program
        self.home = home
        self.app_dir = home / "apps" / APP
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(("LASTSEEK", "LASTDB"))}
        # LASTDB_APP_DATA points somewhere else on purpose. The app index must
        # stay under LASTSEEK_HOME/apps and must not follow the caller's data.
        self.environment.update(
            LASTSEEK_HOME=str(home),
            LASTSEEK_MODEL_DIR=str(model),
            LASTSEEK_EMBEDDER="bge",
            LASTDB_SOCKET_PATH=node,
            LASTDB_APP_DATA=str(home.parent / "lastdb-app-data"),
        )

    def run(self, *arguments: str, check: bool = True, timeout: float = 900):
        started = time.monotonic()
        done = subprocess.run(
            [str(self.program), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        reply = None
        if done.stdout.strip():
            try:
                reply = json.loads(done.stdout)
            except json.JSONDecodeError:
                if check:
                    fail(f"lastseek {arguments[0]} stdout was not one JSON object: "
                         f"{done.stdout[:400]!r}\n{done.stderr[-400:]}")
        if check and done.returncode != 0:
            fail(f"lastseek {' '.join(arguments)} exited {done.returncode} "
                 f"after {elapsed:.1f}s: {done.stderr[-800:]}\nstdout {done.stdout[:400]!r}")
        if check and not isinstance(reply, dict):
            fail(f"lastseek {arguments[0]} printed no JSON object: {done.stdout[:400]!r}")
        return done, reply, elapsed


def expect_first(seek: LastSeek, text: str, key: str) -> None:
    found = seek_keys(seek, text)
    if not found or found[0] != key:
        fail(f"the search {text!r} should rank {key} first, got {found}")
    if "vault" in found:
        fail("a record of a schema nobody named is in the index")


def seek_keys(seek: LastSeek, text: str) -> list[str]:
    _done, reply, _elapsed = seek.run(
        "search", "--app", APP, text, "--schema", NAMED, "--k", "5")
    return [hit["key_hash"] for hit in reply["results"]]


def wait_ready(seek: LastSeek) -> dict:
    deadline = time.monotonic() + READY_DEADLINE_S
    failed_since = None
    last = None
    while True:
        _done, reply, _elapsed = seek.run("status", "--app", APP, timeout=60)
        last = reply
        state = reply.get("state")
        if state == "ready":
            return reply
        now = time.monotonic()
        if state not in ("building", "failed"):
            fail(f"status left the build path: {reply!r}")
        if state == "failed":
            failed_since = failed_since or now
            if now - failed_since > FAILED_GRACE_S:
                log = seek.app_dir / "index.log"
                tail = log.read_text(errors="replace")[-1500:] if log.exists() else ""
                fail(f"the build failed: {reply!r}\n{tail}")
        else:
            failed_since = None
        if now > deadline:
            fail(f"status did not reach ready within {READY_DEADLINE_S}s, last {last!r}")
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lastseek", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--socket", default=os.environ.get("LASTDB_SOCKET_PATH"))
    arguments = parser.parse_args()
    if not arguments.socket:
        fail("pass --socket or set LASTDB_SOCKET_PATH")
    node = arguments.socket

    declare(node, NAMED)
    declare(node, UNNAMED)
    write(node,
          {**record(NAMED, "tides", "Tide tables", "High and low tide times for the harbour, with the phase of the moon."), "op": "create"},
          {**record(NAMED, "bread", "Sourdough loaf", "Feed the starter with flour and water, let the dough rise overnight, then bake the loaf."), "op": "create"},
          {**record(NAMED, "bike", "Bicycle repair", "Patch the puncture, replace the inner tube and pump up the tyre."), "op": "create"})
    write(node, {**record(UNNAMED, "vault", "Sourdough loaf", "Bake the loaf of bread from the sourdough starter."), "op": "create"})

    with tempfile.TemporaryDirectory(prefix="lastseek-ci-") as scratch:
        home = Path(scratch) / "lastseek-home"
        home.mkdir()
        seek = LastSeek(arguments.lastseek.resolve(), home, node, arguments.model_dir.resolve())

        never, never_reply, _elapsed = seek.run(
            "search", "--app", "never", "how do I bake bread", check=False, timeout=60)
        if never.returncode != 3 or not isinstance(never_reply, dict) \
                or never_reply.get("code") != "index_not_ready" \
                or never_reply.get("ok") is not False:
            fail(f"a search of an app that was never indexed should exit 3 with "
                 f"index_not_ready, got {never.returncode} {never.stdout!r} {never.stderr[-400:]}")
        if (home / "apps" / "never").exists():
            fail("search created a directory for an app that was never indexed")

        try:
            _done, started, elapsed = seek.run(
                "index", "--app", APP, "--schema", NAMED, timeout=INDEX_RETURN_S)
        except subprocess.TimeoutExpired:
            fail(f"index did not return within {INDEX_RETURN_S}s")
        if elapsed >= INDEX_RETURN_S or started.get("ok") is not True or started.get("state") != "building":
            fail(f"index should return at once with building, got {elapsed:.1f}s {started!r}")
        app_data_index = home.parent / "lastdb-app-data"
        if app_data_index.exists():
            fail("LASTDB_APP_DATA moved the app index")
        if not str(seek.app_dir).startswith(str(home)):
            fail(f"the app index is not under LASTSEEK_HOME: {seek.app_dir}")

        ready = wait_ready(seek)
        if ready.get("records") != 3 or ready.get("schemas") != [NAMED]:
            fail(f"status should report the three {NAMED} records, got {ready!r}")
        if not ready.get("applied_through"):
            fail(f"a ready index should name applied_through, got {ready!r}")
        print(f"index: {ready['records']} records through {ready['applied_through']}", flush=True)

        state_path = seek.app_dir / "index-state.json"
        log_path = seek.app_dir / "index.log"
        before_state = state_path.read_bytes()
        before_log = log_path.stat().st_size
        try:
            _done, again, again_elapsed = seek.run(
                "index", "--app", APP, "--schema", NAMED, timeout=INDEX_RETURN_S)
        except subprocess.TimeoutExpired:
            fail(f"the second index did not return within {INDEX_RETURN_S}s")
        if again.get("state") != "ready" or again.get("ok") is not True:
            fail(f"the second index of the same schemas should be ready, got {again!r}")
        if state_path.read_bytes() != before_state or log_path.stat().st_size != before_log:
            fail("the second index rebuilt: index-state.json or index.log changed")
        print(f"index: same schemas stayed ready in {again_elapsed:.1f}s and rebuilt nothing",
              flush=True)

        expect_first(seek, "how do I bake bread at home", "bread")
        expect_first(seek, "fix a flat tyre on my bike", "bike")

        write(node,
              {**record(NAMED, "kite", "Kite flying", "Launch the kite into a steady wind on the beach and let out the string."), "op": "create"},
              {**record(NAMED, "bike", "Telescope", "Point the telescope at Saturn on a clear night to see its rings."),
               "op": "replace", "revision": revision(node, "bike")},
              {"schema": NAMED, "op": "delete", "key": {"hash": "tides"}, "revision": revision(node, "tides")})
        expect_first(seek, "flying a kite in the wind", "kite")
        expect_first(seek, "looking at planets through a telescope", "bike")
        if "tides" in seek_keys(seek, "tide times at the harbour"):
            fail("a deleted record still answers a search")
        caught_up = wait_ready(seek)
        if caught_up.get("records") != 3 or caught_up.get("schemas") != [NAMED]:
            fail(f"after create, replace and delete the index should still hold three "
                 f"{NAMED} records, got {caught_up!r}")
        print(f"search: create, replace and delete are visible through {caught_up['applied_through']}",
              flush=True)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write("LastSeek index, search and status passed on the node above. "
                         "A schema nobody named stayed out of the index.\n")
    print("lastseek node check passed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
