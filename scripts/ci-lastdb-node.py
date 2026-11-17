#!/usr/bin/env python3
"""Run the real LastSeek commands against a running LastDB node.

The node is a fresh generated home that the `lastdb-node` CI action started.
This script declares two schemas, writes records, and then runs
`lastseek rebuild`, `lastseek drain` and `lastseek query` the way a user does.
It fails when an indexed record is not found, when a deleted record still
answers, or when a record of a schema nobody named appears.
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

NAMED, UNNAMED = "SeekNote", "SeekPrivate"
FIELDS = ["title", "body"]


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
        self.program, self.home, self.node = program, home, node
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith(("LASTSEEK", "LASTDB"))}
        self.environment.update(LASTSEEK_MODEL_DIR=str(model), LASTSEEK_EMBEDDER="bge")

    def run(self, *arguments: str) -> dict:
        done = subprocess.run([str(self.program), *arguments, "--home", str(self.home)],
                              env=self.environment, capture_output=True, text=True, timeout=900)
        if done.returncode != 0:
            fail(f"lastseek {arguments[0]} exited {done.returncode}: {done.stderr[-800:]}")
        return json.loads(done.stdout)

    def keys(self, text: str) -> list[str]:
        reply = self.run("query", text, "--schema", NAMED, "--k", "5")
        return [hit["key_hash"] for hit in reply["results"]]


def expect_first(seek: LastSeek, text: str, key: str) -> None:
    found = seek.keys(text)
    if not found or found[0] != key:
        fail(f"the query {text!r} should rank {key} first, got {found}")
    if "vault" in found:
        fail("a record of a schema nobody named is in the index")


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
        seek = LastSeek(arguments.lastseek.resolve(), Path(scratch) / "index", node, arguments.model_dir.resolve())
        built = seek.run("rebuild", "--lastdb-socket", node, "--schema", NAMED)
        if built["indexed"]["records"] != 3 or built["schemas"] != [NAMED]:
            fail(f"the rebuild should index the three {NAMED} records, got {built!r}")
        expect_first(seek, "how do I bake bread at home", "bread")
        expect_first(seek, "fix a flat tyre on my bike", "bike")
        print(f"rebuild: indexed {built['indexed']['records']} records through {built['applied_through']}", flush=True)

        write(node,
              {**record(NAMED, "kite", "Kite flying", "Launch the kite into a steady wind on the beach and let out the string."), "op": "create"},
              {**record(NAMED, "bike", "Telescope", "Point the telescope at Saturn on a clear night to see its rings."),
               "op": "replace", "revision": revision(node, "bike")},
              {"schema": NAMED, "op": "delete", "key": {"hash": "tides"}, "revision": revision(node, "tides")})
        drained = seek.run("drain", "--lastdb-socket", node, "--max-files", "500")
        if drained["records"] != 2 or drained["tombstones"] != 1 or drained["has_more"]:
            fail(f"the drain should apply two records and one deletion, got {drained!r}")
        expect_first(seek, "flying a kite in the wind", "kite")
        expect_first(seek, "looking at planets through a telescope", "bike")
        if "tides" in seek.keys("tide times at the harbour"):
            fail("a deleted record still answers a query")
        print(f"drain: applied {drained['records']} records and {drained['tombstones']} deletion "
              f"through {drained['applied_through']}", flush=True)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write("LastSeek rebuild, drain and query passed on the node above. "
                         "A schema nobody named stayed out of the index.\n")
    print("lastseek node check passed", flush=True)


if __name__ == "__main__":
    sys.exit(main())
