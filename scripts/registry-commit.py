#!/usr/bin/env python3
"""Record this main commit through short-lived Actions identity; never log tokens."""
import json
import os
import urllib.parse
import urllib.request


def main():
    for base in ("https://kn2b53tt1c.execute-api.us-east-1.amazonaws.com",
                 "https://t37yopm41e.execute-api.us-east-1.amazonaws.com"):
        url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"] + "&" + urllib.parse.urlencode({"audience": base})
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]})
        with urllib.request.urlopen(req, timeout=30) as response:
            token = json.load(response)["value"]
        req = urllib.request.Request(base + "/v1/commits", data=b"{}", headers={"Authorization": "Bearer " + token,
                                                                               "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.load(response)
        if result.get("status") != "verified" or result.get("sha") != os.environ["GITHUB_SHA"]:
            raise SystemExit("Registry did not verify this exact commit")
        print("Registry verified", os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_SHA"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("Registry commit identity failed; token and response omitted") from None
