# contrib — the host ingest policy

These two files decide how often LastSeek's index is updated and how much it
writes to disk doing it. Until 2026-08-17 they existed **only** as untracked
files on one machine:

```
~/.lastseek/bin/lastseek-drain-tick
~/Library/LaunchAgents/com.edgevector.lastseek-drain.plist
```

So the product's dominant cost parameters had no change history, no review, no
test, and no existence anywhere else. A fresh LastSeek install got a binary and
a frozen index. Two of the four largest defects found on this product in a week
were single lines in those files, and neither line could be attributed, dated,
or reverted.

Tracking them here does not change what runs. **Installing is a separate,
deliberate act** — see below. The point is that the policy now has a reviewer.

## What changed from the host copies these were seeded from

Both changes are backed by measurements recorded in the files' own comments.

| file | was | now | why |
|---|---|---|---|
| `com.edgevector.lastseek-drain.plist` | `ProcessType: Background`, `Nice 5` | `ProcessType: Standard`, no `Nice` | the QoS clamp confined the drain to efficiency cores and cost **17.8x** wall / **3.8x** CPU on an A/B of the same 100 live files |
| `lastseek-drain-tick` | `compact` when `records > 0` | `compact` at `>= 2000` pending ops, or after 1 h | compaction rewrites the **whole slab**, so the old threshold rewrote the corpus for one changed record — ~12.5 GiB/day of SSD writes; folding 5,500 ops costs the same 0.3 s as folding 1 |

`LowPriorityIO` is deliberately kept. The A/B cannot separate its cost from the
QoS clamp's, and disk politeness is the half that protects other work on a
shared machine. Isolating it is the next measurement if lag persists — not a
guess.

The tick also now logs `pending=` and `lag_ms=`. `remaining=` alone could never
answer "is the index current": it is files left after **this bounded pass**, so
a tick slow enough to absorb its own `StartInterval` always empties the inbox
and always reports zero. Every one of 2,132 ticks logged before 2026-08-17 read
`remaining=0` while the corpus behind `brain ask` was 30–59 minutes stale.

## What is deliberately unchanged

`--max-files` and the tick's lock. **The bound is what protects the machine, not
the QoS fence.** The 2026-08-07 incident that took the host down was an
*unbounded* drain that wrote 549.80 GB in twenty minutes, and it would have done
that just as thoroughly on efficiency cores.

## Installing

Not automatic, and not part of any CI or artifact step. Changing a live
LaunchAgent is a host mutation on a machine that also runs the primary LastDB
node, so it is done knowingly:

```bash
cp contrib/lastseek-drain-tick "$HOME/.lastseek/bin/lastseek-drain-tick"
chmod +x "$HOME/.lastseek/bin/lastseek-drain-tick"

sed "s|/Users/tomtang|$HOME|g" contrib/com.edgevector.lastseek-drain.plist \
  > "$HOME/Library/LaunchAgents/com.edgevector.lastseek-drain.plist"
launchctl unload "$HOME/Library/LaunchAgents/com.edgevector.lastseek-drain.plist"
launchctl load   "$HOME/Library/LaunchAgents/com.edgevector.lastseek-drain.plist"
```

Verify with `lastseek status`: `freshness_lag_ms` should fall within a few
ticks, and `lastseek status --queue` shows the backlog it is working through.
If it does not fall, the fence was not the binding constraint and the next
suspect is the arrival mix — see the admission work, which is a separate
change.

The plist still carries an absolute `StandardOutPath`; launchd does not expand
`$HOME` there, which is why the install line rewrites it.
