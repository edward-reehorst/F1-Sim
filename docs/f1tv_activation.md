# Activating F1TV for live timing capture

`f1-sim` reads a **live** Grand Prix by recording F1TV's live-timing feed. That feed
is authenticated: without a valid F1TV token the connection is refused with
`401 Unauthorized`, and there is no anonymous or free alternative.

This document covers getting that token. If you only want to simulate from a race
that has **already finished**, you do not need any of this — use
`--source post-session` (see [live_race_simulation.md](live_race_simulation.md)).

---

## 1. Why a subscription is required

Two separate FastF1 limits are worth understanding, because they are the reason
this cannot be worked around:

| What you try | Result |
|---|---|
| Static archive (`fastf1.get_session(...).load()`) during a live session | `403` — the archive is only published *after* the session ends |
| Static archive after the session ends | `200` — works, and needs no token |
| Live timing stream with no token | `401 Unauthorized` |
| Live timing stream with an F1TV token | works, and needs an **active F1TV subscription** |

So the token is not optional for live capture, and the archive cannot be used as a
stand-in while a session is running.

---

## 2. Get an F1TV subscription with live timing

FastF1 validates the token's subscription server-side and requires an active
**F1TV Access / Pro / Premium** subscription. A plain F1.com login is not enough.

### In the United States

As of the 2026 season, Apple TV is the exclusive US broadcaster for F1, and
**F1 TV Premium is included at no additional cost with an Apple TV subscription**.
It is not activated automatically — you have to turn it on:

1. Subscribe to **Apple TV** if you do not have it (tv.apple.com).
2. Sign in to <https://formula1.com/en-us/subscribe-to-f1-tv> with your **F1
   account**, then press **ACTIVATE** next to "F1 TV Premium — Available with an
   Apple TV subscription".
3. Confirm the activation went through in the F1 app or on that page.

Because the entitlement lives on your F1 account, you then sign in to *that same
F1 account* in the next step.

> US-specific: broadcaster arrangements and bundle contents change between
> seasons. If the activate page offers you a standalone tier instead (F1 TV Access
> is the cheapest, and also includes live timing), any tier with live timing works.

### Elsewhere

Subscribe to any F1 TV tier that includes **live timing** (Access, Pro, or
Premium) at <https://formula1.com/en/subscribe-to-f1-tv>.

---

## 3. Authenticate FastF1 against your account

Run this from the project so it uses the project virtualenv:

```bash
uv run python -m fastf1 auth f1tv --authenticate
```

(Or `.venv/bin/python -m fastf1 auth f1tv --authenticate` if you are not using `uv`.)

FastF1 starts a small local web server on a random port and prints a URL:

```
Please open the following URL in your browser to authenticate FastF1 with your
Formula1/F1TV account:

https://f1login.fastf1.dev?port=54321
```

Open that URL, sign in with your F1/F1TV account, and approve the request. The
browser hands the subscription token back to the local server, which verifies its
signature and stores it. You should see:

```
Sign-in successful.
```

Keep the terminal open until it finishes — the local server is what receives the
token.

---

## 4. Verify

```bash
uv run python -m fastf1 auth f1tv --status
```

A working setup prints your token and subscription state:

```
Token Status: Expires 2026-10-14 18:22:41 (UTC)
Subscription Status: active
Subscribed Product: F1 TV Premium
```

Anything other than that (notably `Not authenticated`) means step 3 did not
complete.

---

## 5. Where the token lives, and how to treat it

| | |
|---|---|
| Location | `~/Library/Application Support/fastf1/f1auth.json` (macOS) |
| Contents | A live subscription token for **your** F1 account |
| Lifetime | Expires; re-run `--authenticate` when `--status` says so |

Treat this file as a **credential**. Anyone holding it can act against your F1
account's API access, so do not commit it, paste it into a bug report, or share a
machine with untrusted users. It is outside the repository, so version control is
not a risk — but syncing tools and backups may pick it up.

To revoke it locally:

```bash
uv run python -m fastf1 auth f1tv --clear
```

---

## 6. Recording a session

Two rules that are easy to get wrong:

1. **Start the recorder before the session begins.** The recording is the only
   source of the driver list and session metadata that the lap data is
   interpreted against. A recorder started mid-session produces a frame with no
   drivers in it.
2. **Try a practice session first.** The recording/tailing path is the one part of
   this feature that cannot be verified without a live feed. Practice sessions
   exercise the same code path as a race and cost you nothing if something is
   off.

```bash
# Start this before lights out:
uv run python scripts/network_capture.py record \
    --year 2026 --gp italian \
    --circuit monza --out out/snapshots --poll 5
```

The script prints the authentication steps above and exits with code `4` if the
stream refuses the connection, so a missing token fails immediately and loudly
rather than silently capturing nothing.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401` / `Unauthorized` | No token, or an expired one | `auth f1tv --authenticate`, then `--status` |
| `This feature requires an active F1TV Access/Pro/Premium subscription` | Signed in, but the entitlement is not active | Complete the activation in §2 and re-authenticate |
| `Subscription token is invalid` | Token was revoked or the subscription lapsed | `--clear`, then `--authenticate` again |
| `Sign-in successful` but `--status` says `Not authenticated` | The local callback server was killed before it stored the token | Re-run `--authenticate` and leave the terminal alone until it finishes |
| Capture writes nothing but never errors | Recorder started after the session, so there is no driver metadata | Start the recorder before the session (§6) |
| Authenticated, still `401` | Captured a token for a different F1 account than the one holding the subscription | Check the account you signed in with, re-`--clear` and re-authenticate |

---

## 8. What this does *not* affect

The token is used by exactly one command: `scripts/network_capture.py`.

Resuming a simulation from a captured snapshot is a **local file read** and makes
no network calls at all — see §7 of [live_race_simulation.md](live_race_simulation.md).
Once you have snapshot JSON, you never need this token again.
