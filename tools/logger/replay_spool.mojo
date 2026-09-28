# +--------------------------------------------------------------------------+ #
# | Send what a run's RemoteLogger could not deliver live
# +--------------------------------------------------------------------------+ #
"""Replay RemoteLogger spool files into the monitor.

    pixi run logger-replay                       # every spool under logs/ and projects/
    pixi run logger-replay projects/so101/runs/<id>/remote.spool

A run writes a spool when the monitor did not take a payload live: an outage
still going at `close()`, an outage past the logger's `hold_bytes`, or a
payload the server rejected (`noeira/io/http_sink.mojo`). Each record is the
exact POST — URL and JSON — so a replay is the run's own traffic, in its own
order, just late.

Records that go through are removed; the rest are kept, so this can be run
again until it reports nothing kept. The API key comes from `.env`
(`NOEIRA_CLOUD_API_KEY`), like the logger's.

⚠ THE URLS ARE THE ONES THE RUN USED. A spool written against another
monitor URL replays against that URL.
"""

from std.sys import argv

from noeira.core.dotenv import load_dotenv
from noeira.io.http_sink import replay_spool
from noeira.io.proc import run_capture


def main() raises:
    var paths = List[String]()
    var args = argv()
    for i in range(1, len(args)):
        paths.append(String(args[i]))
    if len(paths) == 0:
        var found = run_capture(
            "find logs/remote_spool projects -name '*.spool' -type f"
            " 2>/dev/null || true",
            1 << 20,
        )
        for line in found.split("\n"):
            var p = String(String(line).strip())
            if p.byte_length() > 0:
                paths.append(p)
    if len(paths) == 0:
        print("logger-replay: no spool files — nothing was left undelivered")
        return

    var key = String("")
    try:
        key = load_dotenv(".env").get("NOEIRA_CLOUD_API_KEY", "")
    except:
        pass

    var total_sent = 0
    var total_kept = 0
    for p in paths:
        try:
            var r = replay_spool(p, key)
            total_sent += r[0]
            total_kept += r[1]
            print(
                "  " + p + ": " + String(r[0]) + " delivered, "
                + String(r[1]) + " kept"
            )
        except e:
            print("  " + p + ": NOT replayed — " + String(e))
    print(
        "logger-replay: " + String(total_sent) + " delivered, "
        + String(total_kept) + " kept across " + String(len(paths))
        + " spool file(s)"
    )
