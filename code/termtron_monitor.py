"""
termtron_monitor.py - background monitor for a Termetron task (no blocking wait)

After dispatching a task to a Termetron session, run this in the background:
it polls the session until busy, then checks the target log for a completion
marker ("total time") and writes a status marker file.

  Marker file content: DONE | FAILED | RUNNING   (+ result snapshot / error tail)

Usage:
  python code/termtron_monitor.py --session shell --log eval_unified_log.txt
                                   [--port 8900] [--timeout 1800] [--poll 5]
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def fetch(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="shell")
    ap.add_argument("--log", required=True)
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--poll", type=int, default=5)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}/api/sessions"
    marker = f"{args.session}_monitor.txt"
    t0 = time.time()

    def write(state, extra=""):
        with open(marker, "w", encoding="utf-8") as f:
            f.write(f"{state}\n{extra}")

    write("RUNNING", f"monitor started, watching session '{args.session}' -> {args.log}")

    try:
        while time.time() - t0 < args.timeout:
            try:
                data = fetch(base)
            except Exception as e:  # noqa: BLE001
                write("RUNNING", f"poll err: {e}")
                time.sleep(args.poll)
                continue
            s = data.get(args.session, {})
            busy = s.get("busy", False)
            if not busy:
                # session idle -> task finished (success or crash)
                tail = ""
                if os.path.exists(args.log):
                    with open(args.log, encoding="utf-8") as f:
                        txt = f.read()
                    tail = txt[-3000:]
                    if "total time" in txt:
                        write("DONE", tail)
                        print("DONE")
                        return
                # no completion marker -> FAILED (crashed); capture error tail
                # try to read terminal output via status lines
                write("FAILED", tail or "(no log)")
                print("FAILED (no 'total time' marker in log)")
                return
            time.sleep(args.poll)
        write("RUNNING", f"timeout {args.timeout}s, still running")
        print("RUNNING (timeout)")
    except Exception as e:  # noqa: BLE001
        write("FAILED", f"monitor error: {e}")
        print("MONITOR ERROR:", e)


if __name__ == "__main__":
    main()
