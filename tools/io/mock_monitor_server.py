"""A recording stand-in for the noeira-cloud Worker, for the sink gates.

    python3 tools/io/mock_monitor_server.py <port-file> <log-file> [<seconds>] [<port>]

Binds `<port>` (default 0, i.e. any), writes the bound port to `<port-file>`
atomically, and appends one line per request to `<log-file>`:

    <monotonic_ms> <METHOD> <path> <body-or-"<N bytes>">

⚠ IT RECORDS RATHER THAN VALIDATES. The gates' questions are about WHEN, HOW
OFTEN and WITH WHAT BYTES a client speaks — a heartbeat that fires while
payloads are flowing, or a checkpoint uploaded four times when once would do,
are defects no status code can express. So the fixture's whole job is to leave
an ordered, timestamped trace for the Mojo side to assert against.

It plays TWO roles, because the artifact flow needs both and neither may reach
the network:

  the Worker   POST /artifacts  -> {id, upload_url} pointing back at this server
               POST /artifacts/<id>/complete
               POST /runs, /ingest, /runs/<id>/ping, /runs/<id>/finish
  R2           PUT  /r2/<key>   -> stores the bytes in memory, records the sha256
               GET  /r2/<key>   -> hands them back

⚠⚠ A PUT WHOSE KEY CONTAINS `slow` SLEEPS FOR `SLOW_PUT_MS`, and that is not a
convenience. `ArtifactSink`'s supersede rule only does anything when requests
arrive FASTER THAN TRANSFERS COMPLETE — which is the real condition (a 215 MB
ACT checkpoint against a validation every N steps) and never the loopback one,
where a PUT finishes before the next offer is made. Without this the gate
measures a sink that had nothing to collapse and reports success.

⚠ `/fail/...` prefixes let a gate ask for a failure on demand, which is the one
thing a real server will not do reliably.

## Project definition files (private projects)

  PUT  /projects/<slug>                   create; set description
  GET  /projects/<slug>/files             rows + a download_url into /r2/
  POST /projects/<slug>/files/presign     409 on a stale base, 413 over the cap
  POST /projects/<slug>/files/complete    HASHES the object, then compare-and-swap
  POST /__inject_file                     {slug, path, sha256} — a row the real
                                          Worker would refuse, for the client's
                                          own path check

⚠⚠ THIS MIRRORS THE WORKER'S RULES AND IS NOT THEIR GATE. The swap and the
path rules are gated on the Worker itself (`worker/test/project_files.test.ts`
in noeira-cloud); this copy exists so the Mojo client's behaviour under those
rules can be driven without a network. The cap is deliberately TINY
(`PROJECT_FILE_MAX`) so a gate can exceed it with a 5 KB file.

Routes: anything else 200s. `/__shutdown` exits.
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

LOG = None
T0 = time.monotonic()
PORT = 0

# key -> bytes, and the artifact rows the "Worker" has seen.
OBJECTS = {}
ARTIFACTS = {}
ARTIFACT_ROWS = {}  # id -> row, for GET /artifacts?run_id= and /artifacts/<id>
SLOW_PUT_MS = 150
LOCK = threading.Lock()
# `POST /__fail_next` with a count in the body: that many following POSTs to
# /runs or /ingest answer 503 — an overloaded monitor, for the retry gate.
FAIL_NEXT = 0

# slug -> {"description": str, "files": {path: sha256}}
PROJECTS = {}
# (slug, name) -> {"n_episodes", "n_frames", "fps", "files": {path: {sha, size, status}}}
DATASETS = {}
PROJECT_FILE_MAX = 4096
_SEG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _path_ok(path):
    if not 0 < len(path) <= 512:
        return False
    parts = path.split("/")
    return len(parts) <= 8 and all(_SEG.match(p) for p in parts)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _record(self, method, note):
        ms = int((time.monotonic() - T0) * 1000)
        with LOCK:
            with open(LOG, "a") as f:
                f.write(f"{ms} {method} {self.path} {note}\n")

    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _raw(self, code, blob, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        if blob:
            self.wfile.write(blob)

    def log_message(self, *a):
        pass

    # -- routes -----------------------------------------------------------

    def do_PUT(self):
        u = urlparse(self.path)
        blob = self._body()
        m = re.match(r"^/projects/([^/]+)/datasets/([^/]+)$", u.path)
        if m:
            self._record("PUT", blob.decode("utf-8", "replace"))
            d = json.loads(blob or b"{}")
            with LOCK:
                if m.group(1) not in PROJECTS:
                    return self._json(404, {"error": "unknown project"})
                ds = DATASETS.setdefault((m.group(1), m.group(2)), {"files": {}})
                for k in ("n_episodes", "n_frames", "fps"):
                    if k in d:
                        ds[k] = d[k]
            return self._json(200, {"name": m.group(2)})
        m = re.match(r"^/projects/([^/]+)$", u.path)
        if m:
            self._record("PUT", blob.decode("utf-8", "replace"))
            d = json.loads(blob or b"{}")
            with LOCK:
                p = PROJECTS.setdefault(m.group(1), {"description": "", "files": {}})
                if "description" in d:
                    p["description"] = d["description"]
            return self._json(200, {"slug": m.group(1), "description": p["description"]})
        if "slow" in u.path:
            time.sleep(SLOW_PUT_MS / 1000.0)
        self._record("PUT", f"<{len(blob)} bytes sha={hashlib.sha256(blob).hexdigest()[:16]}>")
        if u.path.startswith("/fail/"):
            return self._json(500, {"error": "asked to fail"})
        with LOCK:
            OBJECTS[u.path] = blob
        self._raw(200, b"")

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/__shutdown":
            os._exit(0)
        m = re.match(r"^/projects/([^/]+)/datasets/([^/]+)/files$", u.path)
        if m:
            self._record("GET", "")
            slug, name = m.group(1), m.group(2)
            with LOCK:
                ds = DATASETS.get((slug, name))
                if ds is None:
                    return self._json(404, {"error": "unknown dataset"})
                files = [
                    {
                        "path": path,
                        "sha256": f["sha"] if f["status"] == "ready" else None,
                        "sizeBytes": f["size"] if f["status"] == "ready" else None,
                        "status": f["status"],
                        "download_url": (
                            f"http://127.0.0.1:{PORT}/r2/ds/{slug}/{name}/{path}"
                            if f["status"] == "ready" else ""
                        ),
                    }
                    for path, f in sorted(ds["files"].items())
                ]
                return self._json(200, {"dataset": {"name": name, "nEpisodes": ds.get("n_episodes")}, "files": files})
        m = re.match(r"^/projects/([^/]+)/datasets$", u.path)
        if m:
            self._record("GET", "")
            with LOCK:
                rows = []
                for (slug, name), ds in sorted(DATASETS.items()):
                    if slug != m.group(1):
                        continue
                    fs = ds["files"].values()
                    rows.append({
                        "name": name, "nEpisodes": ds.get("n_episodes"),
                        "fileCount": len(fs),
                        "readyCount": sum(1 for f in fs if f["status"] == "ready"),
                        "sizeBytes": sum((f["size"] or 0) for f in fs),
                    })
            return self._json(200, rows)
        m = re.match(r"^/projects/([^/]+)/files$", u.path)
        if m:
            self._record("GET", "")
            with LOCK:
                p = PROJECTS.get(m.group(1))
                if p is None:
                    return self._json(404, {"error": "unknown project"})
                files = [
                    {
                        "path": path,
                        "sha256": sha,
                        "sizeBytes": len(OBJECTS.get(f"/r2/pf/{m.group(1)}/{sha}", b"")),
                        "download_url": f"http://127.0.0.1:{PORT}/r2/pf/{m.group(1)}/{sha}",
                    }
                    for path, sha in sorted(p["files"].items())
                ]
                return self._json(
                    200,
                    {
                        "project": m.group(1),
                        "description": p["description"],
                        "max_file_bytes": PROJECT_FILE_MAX,
                        "files": files,
                    },
                )
        if u.path == "/artifacts":
            self._record("GET", "")
            run_id = parse_qs(u.query).get("run_id", [""])[0]
            with LOCK:
                rows = [dict(r) for r in ARTIFACT_ROWS.values() if r["runId"] == run_id]
            return self._json(200, rows)
        m = re.match(r"^/artifacts/([^/]+)$", u.path)
        if m:
            self._record("GET", "")
            with LOCK:
                row = ARTIFACT_ROWS.get(m.group(1))
            if row is None:
                return self._json(404, {"error": "not found"})
            if row["status"] != "ready":
                return self._json(409, {"error": "artifact upload never completed"})
            return self._json(200, dict(row, download_url=f"http://127.0.0.1:{PORT}{row['key']}"))
        if u.path.startswith("/r2/"):
            with LOCK:
                blob = OBJECTS.get(u.path)
            self._record("GET", f"<{len(blob) if blob else 0} bytes>")
            if blob is None:
                return self._json(404, {"error": "no such object"})
            return self._raw(200, blob)
        self._record("GET", "")
        self._json(200, {"ok": True})

    def do_POST(self):
        u = urlparse(self.path)
        body = self._body()
        text = body.decode("utf-8", "replace")
        self._record("POST", text)

        if u.path == "/__shutdown":
            os._exit(0)

        global FAIL_NEXT
        if u.path == "/__fail_next":
            with LOCK:
                FAIL_NEXT = int(text or "0")
            return self._json(200, {"ok": True})
        if u.path in ("/runs", "/ingest"):
            with LOCK:
                busy = FAIL_NEXT > 0
                if busy:
                    FAIL_NEXT -= 1
            if busy:
                return self._json(503, {"error": "asked to be busy"})

        # A payload the server refuses for good: never worth retrying.
        if "/reject" in u.path:
            return self._json(400, {"error": "asked to reject"})

        # A registration the gate has asked to fail, so the retry path and the
        # "abandoned" accounting can be exercised without unplugging anything.
        if "/fail" in u.path:
            return self._json(500, {"error": "asked to fail"})

        if u.path == "/artifacts":
            try:
                d = json.loads(text)
            except Exception:
                return self._json(400, {"error": "bad json"})
            run_id, path = d.get("run_id", ""), d.get("path", "")
            key = f"/r2/{run_id}/{path}"
            # One row per (run, path), like the real Worker: the same id comes
            # back for a re-registration, which is what supersede relies on.
            with LOCK:
                aid = ARTIFACTS.setdefault(key, f"art-{len(ARTIFACTS) + 1}")
                ARTIFACT_ROWS[aid] = {"id": aid, "runId": run_id, "path": path, "key": key,
                                      "status": "pending", "sha256": None, "sizeBytes": None}
            return self._json(
                201,
                {
                    "id": aid,
                    "run_id": run_id,
                    "path": path,
                    "object_key": key,
                    "upload_url": f"http://127.0.0.1:{PORT}{key}",
                    "expires_at": "2099-01-01T00:00:00Z",
                },
            )

        if u.path == "/__inject_dataset_file":
            d = json.loads(text)
            with LOCK:
                ds = DATASETS.setdefault((d["slug"], d["name"]), {"files": {}})
                body = d.get("body", "x").encode()
                ds["files"][d["path"]] = {"sha": d["sha256"], "size": len(body), "status": "ready"}
                OBJECTS[f"/r2/ds/{d['slug']}/{d['name']}/{d['path']}"] = body
            return self._json(200, {"ok": True})

        m = re.match(r"^/projects/([^/]+)/datasets/([^/]+)/files(/complete)?$", u.path)
        if m:
            slug, name, complete = m.group(1), m.group(2), m.group(3)
            d = json.loads(text)
            path, sha, size = d["path"], d["sha256"], d["size_bytes"]
            key = f"/r2/ds/{slug}/{name}/{path}"
            with LOCK:
                ds = DATASETS.get((slug, name))
                if ds is None:
                    return self._json(404, {"error": "unknown dataset"})
                if not _path_ok(path):
                    return self._json(400, {"error": f"bad path {path!r}"})
                cur = ds["files"].get(path)
                if not complete:
                    if cur and cur["status"] == "ready" and cur["sha"] == sha and cur["size"] == size:
                        return self._json(200, {"path": path, "unchanged": True})
                    ds["files"][path] = {"sha": None, "size": None, "status": "pending"}
                    return self._json(201, {"path": path, "upload_url": f"http://127.0.0.1:{PORT}{key}"})
                if cur is None:
                    return self._json(404, {"error": "not registered"})
                blob = OBJECTS.get(key)
                if blob is None or len(blob) != size:
                    return self._json(422, {"error": "size mismatch"})
                ds["files"][path] = {"sha": sha, "size": size, "status": "ready"}
            return self._json(200, {"path": path, "status": "ready"})

        if u.path == "/__inject_file":
            d = json.loads(text)
            with LOCK:
                p = PROJECTS.setdefault(d["slug"], {"description": "", "files": {}})
                p["files"][d["path"]] = d["sha256"]
                OBJECTS[f"/r2/pf/{d['slug']}/{d['sha256']}"] = d.get("body", "x").encode()
            return self._json(200, {"ok": True})

        m = re.match(r"^/projects/([^/]+)/files/(presign|complete)$", u.path)
        if m:
            slug, step = m.group(1), m.group(2)
            d = json.loads(text)
            path, sha, size, base = d["path"], d["sha256"], d["size_bytes"], d["base_sha256"]
            key = f"/r2/pf/{slug}/{sha}"
            with LOCK:
                p = PROJECTS.get(slug)
                if p is None:
                    return self._json(404, {"error": "unknown project"})
                if not _path_ok(path):
                    return self._json(400, {"error": f"bad path {path!r}"})
                if size > PROJECT_FILE_MAX:
                    return self._json(413, {"error": "over the cap", "max_file_bytes": PROJECT_FILE_MAX})
                cur = p["files"].get(path, "")
                if cur == sha:
                    return self._json(200, {"path": path, "unchanged": True})
                if cur != base:
                    return self._json(409, {"error": "conflict", "current_sha256": cur})
                if step == "presign":
                    return self._json(201, {"path": path, "upload_url": f"http://127.0.0.1:{PORT}{key}"})
                blob = OBJECTS.get(key)
                if blob is None:
                    return self._json(422, {"error": "nothing uploaded"})
                if len(blob) != size or hashlib.sha256(blob).hexdigest() != sha:
                    return self._json(422, {"error": "bytes do not match sha256/size"})
                p["files"][path] = sha
            return self._json(200, {"path": path, "sha256": sha, "size_bytes": size})

        m = re.match(r"^/artifacts/([^/]+)/complete$", u.path)
        if m:
            d = json.loads(text or "{}")
            with LOCK:
                row = ARTIFACT_ROWS.get(m.group(1))
                if row is not None:
                    row.update(status="ready", sha256=d.get("sha256"), sizeBytes=d.get("size_bytes"))
            return self._json(200, {"ok": True, "status": "ready"})

        if u.path.endswith("/complete"):
            return self._json(200, {"ok": True, "status": "ready"})

        self._json(200, {"ok": True, "cmd": ""})


def main():
    global LOG, PORT
    port_file, LOG = sys.argv[1], sys.argv[2]
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    port = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    open(LOG, "w").close()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    PORT = srv.server_address[1]
    tmp = port_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(PORT))
    os.rename(tmp, port_file)
    srv.timeout = 0.2
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        srv.handle_request()


if __name__ == "__main__":
    main()
