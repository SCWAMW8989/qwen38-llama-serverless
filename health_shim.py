!/usr/bin/env python3
"""Health-check shim (v1.1) for a Runpod load balancing endpoint running llama-server.

Runpod polls GET /ping on PORT_HEALTH (default 8081):
  204 = initializing, 200 = healthy, anything else = unhealthy.
llama-server itself answers 503 while loading the model, which Runpod would
read as "unhealthy", so this shim translates it. Real traffic never goes
through here; it goes straight to llama-server on PORT.
"""
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

MAIN_PORT = int(os.environ.get("PORT", "8080"))
HEALTH_PORT = int(os.environ.get("PORT_HEALTH", "8081"))
STARTUP_GRACE_S = int(os.environ.get("HEALTH_STARTUP_GRACE", "600"))
UPSTREAM = f"http://127.0.0.1:{MAIN_PORT}/health"

START = time.time()
EVER_HEALTHY = False


def probe() -> int:
    """Return the status code Runpod should see."""
    global EVER_HEALTHY
    try:
        with urllib.request.urlopen(UPSTREAM, timeout=3) as r:
            if r.status == 200:
                EVER_HEALTHY = True
                return 200
            return 204
    except urllib.error.HTTPError as e:
        # 503 = model still loading
        return 204 if e.code == 503 else 500
    except Exception:
        # Not listening yet. Fine during startup, a failure after we were healthy.
        if EVER_HEALTHY:
            return 500
        return 204 if (time.time() - START) < STARTUP_GRACE_S else 500


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] not in ("/ping", "/health", "/"):
            self.send_response(404)
            self.end_headers()
            return
        code = probe()
        self.send_response(code)
        if code == 200:
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.end_headers()

    def log_message(self, *args):
        pass  # keep logs quiet; Runpod polls often


if __name__ == "__main__":
    print(f"health_shim: :{HEALTH_PORT} -> {UPSTREAM}", flush=True)
    HTTPServer(("0.0.0.0", HEALTH_PORT), Handler).serve_forever()
