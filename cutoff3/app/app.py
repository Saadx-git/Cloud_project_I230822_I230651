"""CPU-bound workload service for the live experiments (standard library only).

Every GET /work burns a fixed amount of CPU (PBKDF2 iterations), so the
CPU utilisation seen by the HPA is proportional to the request rate, and a pod
has a well-defined capacity C (requests/s at 100 % of its CPU request).

Prometheus metrics on GET /metrics:
  app_requests_total               counter of served /work requests
  app_request_seconds_sum/_count   in-pod service time (queueing excluded)
  app_request_cpu_seconds_total    CPU time spent in /work handlers (thread CPU clock)
"""
import hashlib
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ITERS = int(os.environ.get("WORK_ITERS", "40000"))
POD = os.environ.get("POD_NAME", "local")
lock = threading.Lock()
count, secs, cpus = 0, 0.0, 0.0


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        global count, secs, cpus
        if self.path.startswith("/work"):
            t0, c0 = time.perf_counter(), time.thread_time()
            hashlib.pbkdf2_hmac("sha256", b"payload", b"salt", ITERS)
            dt, dc = time.perf_counter() - t0, time.thread_time() - c0
            with lock:
                count += 1
                secs += dt
                cpus += dc
            self._send(200, "ok\n")
        elif self.path.startswith("/metrics"):
            with lock:
                c, s, u = count, secs, cpus
            self._send(200, f"# TYPE app_requests_total counter\napp_requests_total {c}\n"
                            f"# TYPE app_request_seconds summary\napp_request_seconds_sum {s}\n"
                            f"app_request_seconds_count {c}\n"
                            f"# TYPE app_request_cpu_seconds_total counter\napp_request_cpu_seconds_total {u}\n",
                            "text/plain; version=0.0.4")
        elif self.path.startswith("/healthz"):
            self._send(200, "ok\n")
        else:
            self._send(404, "not found\n")


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), H)
    srv.daemon_threads = True
    srv.serve_forever()
