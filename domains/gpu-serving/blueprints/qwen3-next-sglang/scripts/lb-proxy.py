#!/usr/bin/env python3
"""Simple round-robin load balancer proxy for multiple vLLM backends."""
import argparse
import http.server
import itertools
import urllib.request
import sys
import threading


class RoundRobinHandler(http.server.BaseHTTPRequestHandler):
    backends = []
    _cycle = None
    _lock = threading.Lock()

    def _next_backend(self):
        with self._lock:
            return next(self._cycle)

    def do_POST(self):
        backend = self._next_backend()
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        url = f"{backend}{self.path}"
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f'{{"error": "{e}"}}'.encode())

    def do_GET(self):
        backend = self._next_backend()
        url = f"{backend}{self.path}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(f'{{"error": "{e}"}}'.encode())

    def log_message(self, format, *args):
        pass  # suppress logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--backends", nargs="+", required=True)
    args = parser.parse_args()

    RoundRobinHandler.backends = args.backends
    RoundRobinHandler._cycle = itertools.cycle(args.backends)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), RoundRobinHandler)
    print(f"LB proxy on :{args.port} -> {args.backends}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
