"""
serve.py — HTTP server senza cache per la presentazione MCP_ECOM
Garantisce che ogni richiesta restituisca il file fresco dal disco (no 304).

NOTA: usa UTF-8 stdout anche su terminali Windows cp1252.

Uso:
    python serve.py           # porta 8000 (default)
    python serve.py 9000      # porta custom
"""

import sys
import io

# Forza stdout UTF-8 su Windows per evitare UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
from http.server import SimpleHTTPRequestHandler, HTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    """HTTP handler che aggiunge header no-cache su ogni risposta."""

    # ── Sopprimi i log di accesso per chiarezza (opzionale: rimuovi per debug) ──
    def log_message(self, format, *args):
        print(f"  {self.address_string()} → {args[0]}")

    # ── Sovrascrivi send_response per azzerare la cache ──
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # Rimuove ETag e Last-Modified per bloccare il 304
        super().end_headers()

    # ── Blocca il 304: servi sempre 200 ──
    def send_response(self, code, message=None):
        if code == 304:
            code = 200
        super().send_response(code, message)

    # ── Ignora If-Modified-Since e If-None-Match (causa del 304) ──
    def do_GET(self):
        # Rimuovi gli header condizionali prima che SimpleHTTPRequestHandler li usi
        self.headers._headers = [
            (k, v) for k, v in self.headers._headers
            if k.lower() not in ("if-modified-since", "if-none-match")
        ]
        super().do_GET()


def run(port: int = 8000):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(("", port), NoCacheHandler)
    print(f"\n  [SERVER]  Presentazione --> http://localhost:{port}/index.html")
    print(f"  [OK]  Cache disabilitata - ogni refresh legge i file freschi dal disco\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server fermato.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run(port)
