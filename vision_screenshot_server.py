"""
Screenshot URL server — serve current screenshot as binary so the vision API gets real data.

Design:
  - We pass a URL (http://127.0.0.1:19998/screenshot) to the vision API instead of a
    data:image/png;base64,... string in the request body.
  - mlx-openai-server fetches the image from that URL (binary GET). No base64 in the
    request → less memory, smaller payload, no encoding/decoding on our side.
  - Orchestrator calls set_current_screenshot(png_bytes) before each decide(); Qwen
    receives screenshot_url and the server fetches the PNG.

Starts lazily on first set_current_screenshot(). Port 19998.
"""
from __future__ import annotations

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

_current_screenshot: Optional[bytes] = None
_server: Optional[HTTPServer] = None
_port = 19998


def set_current_screenshot(png_bytes: bytes) -> None:
    """Set the screenshot bytes to serve. Starts the server if not running."""
    global _current_screenshot, _server
    _current_screenshot = png_bytes
    if _server is None:
        _start_server()


def get_screenshot_url() -> str:
    """Return the URL the vision API should use to fetch the current screenshot."""
    return f"http://127.0.0.1:{_port}/screenshot"


def _start_server() -> None:
    global _server
    if _server is not None:
        return

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/screenshot", "/screenshot.png"):
                data = _current_screenshot
                if not data:
                    self.send_error(404, "No screenshot set")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # quiet

    _server = HTTPServer(("127.0.0.1", _port), _Handler)
    thread = threading.Thread(target=_server.serve_forever, daemon=True)
    thread.start()
