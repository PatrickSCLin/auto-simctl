"""
UI-UG-7B HTTP Server — keeps the model loaded in memory as a long-running service.

Endpoints:
    POST /grounding   { image_base64?: str, image_url?: str }  -> { elements: [...] }
    POST /referring   { image_base64?: str, image_url?: str, bbox: [int] } -> { description: str }
    GET  /health      -> { status: "ok", model: str }

    Prefer image_url (server fetches binary) over image_base64 to avoid large request bodies.

Usage:
    python3 ui_server.py                         # default port 8081
    python3 ui_server.py --port 8082 --model-path /path/to/model
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_MODEL_PATH = str(
    Path.home() / ".cache/huggingface/hub/ui-ug-7b-2601-4bit"
)

# ── Global model state ─────────────────────────────────────────────────────────
_model = None
_processor = None
_model_path: str = DEFAULT_MODEL_PATH


def load_model(model_path: str) -> None:
    global _model, _processor, _model_path
    from mlx_vlm import load
    _model_path = model_path
    print(f"[ui_server] Loading UI-UG model from {model_path}...", flush=True)
    t0 = time.time()
    _model, _processor = load(model_path)
    print(f"[ui_server] Model ready in {time.time()-t0:.1f}s", flush=True)


def _infer(image_bytes: bytes, prompt: str, max_tokens: int = 512) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        tmp = f.name
    try:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": tmp},
                {"type": "text",  "text": prompt},
            ],
        }]
        formatted = apply_chat_template(_processor, config=_model.config, prompt=messages)
        t0 = time.time()
        result = generate(_model, _processor, image=tmp, prompt=formatted,
                          max_tokens=max_tokens, verbose=False)
        print(f"[ui_server] inference done in {time.time()-t0:.1f}s", flush=True)
        # mlx-vlm may return a GenerationResult object; extract the text
        if isinstance(result, str):
            return result
        if hasattr(result, "text"):
            return result.text
        return str(result)
    finally:
        os.unlink(tmp)


# ── HTTP Handler ───────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[ui_server] {fmt % args}", flush=True)

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": _model_path,
                "ready": _model is not None,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = self._read_json()
        except Exception as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        if _model is None:
            self._send_json(503, {"error": "Model not loaded yet"})
            return

        # Prefer image_url (fetch binary) over image_base64
        image_url = body.get("image_url")
        if image_url:
            try:
                with urllib.request.urlopen(image_url, timeout=30) as r:
                    image_bytes = r.read()
            except Exception as e:
                self._send_json(400, {"error": f"Failed to fetch image_url: {e}"})
                return
        else:
            img_b64 = body.get("image_base64", "")
            try:
                image_bytes = base64.b64decode(img_b64)
            except Exception as e:
                self._send_json(400, {"error": f"Invalid base64: {e}"})
                return

        if self.path == "/grounding":
            query = body.get("query", "").strip()
            if query:
                prompt = f"Find and locate: {query}. Output only the matching item with its bounding box."
            else:
                prompt = "List all the UI items with their bounding box coordinates."
            # Use more tokens for full-screen, fewer for targeted (single element)
            max_tok = 256 if query else 1024
            raw = _infer(image_bytes, prompt, max_tokens=max_tok)
            self._send_json(200, {"raw": raw})

        elif self.path == "/referring":
            bbox = body.get("bbox", [])
            if len(bbox) != 4:
                self._send_json(400, {"error": "bbox must be [x1,y1,x2,y2]"})
                return
            x1, y1, x2, y2 = bbox
            raw = _infer(image_bytes, f"Describe the region ({x1}, {y1}),({x2}, {y2})", max_tokens=256)
            self._send_json(200, {"description": raw})

        else:
            self._send_json(404, {"error": "unknown endpoint"})


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="UI-UG-7B HTTP inference server")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    load_model(args.model_path)

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[ui_server] Listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[ui_server] Shutting down", flush=True)


if __name__ == "__main__":
    main()
