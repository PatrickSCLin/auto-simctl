"""
UIAgent — calls UI-UG-7B-2601 via local HTTP server (ui_server.py).

Preferred mode: HTTP client → calls ui_server running on localhost:8081.
Fallback mode:  in-process (for testing only, not recommended for production).

Always use `server start` to pre-load both models before running tasks.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from logger import get_logger
from mdb.models import UIElement

log = get_logger("ui_agent")

DEFAULT_MODEL_PATH = str(
    Path.home() / ".cache" / "huggingface" / "hub" / "ui-ug-7b-2601-4bit"
)
DEFAULT_SERVER_URL = "http://127.0.0.1:8081"

# Coordinate patterns emitted by UI-UG: "(x1, y1),(x2, y2)"
_COORD_RE = re.compile(r"\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)")
_ITEM_LINE_RE = re.compile(
    r'(?:(\w+)\s+)?["\']?([^"\'(]+?)["\']?\s+(?:at\s+)?\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)',
    re.IGNORECASE,
)


class UIAgent:
    """
    HTTP client for ui_server.py (UI-UG-7B-2601).
    Both models are loaded once by `server start` and stay in memory.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        model_path: str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.server_url = server_url
        self.model_path = model_path

    # ── Server health ──────────────────────────────────────────────────────────

    def server_running(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.server_url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    # ── HTTP calls ─────────────────────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.server_url}{endpoint}"
        body = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    # ── Public API ─────────────────────────────────────────────────────────────

    def grounding(self, png_bytes: bytes) -> list[UIElement]:
        """List ALL visible UI elements (no specific query)."""
        return self._grounding_call(png_bytes, query=None)

    def grounding_targeted(self, png_bytes: bytes, query: str) -> list[UIElement]:
        """
        Find a SPECIFIC element by description.
        Called by orchestrator when Qwen issues a 'ground' action.
        e.g. query = "Settings app icon on home screen"
        """
        log.info(f"UI targeted grounding: {query!r}")
        return self._grounding_call(png_bytes, query=query)

    def _grounding_call(self, png_bytes: bytes, query: Optional[str]) -> list[UIElement]:
        if not self.server_running():
            raise RuntimeError(
                "UI server not running. Start with: python3 cli.py server start"
            )
        label = f"query={query!r}" if query else "full-screen"
        log.info(f"UI grounding ({label}): {len(png_bytes)//1024}KB → ui_server")
        t0 = time.time()
        payload: dict = {"image_base64": base64.b64encode(png_bytes).decode()}
        if query:
            payload["query"] = query
        resp = self._post("/grounding", payload)
        log.debug(f"UI grounding response ({time.time()-t0:.1f}s): {resp.get('raw','')!r:.120}")
        elements = self._parse_grounding(resp.get("raw", ""))
        log.info(f"UI grounding done: {len(elements)} elements ({time.time()-t0:.1f}s)")
        return elements

    def referring(self, png_bytes: bytes, bbox: list[int]) -> str:
        """Describe a UI region. Calls /referring on ui_server."""
        if not self.server_running():
            raise RuntimeError("UI server not running. Start with: python3 cli.py server start")
        resp = self._post("/referring", {
            "image_base64": base64.b64encode(png_bytes).decode(),
            "bbox": bbox,
        })
        return resp.get("description", "").strip()

    # ── Parsing ────────────────────────────────────────────────────────────────
    #
    # UI-UG (Qwen2.5-VL based) returns coordinates in 0-1000 NORMALIZED space.
    # We store them as-is (0-1000) in UIElement.bbox.
    # Conversion to device logical points happens in DeviceBridge.execute()
    # using the actual screenshot dimensions and device scale.

    # Matches Qwen2.5-VL box tokens:  <|box_start|>(x1, y1),(x2, y2)<|box_end|>
    _BOX_TOKEN_RE = re.compile(
        r'<\|box_start\|>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*<\|box_end\|>'
    )

    def _parse_grounding(self, raw: str) -> list[UIElement]:
        # Strategy 1: Qwen2.5-VL JSON array with <|box_start|> tokens
        elements = self._try_parse_qwen_vlm_json(raw)
        if elements:
            return elements

        # Strategy 2: plain JSON array with numeric bbox arrays
        elements = self._try_parse_plain_json(raw)
        if elements:
            return elements

        # Strategy 3: line-by-line regex (legacy)
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = _ITEM_LINE_RE.search(line)
            if m:
                el_type = m.group(1) or ""
                label = m.group(2).strip().strip('"\'')
                x1, y1, x2, y2 = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
                elements.append(UIElement(label=label, bbox=[x1, y1, x2, y2], element_type=el_type.lower()))
            else:
                for c in _COORD_RE.findall(line):
                    x1, y1, x2, y2 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
                    label = re.sub(r'\(.*', '', line).strip().strip('"\'').strip()
                    elements.append(UIElement(label=label, bbox=[x1, y1, x2, y2]))
        return elements

    def _try_parse_qwen_vlm_json(self, raw: str) -> list[UIElement]:
        """
        Parse Qwen2.5-VL output format:
          [{"type":"...", "box":"<|box_start|>(x1,y1),(x2,y2)<|box_end|>", "text":"label"}, ...]
        Coordinates are in 0-1000 normalized space.
        Also handles truncated JSON (model hit max_tokens before closing the array).
        """
        start = raw.find("[")
        if start == -1:
            return []
        end = raw.rfind("]") + 1

        # Try full JSON first
        if end > start:
            try:
                data = json.loads(raw[start:end])
                if isinstance(data, list):
                    pass  # fall through to processing
                else:
                    data = None
            except Exception:
                data = None
        else:
            data = None

        # Repair truncated JSON: find all complete {...} objects
        if data is None:
            chunk = raw[start:]
            # Extract all complete JSON objects between { and matching }
            objects = []
            depth = 0
            obj_start = -1
            for i, ch in enumerate(chunk):
                if ch == "{":
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and obj_start != -1:
                        try:
                            obj = json.loads(chunk[obj_start:i+1])
                            objects.append(obj)
                        except Exception:
                            pass
                        obj_start = -1
            data = objects if objects else None

        if not data:
            return []
        if not isinstance(data, list):
            return []

        elements = []
        for item in data:
            if not isinstance(item, dict):
                continue
            box_str = item.get("box", "")
            if not isinstance(box_str, str):
                continue
            m = self._BOX_TOKEN_RE.search(box_str)
            if not m:
                continue
            x1, y1, x2, y2 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            label   = str(item.get("text") or item.get("label") or item.get("name") or "")
            el_type = str(item.get("type") or "")
            elements.append(UIElement(
                label=label,
                bbox=[x1, y1, x2, y2],   # 0-1000 normalized
                element_type=el_type.lower(),
                description="norm1000",   # flag: coords are 0-1000
            ))
        return elements

    def _try_parse_plain_json(self, raw: str) -> list[UIElement]:
        """Parse plain JSON arrays with numeric bbox/coordinates fields."""
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            data = json.loads(raw[start:end])
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        elements = []
        for item in data:
            if not isinstance(item, dict):
                continue
            bbox_raw = item.get("bbox") or item.get("coordinates", [])
            if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
                try:
                    bbox = [int(v) for v in bbox_raw]
                except (TypeError, ValueError):
                    continue
                label   = str(item.get("label") or item.get("text") or item.get("name") or "")
                el_type = str(item.get("type") or item.get("category") or "")
                elements.append(UIElement(label=label, bbox=bbox, element_type=el_type.lower()))
        return elements
