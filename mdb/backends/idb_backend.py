"""
iOS Simulator backend using the `idb` CLI tool (fb-idb).

All actions go through subprocess calls to `idb` (installed via pip install fb-idb).
xcrun simctl is used as a fallback for boot/screenshot when idb fails.

Requires:
  pip install fb-idb
  brew tap facebook/fb && brew install idb-companion
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Optional

from ..models import Action, DeviceInfo, DeviceState, DeviceType, Screenshot

# Keys that use `idb ui button <name>`
_BUTTON_KEYS = {"HOME", "LOCK", "SIDE_BUTTON", "SIRI", "APPLE_PAY"}

# Keys that use `idb ui key <keycode>`
_KEY_MAP: dict[str, str] = {
    "ENTER":       "40",   # HID keycode for Return
    "VOLUME_UP":   "128",
    "VOLUME_DOWN": "129",
    "BACK":        "HOME",  # iOS has no BACK — map to HOME button
}

_IDB = "idb"


def _idb(*args: str, udid: str, check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run an idb CLI command with --udid."""
    cmd = [_IDB, *args, "--udid", udid]
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout)


def _xcrun(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["xcrun", *args], capture_output=True, text=True, check=True)


class IdbBackend:
    """Controls iOS Simulators via the idb CLI tool."""

    # ── Discovery ──────────────────────────────────────────────────────────────

    def list_devices(self) -> list[DeviceInfo]:
        """List iOS Simulators via xcrun simctl (most reliable source)."""
        from ..screen import get_screen_spec
        try:
            result = subprocess.run(
                ["xcrun", "simctl", "list", "devices", "--json"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

        devices: list[DeviceInfo] = []
        data = json.loads(result.stdout)
        for os_label, sim_list in data.get("devices", {}).items():
            for sim in sim_list:
                if sim.get("isAvailable", False):
                    state_raw = sim.get("state", "").lower()
                    state = DeviceState.BOOTED if state_raw == "booted" else DeviceState.SHUTDOWN
                    name = sim["name"]
                    devices.append(DeviceInfo(
                        udid=sim["udid"],
                        name=name,
                        device_type=DeviceType.IOS,
                        state=state,
                        os_version=os_label
                            .replace("com.apple.CoreSimulator.SimRuntime.", "")
                            .replace("-", " "),
                        model=sim.get("deviceTypeIdentifier", name),
                        screen=get_screen_spec(name),
                    ))
        return devices

    # ── Boot ───────────────────────────────────────────────────────────────────

    def boot_simulator(self, udid: str, wait_secs: int = 60) -> None:
        """Boot simulator and wait until SpringBoard is ready."""
        subprocess.run(["xcrun", "simctl", "boot", udid], capture_output=True)
        subprocess.run(["open", "-a", "Simulator"], capture_output=True)
        result = subprocess.run(
            ["xcrun", "simctl", "bootstatus", udid, "-b"],
            capture_output=True, text=True, timeout=wait_secs,
        )
        if result.returncode != 0:
            raise TimeoutError(
                f"Simulator {udid} did not finish booting: {result.stderr.strip()}"
            )

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def screenshot(self, udid: str) -> Screenshot:
        """Take screenshot. Primary: idb. Fallback: xcrun simctl io."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            result = _idb("screenshot", tmp, udid=udid, check=False)
            if result.returncode == 0 and os.path.getsize(tmp) > 0:
                return Screenshot(png_bytes=open(tmp, "rb").read(), device_udid=udid)
            # Fallback
            _xcrun("simctl", "io", udid, "screenshot", tmp)
            return Screenshot(png_bytes=open(tmp, "rb").read(), device_udid=udid)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # ── Touch / input ──────────────────────────────────────────────────────────

    def tap(self, udid: str, x: int, y: int) -> None:
        _idb("ui", "tap", str(x), str(y), udid=udid)

    def swipe(self, udid: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        duration_s = duration_ms / 1000.0
        _idb("ui", "swipe", str(x1), str(y1), str(x2), str(y2),
             "--duration", str(duration_s), udid=udid)

    def input_text(self, udid: str, text: str) -> None:
        """Type text. Uses idb for ASCII, falls back to pasteboard for Unicode."""
        if text.isascii():
            _idb("ui", "text", text, udid=udid)
        else:
            # idb ui text doesn't support Unicode — inject via simctl pasteboard
            import subprocess as _sp
            _sp.run(
                ["xcrun", "simctl", "pbcopy", udid],
                input=text.encode("utf-8"), check=True,
            )
            # Simulate Cmd+V paste
            _idb("ui", "key", "47", "--modifier", "command", udid=udid, check=False)

    def press_key(self, udid: str, key: str) -> None:
        normalized = _KEY_MAP.get(key.upper(), key.upper())
        if normalized in _BUTTON_KEYS:
            _idb("ui", "button", normalized, udid=udid)
        else:
            _idb("ui", "key", normalized, udid=udid)

    def launch_app(self, udid: str, bundle_id: str) -> None:
        """Launch app by bundle ID. If it looks like a URL/scheme, use open_url."""
        if "://" in bundle_id or (":" in bundle_id and not bundle_id.startswith("com.")):
            self.open_url(udid, bundle_id)
        else:
            _idb("launch", bundle_id, udid=udid)

    def get_foreground_app(self, udid: str) -> Optional[dict]:
        """
        Return the current foreground (running) app's bundle_id and name.

        Uses `idb list-apps --json --fetch-process-state` and returns the first
        app with process_state == "Running". On home screen this is often
        SpringBoard; when an app is open, that app is Running.

        Returns:
            {"bundle_id": str, "name": str} or None if unavailable.
        """
        result = _idb(
            "list-apps", "--json", "--fetch-process-state",
            udid=udid, check=False, timeout=15,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        try:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                app = json.loads(line)
                if app.get("process_state") == "Running":
                    return {
                        "bundle_id": app.get("bundle_id", ""),
                        "name": app.get("name", ""),
                    }
        except Exception:
            pass
        return None

    def open_url(self, udid: str, url: str) -> None:
        """
        Open a URL or custom scheme via xcrun simctl openurl.
        Falls back to idb open if simctl fails.
        """
        result = subprocess.run(
            ["xcrun", "simctl", "openurl", udid, url],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            # Fallback to idb
            _idb("open", url, udid=udid, check=False)

    def approve_permissions(self, udid: str, bundle_id: str, permissions: list[str]) -> None:
        """
        Pre-approve system permissions for an app.
        Supported: photos, camera, contacts, location, etc.
        """
        for perm in permissions:
            try:
                _idb("approve", bundle_id, perm, udid=udid, check=False)
            except Exception:
                pass

    # ── UI dump ────────────────────────────────────────────────────────────────

    def dump_ui(self, udid: str) -> str:
        """Return accessibility tree as JSON via idb ui describe-all."""
        result = _idb("ui", "describe-all", "--json", udid=udid, check=False)
        return result.stdout if result.returncode == 0 else ""

    def list_elements(self, udid: str) -> list[dict]:
        """
        Return all labeled elements from the accessibility tree as a clean list.

        Each entry:
          { "label": str, "type": str, "cx": int, "cy": int,
            "x": float, "y": float, "width": float, "height": float,
            "visible": bool }   ← False if element is off the visible screen area

        Only elements with a non-empty label are included.
        The list is ordered top-to-bottom, left-to-right (reading order).
        Includes off-screen elements so Qwen knows what content exists beyond the fold.
        """
        import json as _json
        raw = self.dump_ui(udid)
        if not raw:
            return []
        try:
            nodes = _json.loads(raw)
        except Exception:
            return []
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Try to infer the visible screen region from the Application node
        # (Application frame = full logical screen size)
        screen_h = 874.0  # iPhone 16 Pro default; overridden by Application node
        screen_w = 402.0
        for node in nodes:
            if node.get("type") == "Application":
                f = node.get("frame", {})
                if f.get("width") and f.get("height"):
                    screen_w = float(f["width"])
                    screen_h = float(f["height"])
                break

        elements: list[dict] = []
        for node in nodes:
            t     = node.get("type", "")
            label = (node.get("AXLabel") or node.get("title") or
                     node.get("AXValue") or "").strip()
            if not label:
                continue
            frame = node.get("frame", {})
            if not frame:
                continue
            x = float(frame.get("x", 0))
            y = float(frame.get("y", 0))
            w = float(frame.get("width", 0))
            h = float(frame.get("height", 0))
            cx = round(x + w / 2)
            cy = round(y + h / 2)
            # An element is "visible" if its center is within the screen bounds
            # (with small tolerance for status-bar / home-indicator overlap)
            visible = (0 <= cx <= screen_w) and (40 <= cy <= screen_h - 20)
            elements.append({
                "label":   label,
                "type":    t,
                "cx":      cx,
                "cy":      cy,
                "x":       x,   "y":     y,
                "width":   w,   "height": h,
                "visible": visible,
            })

        # Sort: top-to-bottom, then left-to-right (reading order)
        elements.sort(key=lambda e: (e["cy"], e["cx"]))
        return elements

    def get_scroll_info(self, udid: str, screen_pt_h: int = 874, screen_pt_w: int = 402) -> dict:
        """
        Derive scroll boundary information from the accessibility tree.

        iOS accessibility returns ALL elements in a scrollable list, including those
        that are off-screen. By comparing max/min element y with the screen height
        we can determine:
          - Whether there is content below the current view
          - Whether there is content above the current view
          - Estimated total scrollable content height

        Returns:
          {
            "has_content_above": bool,
            "has_content_below": bool,
            "has_content_left":  bool,
            "has_content_right": bool,
            "content_height_pt": int,   # total height of all content in logical pts
            "content_width_pt":  int,
            "visible_min_y": int,       # topmost visible element y
            "visible_max_y": int,       # bottommost visible element y
          }
        """
        elements = self.list_elements(udid)
        if not elements:
            return {
                "has_content_above": False, "has_content_below": False,
                "has_content_left":  False, "has_content_right": False,
                "content_height_pt": screen_pt_h, "content_width_pt": screen_pt_w,
                "visible_min_y": 0, "visible_max_y": screen_pt_h,
            }

        all_cy  = [el["cy"] for el in elements]
        all_cx  = [el["cx"] for el in elements]
        min_cy  = min(all_cy)
        max_cy  = max(all_cy)
        min_cx  = min(all_cx)
        max_cx  = max(all_cx)

        vis = [el for el in elements if el["visible"]]
        vis_min_y = min((el["cy"] for el in vis), default=0)
        vis_max_y = max((el["cy"] for el in vis), default=screen_pt_h)

        status_bar = 55   # elements above this are system UI, not content
        home_area  = screen_pt_h - 60

        return {
            "has_content_above": min_cy < status_bar - 20,
            "has_content_below": max_cy > home_area + 20,
            "has_content_left":  min_cx < 0,
            "has_content_right": max_cx > screen_pt_w,
            "content_height_pt": max(max_cy - min_cy, screen_pt_h),
            "content_width_pt":  max(max_cx - min_cx, screen_pt_w),
            "visible_min_y": vis_min_y,
            "visible_max_y": vis_max_y,
        }

    def detect_system_dialog(self, udid: str) -> Optional[dict]:
        """
        Detect if a system dialog / alert / permission sheet is currently shown.

        Parses the accessibility tree looking for:
          - Permission dialogs  (Allow / Don't Allow)
          - Alert dialogs       (OK / Cancel / Dismiss)
          - Confirmation sheets (Continue / Cancel)

        Returns a dict when a dialog is found:
          {
            "type": "permission" | "alert" | "sheet",
            "message": str,          # dialog title/body text
            "buttons": [             # available button labels (ordered top→bottom)
                {"label": "Allow Once",      "cx": int, "cy": int},
                {"label": "Allow While ...", "cx": int, "cy": int},
                {"label": "Don't Allow",     "cx": int, "cy": int},
            ],
            "dismiss_label": str,    # suggested safe-dismiss button label
          }
        Returns None if no dialog is detected.
        """
        import re as _re
        raw = self.dump_ui(udid)
        if not raw:
            return None

        # Parse structured children list from the JSON output
        try:
            import json as _json
            tree = _json.loads(raw)
            nodes = tree if isinstance(tree, list) else [tree]
        except Exception:
            nodes = []

        def _walk(node) -> list[dict]:
            """Flatten tree into list of {type, label, cx, cy}."""
            out = []
            t     = node.get("type", "")
            label = (node.get("AXLabel") or node.get("title") or
                     node.get("AXValue") or "").strip()
            frame = node.get("frame", {})
            if frame:
                x = frame.get("x", 0); y = frame.get("y", 0)
                w = frame.get("width", 0); h = frame.get("height", 0)
                out.append({
                    "type": t, "label": label,
                    "cx": round(x + w / 2), "cy": round(y + h / 2),
                })
            for child in node.get("children", []):
                out.extend(_walk(child))
            return out

        flat: list[dict] = []
        for n in nodes:
            flat.extend(_walk(n))

        # Find all buttons and text nodes
        buttons    = [el for el in flat if el["type"] == "Button" and el["label"]]
        text_nodes = [el for el in flat if el["type"] == "StaticText" and el["label"]]

        if not buttons:
            return None

        # ── Keyboard guard ────────────────────────────────────────────────────
        # If any button has a single-character label (a, b, q, w, ...) it's a
        # software keyboard, not a dialog. Never treat keyboards as dialogs.
        if any(len(b["label"].strip()) == 1 for b in buttons):
            return None

        # ── Max-button guard ──────────────────────────────────────────────────
        # Real alert/permission dialogs have at most 4 buttons.
        # More than that means it's a list, toolbar, or suggestions widget.
        if len(buttons) > 4:
            return None

        # Normalize labels: replace Unicode smart quotes/dashes with ASCII
        def _norm(s: str) -> str:
            return (s.lower()
                    .replace("\u2019", "'")   # right single quotation mark → '
                    .replace("\u2018", "'")   # left  single quotation mark → '
                    .replace("\u201c", '"')   # left  double quotation mark → "
                    .replace("\u201d", '"')   # right double quotation mark → "
                    .replace("\u2014", "-")   # em dash                     → -
                    .replace("\xa0", " "))    # non-breaking space          → space

        # ── Meaningful message guard ──────────────────────────────────────────
        # A real dialog always has a message/title in StaticText.
        # If all StaticText nodes look like app-name lists (Siri Suggestions area),
        # this is not a modal dialog — it's a contextual widget.
        meaningful_text = [
            t for t in text_nodes
            if (
                (len(t["label"]) > 15 and "?" in t["label"]) or   # question sentence
                any(kw in t["label"].lower() for kw in
                    ("allow", "enable", "permission", "access", "location",
                     "camera", "microphone", "notification", "dictation",
                     "contact", "your ", "this app", "would like"))
            )
        ]
        # Siri Suggestions widget has NO meaningful dialog text → skip it
        # BUT if we have clear permission/alert buttons, still catch it even
        # without matching text (e.g. pure OK/Cancel alert with no body)

        # Classify dialog type by button labels
        btn_labels = {_norm(b["label"]) for b in buttons}

        _PERMISSION_WORDS = {"allow", "allow once", "allow while using app",
                              "don't allow", "deny", "while using"}
        _ALERT_WORDS      = {"ok", "cancel", "dismiss", "close", "got it",
                              "not now", "later", "done", "confirm",
                              "delete", "remove", "enable dictation",
                              "enable", "sign in", "sign out", "update"}

        is_permission = bool(btn_labels & _PERMISSION_WORDS)
        # "continue" is intentionally removed from _ALERT_WORDS — it appears in
        # too many non-dialog contexts (Siri Suggestions, onboarding strips, etc.)
        is_alert      = bool(btn_labels & _ALERT_WORDS)

        if not (is_permission or is_alert):
            return None

        # If only "continue"-like words matched (not true alert verbs), require
        # meaningful text to avoid Siri Suggestions false-positives
        strong_alert_verbs = btn_labels & (_PERMISSION_WORDS |
                                           {"not now", "enable", "delete",
                                            "enable dictation", "sign in"})
        if not strong_alert_verbs and not meaningful_text:
            return None

        # Update button labels to normalized form for matching
        for b in buttons:
            b["_norm_label"] = _norm(b["label"])

        dialog_type = "permission" if is_permission else "alert"

        # Message = first StaticText lines
        message = " ".join(t["label"] for t in text_nodes[:2])

        # Suggest dismiss: prefer "Don't Allow" → "Cancel" → "OK" → last button
        _DISMISS_PRIORITY = ["don't allow", "cancel", "not now", "dismiss",
                              "close", "later", "ok", "got it"]
        dismiss_label = buttons[-1]["label"]  # fallback: last button
        for pref in _DISMISS_PRIORITY:
            match = next(
                (b["label"] for b in buttons if b.get("_norm_label", _norm(b["label"])) == pref),
                None
            )
            if match:
                dismiss_label = match
                break

        return {
            "type": dialog_type,
            "message": message,
            "buttons": buttons,
            "dismiss_label": dismiss_label,
        }

    def find_element_by_label(self, udid: str, keyword: str) -> Optional[dict]:
        """
        Search the accessibility tree for an element whose AXUniqueId or AXLabel
        matches `keyword` (case-insensitive).

        Matching rules (any = hit):
          - label  IS contained in keyword    ("Settings" in "Settings app icon") ✓
          - keyword IS contained in label     ("settings" in "com.apple.settings")
          - any single word in keyword appears in label

        Returns a dict with at minimum:
          { "label": str, "x": float, "y": float,
            "width": float, "height": float,
            "cx": float, "cy": float }   ← cx/cy are logical point center
        Returns None if not found.
        """
        import re as _re

        raw = self.dump_ui(udid)
        if not raw:
            return None

        kw = keyword.lower().strip()
        # Individual words (length > 2) for word-level matching
        kw_words = {w for w in kw.split() if len(w) > 2}

        def _matches(label: str) -> bool:
            lo = label.lower()
            return (
                lo in kw           # label ⊂ keyword: "settings" in "settings app icon"
                or kw in lo        # keyword ⊂ label: "settings" in "com.apple.settings"
                or any(w in lo for w in kw_words)  # any word: "settings" in "settings"
            )

        def _score(label: str) -> int:
            lo = label.lower()
            if lo == kw:          return 4   # exact match
            if lo in kw:          return 3   # label is substring of keyword
            if kw in lo:          return 2   # keyword is substring of label
            return sum(1 for w in kw_words if w in lo)  # word overlap

        # idb describe-all outputs JSON where frame can appear before OR after AXLabel.
        # idb describe-all outputs JSON where frame can appear before OR after AXLabel.
        # We collect ALL matching candidates and return the best-scored one.
        candidates: list[tuple[int, dict]] = []   # (score, result)

        # Pattern A: AXUniqueId appears right before frame
        for m in _re.finditer(
            r'"AXUniqueId"\s*:\s*"([^"]*)"'
            r'[^{]{0,20}"frame"\s*:\s*\{'
            r'[^}]*"y"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"x"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"width"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"height"\s*:\s*([\d.eE+\-]+)',
            raw,
        ):
            uid = m.group(1)
            if not _matches(uid):
                continue
            try:
                y = float(m.group(2)); x = float(m.group(3))
                w = float(m.group(4)); h = float(m.group(5))
                candidates.append((_score(uid), {
                    "label": uid, "x": x, "y": y,
                    "width": w, "height": h,
                    "cx": round(x + w / 2), "cy": round(y + h / 2),
                }))
            except (ValueError, IndexError):
                continue

        # Pattern B: frame appears before AXLabel
        for m in _re.finditer(
            r'"frame"\s*:\s*\{'
            r'[^}]*"y"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"x"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"width"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*"height"\s*:\s*([\d.eE+\-]+)'
            r'[^}]*\}[^"]{0,200}"AXLabel"\s*:\s*"([^"]*)"',
            raw,
        ):
            label_val = m.group(5)
            if not _matches(label_val):
                continue
            try:
                y = float(m.group(1)); x = float(m.group(2))
                w = float(m.group(3)); h = float(m.group(4))
                candidates.append((_score(label_val), {
                    "label": label_val, "x": x, "y": y,
                    "width": w, "height": h,
                    "cx": round(x + w / 2), "cy": round(y + h / 2),
                }))
            except (ValueError, IndexError):
                continue

        if not candidates:
            return None
        # Return the candidate with the highest score
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]
