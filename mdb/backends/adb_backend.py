"""
Android device backend using pure-python-adb.

Requires: pip install pure-python-adb
Requires: adb CLI in PATH (brew install android-platform-tools)
"""
from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from ..models import Action, DeviceInfo, DeviceState, DeviceType, Screenshot

if TYPE_CHECKING:
    pass

KEY_MAP: dict[str, str] = {
    "HOME": "3",
    "BACK": "4",
    "ENTER": "66",
    "LOCK": "26",
    "VOLUME_UP": "24",
    "VOLUME_DOWN": "25",
}


class AdbBackend:
    """Controls Android devices via pure-python-adb Python client."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5037) -> None:
        self._host = host
        self._port = port
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from ppadb.client import Client as PpadbClient
                self._client = PpadbClient(host=self._host, port=self._port)
            except ImportError as e:
                raise RuntimeError("pure-python-adb not installed. Run: pip install pure-python-adb") from e
        return self._client

    def _get_device(self, udid: str):
        client = self._get_client()
        for dev in client.devices():
            if dev.serial == udid:
                return dev
        raise ValueError(f"Android device not found: {udid}")

    # ── Discovery ──────────────────────────────────────────────────────────────

    def list_devices(self) -> list[DeviceInfo]:
        try:
            client = self._get_client()
            devices = client.devices()
        except Exception:
            # adb server not running — try to start it
            subprocess.run(["adb", "start-server"], capture_output=True)
            try:
                client = self._get_client()
                devices = client.devices()
            except Exception:
                return []

        result = []
        for dev in devices:
            serial = dev.serial
            try:
                model = dev.shell("getprop ro.product.model").strip()
                os_ver = dev.shell("getprop ro.build.version.release").strip()
            except Exception:
                model = ""
                os_ver = ""
            result.append(DeviceInfo(
                udid=serial,
                name=model or serial,
                device_type=DeviceType.ANDROID,
                state=DeviceState.ONLINE,
                os_version=os_ver,
                model=model,
            ))
        return result

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def screenshot(self, udid: str) -> Screenshot:
        dev = self._get_device(udid)
        png_bytes = dev.screencap()
        return Screenshot(png_bytes=png_bytes, device_udid=udid)

    # ── Actions ────────────────────────────────────────────────────────────────

    def tap(self, udid: str, x: int, y: int) -> None:
        dev = self._get_device(udid)
        dev.shell(f"input tap {x} {y}")

    def swipe(self, udid: str, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        dev = self._get_device(udid)
        dev.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, udid: str, text: str) -> None:
        dev = self._get_device(udid)
        # Replace spaces with %s for adb input text
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        dev.shell(f"input text '{escaped}'")

    def press_key(self, udid: str, key: str) -> None:
        keycode = KEY_MAP.get(key.upper())
        if keycode is None:
            raise ValueError(f"Unknown key: {key}. Valid: {list(KEY_MAP)}")
        dev = self._get_device(udid)
        dev.shell(f"input keyevent {keycode}")

    def launch_app(self, udid: str, package: str) -> None:
        dev = self._get_device(udid)
        # Try monkey first (launches main activity), fall back to am start
        result = dev.shell(f"monkey -p {package} -c android.intent.category.LAUNCHER 1")
        if "error" in result.lower():
            dev.shell(f"am start -n {package}")

    def dump_ui(self, udid: str) -> str:
        """Return the UI hierarchy as XML string (uiautomator dump)."""
        dev = self._get_device(udid)
        dev.shell("uiautomator dump /sdcard/ui_dump.xml")
        xml = dev.shell("cat /sdcard/ui_dump.xml")
        dev.shell("rm /sdcard/ui_dump.xml")
        return xml

    # ── Action dispatcher ──────────────────────────────────────────────────────

    def execute(self, udid: str, action: Action) -> None:
        if action.action_type == "tap":
            self.tap(udid, action.x, action.y)
        elif action.action_type == "swipe":
            self.swipe(udid, action.x, action.y, action.x2, action.y2, action.duration_ms)
        elif action.action_type == "input_text":
            self.input_text(udid, action.text)
        elif action.action_type == "press_key":
            self.press_key(udid, action.key)
        elif action.action_type == "launch_app":
            self.launch_app(udid, action.app_id)
        elif action.action_type in ("done", "error"):
            pass  # handled by orchestrator
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
