"""
DeviceBridge — unified API over Android (adb) and iOS (idb) backends.

Auto-routes all calls to the correct backend based on device type.

Coordinate system
-----------------
UI-UG (Qwen2.5-VL) returns coordinates in 0-1000 NORMALIZED space.
idb tap / swipe expects LOGICAL POINTS (device resolution).

We track the last screenshot dimensions so we can auto-convert whenever
an action uses normalized coords (stored in Action.x/y/x2/y2 as ints in
0-1000 range, with action.coord_space == "norm1000").

Conversion:
  pt = round(norm * device_pts / 1000)
"""
from __future__ import annotations

from typing import Optional

from .backends.adb_backend import AdbBackend
from .backends.idb_backend import IdbBackend
from .models import Action, DeviceInfo, DeviceType, Screenshot, UIElement
from .screen import ScreenSpec, get_screen_spec, spec_from_screenshot


class DeviceBridge:
    """
    Single entry point for all device operations.
    Automatically routes to AdbBackend (Android) or IdbBackend (iOS).
    """

    def __init__(self) -> None:
        self._adb = AdbBackend()
        self._idb = IdbBackend()
        self._device_cache: dict[str, DeviceInfo] = {}
        # last screenshot size per device: {udid: (width_px, height_px)}
        self._last_screenshot_size: dict[str, tuple[int, int]] = {}

    # ── Internal routing ───────────────────────────────────────────────────────

    def _backend(self, udid: str):
        info = self._device_cache.get(udid)
        if info is None:
            # Try to find the device by refreshing the cache
            self._refresh_cache()
            info = self._device_cache.get(udid)
        if info is None:
            raise ValueError(f"Device not found: {udid}. Run list_devices() first.")
        return self._adb if info.device_type == DeviceType.ANDROID else self._idb

    def _refresh_cache(self) -> None:
        for dev in self._adb.list_devices() + self._idb.list_devices():
            self._device_cache[dev.udid] = dev

    # ── Discovery ──────────────────────────────────────────────────────────────

    def list_devices(self) -> list[DeviceInfo]:
        """Return all connected Android devices + available iOS Simulators."""
        devices = self._adb.list_devices() + self._idb.list_devices()
        self._device_cache = {d.udid: d for d in devices}
        return devices

    def first_device(self, prefer_booted: bool = True) -> Optional[DeviceInfo]:
        """Return the first available device, preferring booted simulators."""
        devices = self.list_devices()
        if not devices:
            return None
        if prefer_booted:
            from .models import DeviceState
            booted = [d for d in devices if d.state in (DeviceState.BOOTED, DeviceState.ONLINE)]
            return booted[0] if booted else devices[0]
        return devices[0]

    def boot_simulator(self, udid: str, wait_secs: int = 60) -> None:
        """
        Boot an iOS Simulator and ensure the Simulator.app window is visible.
        - If already booted: just opens/focuses the Simulator.app window.
        - If shutdown: boots it, opens Simulator.app, waits for SpringBoard.
        """
        import subprocess
        self._refresh_cache()
        info = self._device_cache.get(udid)
        if info is None or info.device_type != DeviceType.IOS:
            return
        from .models import DeviceState
        if info.state == DeviceState.BOOTED:
            # Already booted — just make sure the window is open
            subprocess.run(["open", "-a", "Simulator"], capture_output=True)
        else:
            self._idb.boot_simulator(udid, wait_secs=wait_secs)
        self._refresh_cache()

    def get_device(self, udid: str) -> DeviceInfo:
        self._refresh_cache()
        if udid not in self._device_cache:
            raise ValueError(f"Device not found: {udid}")
        return self._device_cache[udid]

    # ── Screenshot ─────────────────────────────────────────────────────────────

    def _screen_spec(self, udid: str) -> ScreenSpec:
        """Return the ScreenSpec for a device, using cached screenshot dims if available."""
        dev = self._device_cache.get(udid)
        device_name = dev.name if dev else ""

        ss = self._last_screenshot_size.get(udid)
        if ss:
            return spec_from_screenshot(ss[0], ss[1], device_name)
        if dev and dev.screen:
            return dev.screen
        return get_screen_spec(device_name)

    def screenshot(self, udid: str) -> Screenshot:
        shot = self._backend(udid).screenshot(udid)
        # Track screenshot pixel dimensions for accurate scale computation
        try:
            import struct
            data = shot.png_bytes
            if data[:4] == b"\x89PNG":
                w = struct.unpack(">I", data[16:20])[0]
                h = struct.unpack(">I", data[20:24])[0]
                self._last_screenshot_size[udid] = (w, h)
                shot = Screenshot(png_bytes=data, device_udid=udid, width=w, height=h)
        except Exception:
            pass
        return shot

    def _norm1000_to_pt(self, udid: str, nx: int, ny: int) -> tuple[int, int]:
        """Convert 0-1000 normalized coords (from UI-UG) to device logical points."""
        return self._screen_spec(udid).norm1000_to_pt(nx, ny)

    # ── Actions ────────────────────────────────────────────────────────────────

    def tap(self, udid: str, x: int, y: int) -> None:
        self._backend(udid).tap(udid, x, y)

    def swipe(
        self, udid: str,
        x1: int, y1: int, x2: int, y2: int,
        duration_ms: int = 300,
    ) -> None:
        self._backend(udid).swipe(udid, x1, y1, x2, y2, duration_ms)

    def input_text(self, udid: str, text: str) -> None:
        self._backend(udid).input_text(udid, text)

    def press_key(self, udid: str, key: str) -> None:
        self._backend(udid).press_key(udid, key)

    def launch_app(self, udid: str, app_id: str) -> None:
        self._backend(udid).launch_app(udid, app_id)

    def open_url(self, udid: str, url: str) -> None:
        backend = self._backend(udid)
        if hasattr(backend, "open_url"):
            backend.open_url(udid, url)

    def approve_permissions(self, udid: str, bundle_id: str, permissions: list[str]) -> None:
        backend = self._backend(udid)
        if hasattr(backend, "approve_permissions"):
            backend.approve_permissions(udid, bundle_id, permissions)

    def dump_ui(self, udid: str) -> str:
        """Return raw accessibility/UI hierarchy as string (XML for Android, JSON for iOS)."""
        return self._backend(udid).dump_ui(udid)

    def get_scroll_info(self, udid: str) -> dict:
        """
        Return scroll boundary info: whether content exists above/below/left/right
        the current viewport, and the estimated total content size in logical pts.
        """
        backend = self._backend(udid)
        if hasattr(backend, "get_scroll_info"):
            spec = self._screen_spec(udid)
            return backend.get_scroll_info(udid,
                                           screen_pt_h=spec.pt_h,
                                           screen_pt_w=spec.pt_w)
        return {}

    def list_elements(self, udid: str) -> list[dict]:
        """
        Return all labeled accessibility elements on screen as a clean list.
        Each entry: {label, type, cx, cy, x, y, width, height}
        Sorted top-to-bottom, left-to-right.
        Returns [] if accessibility is unavailable or screen has no labeled elements.
        """
        backend = self._backend(udid)
        if hasattr(backend, "list_elements"):
            return backend.list_elements(udid)
        return []

    def detect_system_dialog(self, udid: str) -> Optional[dict]:
        """
        Check if a system alert/permission dialog is currently on screen.
        Returns dialog info dict or None.  See IdbBackend.detect_system_dialog().
        """
        backend = self._backend(udid)
        if hasattr(backend, "detect_system_dialog"):
            return backend.detect_system_dialog(udid)
        return None

    def find_element_by_label(self, udid: str, keyword: str) -> Optional[dict]:
        """
        Lookup an element by accessibility label keyword.
        Returns {"label", "cx", "cy", "x", "y", "width", "height"} in LOGICAL POINTS,
        or None if not found.
        This is faster and more accurate than UI-UG for named elements.
        """
        backend = self._backend(udid)
        if hasattr(backend, "find_element_by_label"):
            return backend.find_element_by_label(udid, keyword)
        return None

    def get_foreground_app(self, udid: str) -> Optional[dict]:
        """
        Return the current foreground app (iOS only).

        Uses idb list-apps --fetch-process-state; returns the app with
        process_state "Running". Helps Qwen know "we're in X app" for done detection.

        Returns:
            {"bundle_id": str, "name": str} or None if unavailable / Android.
        """
        backend = self._backend(udid)
        if hasattr(backend, "get_foreground_app"):
            return backend.get_foreground_app(udid)
        return None

    # ── Action dispatcher ──────────────────────────────────────────────────────

    def execute(self, udid: str, action: Action) -> None:
        """
        Execute any Action object on the specified device.

        Coordinates in actions derived from UI-UG grounding results are stored
        in 0-1000 normalized space.  We detect this and auto-convert to device
        logical points before sending to idb / adb.
        """
        def _resolve(nx: Optional[int], ny: Optional[int]) -> tuple[int, int]:
            """Convert coords if they look like 0-1000 normalized."""
            if nx is None or ny is None:
                return 0, 0
            # Heuristic: if both axes ≤ 1000 and the device logical size is
            # larger, treat as norm1000.  Qwen sometimes outputs real pts too,
            # so only convert when coords can't be real pts (real pts for
            # iPhone are at most ~932 which overlaps — so we always convert
            # when the action has coord_space flag or when coords fit norm range
            # AND the reasoning mentions "norm" or came from auto-pick).
            reasoning = (action.reasoning or "").lower()
            from_norm = (
                "norm1000" in reasoning
                or "auto-picked" in reasoning
                or getattr(action, "_from_grounding", False)
            )
            if from_norm:
                return self._norm1000_to_pt(udid, nx, ny)
            return nx, ny

        # Clamp helper: ensure coordinates stay within the device logical screen.
        def _clamp(x: int, y: int) -> tuple[int, int]:
            spec = self._screen_spec(udid)
            w = spec.pt_w or 402
            h = spec.pt_h or 874
            return max(0, min(x, w)), max(0, min(y, h))

        if action.action_type == "tap":
            x, y = _resolve(action.x, action.y)
            x, y = _clamp(x, y)
            self.tap(udid, x, y)
        elif action.action_type == "swipe":
            x1, y1 = _resolve(action.x, action.y)
            x2, y2 = _resolve(action.x2, action.y2)
            x1, y1 = _clamp(x1, y1)
            x2, y2 = _clamp(x2, y2)
            self.swipe(udid, x1, y1, x2, y2, action.duration_ms)
        elif action.action_type == "input_text":
            self.input_text(udid, action.text)
        elif action.action_type == "press_key":
            self.press_key(udid, action.key)
        elif action.action_type == "launch_app":
            self.launch_app(udid, action.app_id)
        elif action.action_type in ("done", "error"):
            pass
        else:
            raise ValueError(f"Unknown action type: {action.action_type}")
