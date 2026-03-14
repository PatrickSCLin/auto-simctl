"""
ScreenSpec — device screen resolution and scale information.

Primary source: CoreSimulator simdevicetype plists (always accurate for simulators).
Fallback: hardcoded table for physical devices or when plists are unavailable.

Coordinate spaces used in this project:
  - physical_px  : raw pixel dimensions of the screenshot PNG
  - logical_pt   : UIKit logical points (what idb tap/swipe expects)
  - norm1000     : 0-1000 normalized (Qwen2.5-VL / UI-UG output)

Conversions:
  physical_px  →  logical_pt  :  divide by scale
  norm1000     →  logical_pt  :  multiply by logical_pt / 1000
  physical_px  →  norm1000    :  multiply by 1000 / physical_px
"""
from __future__ import annotations

import plistlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── ScreenSpec dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScreenSpec:
    """Screen specifications for a device model."""
    name:     str          # human-readable name, e.g. "iPhone 16 Pro"
    px_w:     int          # physical pixel width  (screenshot PNG width)
    px_h:     int          # physical pixel height (screenshot PNG height)
    scale:    float        # pixel density multiplier (2.0 or 3.0)
    pt_w:     int          # logical point width  (idb coordinate space)
    pt_h:     int          # logical point height (idb coordinate space)

    @classmethod
    def from_px_scale(cls, name: str, px_w: int, px_h: int, scale: float) -> "ScreenSpec":
        return cls(name=name, px_w=px_w, px_h=px_h, scale=scale,
                   pt_w=round(px_w / scale), pt_h=round(px_h / scale))

    def norm1000_to_pt(self, nx: float, ny: float) -> tuple[int, int]:
        """Convert 0-1000 normalized coords (UI-UG output) to logical points."""
        return round(nx * self.pt_w / 1000), round(ny * self.pt_h / 1000)

    def px_to_pt(self, px_x: float, px_y: float) -> tuple[int, int]:
        """Convert physical pixel coords to logical points."""
        return round(px_x / self.scale), round(px_y / self.scale)

    def px_to_norm1000(self, px_x: float, px_y: float) -> tuple[int, int]:
        """Convert physical pixel coords to 0-1000 normalized space."""
        return round(px_x * 1000 / self.px_w), round(px_y * 1000 / self.px_h)

    def __str__(self) -> str:
        return (f"{self.name}: {self.px_w}×{self.px_h}px @{self.scale}x "
                f"→ {self.pt_w}×{self.pt_h}pt")


# ── Plist loader ──────────────────────────────────────────────────────────────

_SIMDEVICETYPE_BASE = Path(
    "/Library/Developer/CoreSimulator/Profiles/DeviceTypes"
)


def _load_from_plists() -> dict[str, ScreenSpec]:
    """
    Build a lookup table from CoreSimulator simdevicetype plists.
    Keys: device name (lowercase, e.g. "iphone 16 pro") and
          simdevicetype identifier (e.g. "com.apple.coresimulator.simdevicetype.iphone-16-pro").
    """
    specs: dict[str, ScreenSpec] = {}
    if not _SIMDEVICETYPE_BASE.exists():
        return specs

    for plist_path in _SIMDEVICETYPE_BASE.glob(
        "*.simdevicetype/Contents/Resources/profile.plist"
    ):
        try:
            data = plistlib.loads(plist_path.read_bytes())
            w     = int(data.get("mainScreenWidth", 0))
            h     = int(data.get("mainScreenHeight", 0))
            scale = float(data.get("mainScreenScale", 0))
            if not (w and h and scale):
                continue
            name = plist_path.parent.parent.parent.stem  # strip .simdevicetype
            spec = ScreenSpec.from_px_scale(name, w, h, scale)
            specs[name.lower()] = spec
        except Exception:
            continue

    return specs


# ── Hardcoded fallback table ──────────────────────────────────────────────────
# Used when CoreSimulator plists are unavailable (CI, physical devices, etc.)
# Values: (px_w, px_h, scale)

_FALLBACK: dict[str, tuple[int, int, float]] = {
    # iPhone 17
    "iphone 17 pro max": (1320, 2868, 3.0),
    "iphone 17 pro":     (1206, 2622, 3.0),
    "iphone 17 plus":    (1290, 2796, 3.0),
    "iphone 17":         (1206, 2622, 3.0),
    "iphone air":        (1260, 2736, 3.0),
    # iPhone 16
    "iphone 16 pro max": (1320, 2868, 3.0),
    "iphone 16 pro":     (1206, 2622, 3.0),
    "iphone 16 plus":    (1290, 2796, 3.0),
    "iphone 16":         (1179, 2556, 3.0),
    "iphone 16e":        (1170, 2532, 3.0),
    # iPhone 15
    "iphone 15 pro max": (1290, 2796, 3.0),
    "iphone 15 pro":     (1179, 2556, 3.0),
    "iphone 15 plus":    (1290, 2796, 3.0),
    "iphone 15":         (1179, 2556, 3.0),
    # iPhone 14
    "iphone 14 pro max": (1290, 2796, 3.0),
    "iphone 14 pro":     (1179, 2556, 3.0),
    "iphone 14 plus":    (1284, 2778, 3.0),
    "iphone 14":         (1170, 2532, 3.0),
    # iPhone 13
    "iphone 13 pro max": (1284, 2778, 3.0),
    "iphone 13 pro":     (1170, 2532, 3.0),
    "iphone 13 mini":    (1080, 2340, 3.0),
    "iphone 13":         (1170, 2532, 3.0),
    # iPhone 12
    "iphone 12 pro max": (1284, 2778, 3.0),
    "iphone 12 pro":     (1170, 2532, 3.0),
    "iphone 12 mini":    (1080, 2340, 3.0),
    "iphone 12":         (1170, 2532, 3.0),
    # iPhone 11
    "iphone 11 pro max": (1242, 2688, 3.0),
    "iphone 11 pro":     (1125, 2436, 3.0),
    "iphone 11":         ( 828, 1792, 2.0),
    # iPhone SE
    "iphone se (3rd generation)": (750, 1334, 2.0),
    "iphone se (2nd generation)": (750, 1334, 2.0),
    "iphone se":                  (640, 1136, 2.0),
    # iPad Air
    "ipad air 13-inch (m3)":     (2048, 2732, 2.0),
    "ipad air 13-inch (m2)":     (2048, 2732, 2.0),
    "ipad air 11-inch (m3)":     (1640, 2360, 2.0),
    "ipad air 11-inch (m2)":     (1640, 2360, 2.0),
    # iPad Pro
    "ipad pro 13-inch (m4)":     (2064, 2752, 2.0),
    "ipad pro 11-inch (m4)":     (1668, 2420, 2.0),
    # iPad mini
    "ipad mini (a17 pro)":       (1488, 2266, 2.0),
    "ipad mini (6th generation)":(1488, 2266, 2.0),
}


# ── Module-level lookup table (built once at import time) ─────────────────────

_SPECS: dict[str, ScreenSpec] = _load_from_plists()

# Merge fallback for any missing entries
for _name, (_w, _h, _s) in _FALLBACK.items():
    if _name not in _SPECS:
        _SPECS[_name] = ScreenSpec.from_px_scale(_name, _w, _h, _s)

# Default fallback for unknown devices
_DEFAULT_SPEC = ScreenSpec.from_px_scale("unknown", 1179, 2556, 3.0)  # iPhone 15 Pro


# ── Public API ────────────────────────────────────────────────────────────────

def get_screen_spec(device_name: str) -> ScreenSpec:
    """
    Return ScreenSpec for a device.

    Args:
        device_name: Case-insensitive device name (e.g. "iPhone 16 Pro").

    Returns:
        ScreenSpec with px/pt/scale info. Falls back to _DEFAULT_SPEC if unknown.
    """
    key = device_name.lower().strip()

    # Exact match
    if key in _SPECS:
        return _SPECS[key]

    # Partial match (longest matching substring wins)
    candidates = [(name, spec) for name, spec in _SPECS.items() if name in key or key in name]
    if candidates:
        best_name, best_spec = max(candidates, key=lambda t: len(t[0]))
        return best_spec

    return _DEFAULT_SPEC


def spec_from_screenshot(px_w: int, px_h: int, device_name: str = "") -> ScreenSpec:
    """
    Return the ScreenSpec appropriate for a screenshot of given pixel dimensions.

    Logical point resolution (pt_w, pt_h) is ALWAYS the canonical device value
    — it never changes regardless of screenshot resolution.  Only the px_w/px_h
    and derived scale reflect the actual image.

    Priority:
      1. Exact pixel match in known specs
      2. Device name lookup → canonical pt_w/pt_h, recomputed scale from px dims
      3. Infer scale heuristically (3x for phones, 2x for tablets/downsampled)
    """
    # 1. Exact pixel match (real full-res simulator screenshot)
    for spec in _SPECS.values():
        if spec.px_w == px_w and spec.px_h == px_h:
            return spec

    # 2. Device name known → keep canonical logical resolution, update px dims
    if device_name:
        ref = get_screen_spec(device_name)
        if ref is not _DEFAULT_SPEC or device_name:
            # Compute the actual scale from the screenshot vs canonical pixels
            actual_scale = px_w / ref.pt_w if ref.pt_w else ref.scale
            return ScreenSpec(
                name=ref.name,
                px_w=px_w, px_h=px_h,
                scale=round(actual_scale, 4),
                pt_w=ref.pt_w, pt_h=ref.pt_h,   # logical res is always canonical
            )

    # 3. Heuristic: infer scale from pixel width
    scale = 3.0 if px_w > 900 and px_h > 1800 else 2.0
    return ScreenSpec.from_px_scale(device_name or "unknown", px_w, px_h, scale)


def all_specs() -> list[ScreenSpec]:
    """Return all known ScreenSpecs, sorted by name."""
    return sorted(set(_SPECS.values()), key=lambda s: s.name)
