"""
Test: UI-UG finds Settings icon + coordinate scale conversion.

Uses the provided simulator screenshot (470x1024 px) to verify:
1. UI-UG can locate Settings icon
2. Coordinates are correctly scaled from screenshot pixels → device logical points
3. The final tap coordinate lands on the Settings icon

Usage:
    python3 test_ui_ug_settings.py
    python3 test_ui_ug_settings.py --draw   # also save annotated image
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
_RAW_ASSET = Path(
    "/Users/patrick/.cursor/projects/Users-patrick-Desktop-auto-simctl/assets/"
    "Simulator_Screenshot_-_iPhone_16_Pro_-_2026-03-14_at_01.34.32"
    "-70deebe8-58ce-4b2a-a41f-2a72181852ac.png"
)
# Cursor saves simulator screenshots as JPEG regardless of extension.
# Normalise to PNG so UI-UG can read it.
SCREENSHOT_PATH = Path("/tmp/auto_simctl_test_screenshot.png")
if not SCREENSHOT_PATH.exists() or SCREENSHOT_PATH.stat().st_mtime < _RAW_ASSET.stat().st_mtime:
    from PIL import Image as _PILImage
    _PILImage.open(_RAW_ASSET).save(SCREENSHOT_PATH)
UI_SERVER_URL   = "http://127.0.0.1:8081"

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_screenshot_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of the PNG without PIL if possible."""
    import struct
    data = path.read_bytes()
    # PNG IHDR chunk is at offset 16, 4 bytes W + 4 bytes H
    if data[:4] == b"\x89PNG":
        w = struct.unpack(">I", data[16:20])[0]
        h = struct.unpack(">I", data[20:24])[0]
        return w, h
    raise ValueError("Not a PNG file")


def scale_element(el: dict, spec: "ScreenSpec") -> dict:
    """
    Convert a UI element's center from 0-1000 normalized to device logical points.
    UI-UG always returns 0-1000 normalized coords.
    """
    cx_norm, cy_norm = el.get("center", [0, 0])
    cx_pt, cy_pt = spec.norm1000_to_pt(cx_norm, cy_norm)
    return {**el, "center_pt": [cx_pt, cy_pt], "center_norm": [cx_norm, cy_norm]}


def call_grounding(image_bytes: bytes, query: str) -> dict:
    payload = json.dumps({
        "image_base64": base64.b64encode(image_bytes).decode(),
        "query": query,
    }).encode()
    req = urllib.request.Request(
        f"{UI_SERVER_URL}/grounding", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def find_best(elements: list[dict], keywords: list[str]) -> dict | None:
    """Pick element whose label best matches the given keywords."""
    kws = {k.lower() for k in keywords}

    def score(el: dict) -> int:
        label = el.get("label", "").lower()
        el_type = el.get("type", "").lower()
        return sum(1 for kw in kws if kw in label or kw in el_type)

    scored = sorted(elements, key=score, reverse=True)
    return scored[0] if scored else None


def draw_results(image_path: Path, elements: list[dict], best: dict | None,
                 sx: float, sy: float, out_path: Path) -> None:
    """Draw bounding boxes and center dots on the image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  (PIL not installed, skipping draw)")
        return

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    for el in elements:
        bbox = el.get("bbox", [])
        if len(bbox) == 4:
            color = "red" if el is best else "#00AAFF"
            draw.rectangle(bbox, outline=color, width=3)
            label = el.get("label", "")[:20]
            draw.text((bbox[0]+2, bbox[1]-14), label, fill=color)

    if best:
        cx, cy = best.get("center", [0, 0])
        r = 8
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill="red", outline="white", width=2)
        # Draw cross for logical tap point
        cx_pt = round(cx / sx)
        cy_pt = round(cy / sy)
        draw.text((cx+10, cy), f"tap({cx_pt}, {cy_pt})", fill="red")

    img.save(out_path)
    print(f"\n  Annotated image saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draw", action="store_true",
                        help="Save annotated PNG with bounding boxes")
    parser.add_argument("--query", default="Settings app icon",
                        help="Grounding query to test")
    args = parser.parse_args()

    # 1. Load screenshot + resolve ScreenSpec
    if not SCREENSHOT_PATH.exists():
        print(f"Screenshot not found: {SCREENSHOT_PATH}")
        sys.exit(1)

    image_bytes = SCREENSHOT_PATH.read_bytes()
    ss_w, ss_h  = get_screenshot_size(SCREENSHOT_PATH)

    sys.path.insert(0, str(Path(__file__).parent))
    from mdb.screen import spec_from_screenshot, get_screen_spec
    spec = spec_from_screenshot(ss_w, ss_h, "iPhone 16 Pro")

    sx = spec.scale
    sy = spec.scale

    print(f"Screenshot:    {ss_w} × {ss_h} px")
    print(f"ScreenSpec:    {spec}")
    print(f"Scale factor:  {spec.scale}x")
    print(f"Query:         {args.query!r}")
    print()

    # 2. Check server
    try:
        with urllib.request.urlopen(f"{UI_SERVER_URL}/health", timeout=3) as r:
            health = json.loads(r.read())
        print(f"UI server:     {health.get('status')}  model={health.get('model','?')}")
    except Exception as e:
        print(f"UI server not reachable: {e}")
        print("Start it with: python3 cli.py server start")
        sys.exit(1)
    print()

    # 3. Ground
    print(f"Calling UI-UG grounding: {args.query!r} ...")
    t0 = time.time()
    resp = call_grounding(image_bytes, args.query)
    elapsed = time.time() - t0
    print(f"Response in {elapsed:.1f}s")
    print(f"Raw output:\n  {resp.get('raw','')[:300]!r}")
    print()

    # 4. Parse elements (reuse UIAgent parser via import)
    sys.path.insert(0, str(Path(__file__).parent))
    from agents.ui_agent import UIAgent
    agent = UIAgent()
    elements_raw = agent._parse_grounding(resp.get("raw", ""))
    element_dicts = [e.to_dict() for e in elements_raw]

    print(f"Parsed {len(element_dicts)} elements:")
    scaled_elements = []
    for i, el in enumerate(element_dicts):
        sel = scale_element(el, spec)
        scaled_elements.append(sel)
        is_settings = "settings" in el.get("label","").lower()
        marker = " ★ SETTINGS" if is_settings else ""
        print(f"  [{i:2d}] type={el.get('type','?'):8s}  "
              f"label={el.get('label','?')!r:20s}  "
              f"norm={el.get('center',[0,0])}  "
              f"→ pt={sel['center_pt']}{marker}")

    # 5. Best match
    print()
    best_raw  = find_best(element_dicts, args.query.split())
    best_idx  = element_dicts.index(best_raw) if best_raw else -1
    best_scaled = scaled_elements[best_idx] if best_idx >= 0 else None

    if best_scaled:
        cx_pt, cy_pt = best_scaled["center_pt"]
        cx_norm, cy_norm = best_scaled["center_norm"]
        print(f"Best match:    [{best_idx}] label={best_raw.get('label')!r}")
        print(f"  UI-UG norm1000: ({cx_norm}, {cy_norm})")
        print(f"  Device pts:     ({cx_pt}, {cy_pt})  ← use for idb tap")
        print()
        print(f"Expected tap:  idb ui tap {cx_pt} {cy_pt} --udid <UDID>")
    else:
        print("No matching element found.")

    # 6. Annotate image
    if args.draw:
        out = Path("/tmp/ui_ug_test_settings.png")
        draw_results(SCREENSHOT_PATH, element_dicts,
                     best_raw, spec.scale, spec.scale, out)

    print()
    # 7. Cross-check with accessibility tree (ground truth)
    print("─" * 60)
    print("Cross-check: accessibility tree lookup")
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from mdb.backends.idb_backend import IdbBackend
        idb = IdbBackend()
        udid = "90D0D4F0-677D-447B-8DE9-AC2074D28214"
        acc = idb.find_element_by_label(udid, "settings")
        if acc:
            print(f"  ✓ Accessibility found: {acc['label']!r}")
            print(f"    Logical frame: x={acc['x']:.1f} y={acc['y']:.1f} "
                  f"w={acc['width']:.1f} h={acc['height']:.1f}")
            print(f"    Logical center: ({acc['cx']}, {acc['cy']})  ← CORRECT tap target")
            print()
            print(f"  idb ui tap {acc['cx']} {acc['cy']} --udid {udid}")
        else:
            print("  ✗ Not found in accessibility tree (simulator may not be on home screen)")
    except Exception as e:
        print(f"  Accessibility lookup error: {e}")

    # 8. Sanity check: is Settings visible in this screenshot at all?
    print()
    settings_found = any("settings" in e.get("label","").lower()
                         for e in element_dicts)
    if settings_found:
        print("✓ UI-UG also found Settings label.")
    else:
        print("✗ UI-UG did NOT find 'Settings' label — accessibility is the reliable source.")


if __name__ == "__main__":
    main()
