"""
Test: UI Agent understanding of dialog vs normal screens
Verifies:
  1. Accessibility-based dialog detection accuracy
  2. UI-UG visual grounding on dialog buttons (does it find them?)
  3. UI-UG visual grounding on normal app screens
  4. Chinese→English label bridge in find_element_by_label
  5. Coordinate accuracy for both accessibility and UI-UG results

Usage:
  python3 test_dialog_understanding.py
"""
import sys
import json
import base64
import struct
import zlib
from pathlib import Path

ASSETS = Path("/Users/patrick/.cursor/projects/Users-patrick-Desktop-auto-simctl/assets")
DIALOG_SCREENSHOT = ASSETS / "Screenshot_2026-03-14_at_7.33.38_AM-e584e3dd-ea30-44b8-b28b-8bda67d9d4d8.png"
HOME_SCREENSHOT   = ASSETS / "Simulator_Screenshot_-_iPhone_16_Pro_-_2026-03-14_at_09.06.56-11329d34-2054-4c49-a5d8-8f756b4d44ae.png"

UDID = "90D0D4F0-677D-447B-8DE9-AC2074D28214"
UI_SERVER_URL = "http://127.0.0.1:8081"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _img_wh(path: Path) -> tuple[int, int]:
    """Read image dimensions without PIL — handles PNG and JPEG."""
    data = path.read_bytes()
    if data[:4] == b"\x89PNG":
        # PNG: width/height at bytes 16-24
        return struct.unpack(">II", data[16:24])
    elif data[:2] == b"\xff\xd8":
        # JPEG: scan for SOF marker
        i = 2
        while i < len(data) - 8:
            if data[i] != 0xFF:
                break
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # SOF0/1/2
                h = struct.unpack(">H", data[i + 5:i + 7])[0]
                w = struct.unpack(">H", data[i + 7:i + 9])[0]
                return w, h
            seg_len = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seg_len
    raise ValueError(f"Cannot read dimensions from {path}")


def _img_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"


def _to_png_bytes(img_path: Path) -> bytes:
    """Convert image to PNG bytes using sips (macOS built-in) if needed."""
    if img_path.suffix.lower() == ".png":
        return img_path.read_bytes()
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    subprocess.run(["sips", "-s", "format", "png", str(img_path), "--out", tmp],
                   capture_output=True, check=True)
    data = Path(tmp).read_bytes()
    Path(tmp).unlink(missing_ok=True)
    return data


def _call_ui_server(img_path: Path, query: str = "") -> dict:
    import urllib.request
    img_bytes = _to_png_bytes(img_path)
    payload = json.dumps({
        "image_base64": base64.b64encode(img_bytes).decode(),
        "query": query,
    }).encode()
    req = urllib.request.Request(
        f"{UI_SERVER_URL}/grounding",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def sep(title: str = ""):
    width = 70
    if title:
        print(f"\n{'─'*4} {title} {'─'*(width - len(title) - 6)}")
    else:
        print("─" * width)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Accessibility dialog detection
# ──────────────────────────────────────────────────────────────────────────────

def test_accessibility_dialog():
    sep("TEST 1: Accessibility dialog detection (live device)")
    from mdb.backends.idb_backend import IdbBackend
    idb = IdbBackend()

    # First check current state
    d = idb.detect_system_dialog(UDID)
    if d:
        print(f"✓ Dialog detected: type={d['type']!r}")
        print(f"  message: {d['message'][:80]!r}")
        print(f"  dismiss: {d['dismiss_label']!r}")
        print("  buttons:")
        for b in d["buttons"]:
            print(f"    • '{b['label']}' → ({b['cx']}, {b['cy']})")
    else:
        print("ℹ No dialog on live device (expected — dialog was already dismissed)")
        print("  Testing with raw accessibility tree dump instead:")
        raw_tree = idb.dump_ui(UDID)
        data = json.loads(raw_tree)
        print(f"  Elements on screen: {len(data) if isinstance(data,list) else '?'}")
        for n in (data if isinstance(data, list) else [data]):
            t = n.get("type", "")
            l = (n.get("AXLabel") or "").strip()
            f = n.get("frame", {})
            cx = round(f.get("x", 0) + f.get("width", 0) / 2)
            cy = round(f.get("y", 0) + f.get("height", 0) / 2)
            print(f"    [{t:20s}] {l!r:40s} center=({cx},{cy})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: UI-UG grounding on dialog screenshot
# ──────────────────────────────────────────────────────────────────────────────

def test_uiug_on_dialog_screenshot():
    sep("TEST 2: UI-UG grounding on dialog screenshot")
    from mdb.screen import spec_from_screenshot
    from agents.ui_agent import UIAgent

    if not DIALOG_SCREENSHOT.exists():
        print(f"✗ Dialog screenshot not found: {DIALOG_SCREENSHOT}")
        return

    w, h = _img_wh(DIALOG_SCREENSHOT)
    spec = spec_from_screenshot(w, h, "iPhone 16 Pro")
    print(f"Screenshot: {w}×{h}px  spec={spec.pt_w}×{spec.pt_h}pt @{spec.scale}x")
    print(f"Image path: {DIALOG_SCREENSHOT.name}")

    agent = UIAgent(server_url=UI_SERVER_URL)

    queries = [
        "",                          # full screen listing
        "Don't Allow button",        # specific dismiss button
        "Allow Once button",         # permission button
        "dialog buttons",            # all dialog buttons
    ]

    for q in queries:
        print(f"\n  Query: {q!r}")
        try:
            result = _call_ui_server(DIALOG_SCREENSHOT, q)
            raw = result.get("raw", "")
            print(f"  Raw (first 200): {raw[:200]!r}")
            elements = agent._parse_grounding(raw)
            print(f"  Parsed: {len(elements)} element(s)")
            for el in elements[:5]:
                # Convert norm1000 → logical points
                if el.description == "norm1000":
                    cx_n = (el.bbox[0] + el.bbox[2]) / 2
                    cy_n = (el.bbox[1] + el.bbox[3]) / 2
                    cx_pt, cy_pt = spec.norm1000_to_pt(cx_n, cy_n)
                    print(f"    {el.label!r:35s}  norm=({cx_n:.0f},{cy_n:.0f})  pt=({cx_pt},{cy_pt})")
                else:
                    print(f"    {el.label!r:35s}  bbox={el.bbox}  desc={el.description!r}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Ground truth from accessibility (stored from earlier run)
    print("\n  Ground truth (accessibility buttons):")
    gt = [
        {"label": "Allow Once",            "cx": 201, "cy": 474},
        {"label": "Allow While Using App", "cx": 201, "cy": 518},
        {"label": "Don't Allow",           "cx": 201, "cy": 563},
    ]
    for b in gt:
        print(f"    {b['label']!r:35s}  pt=({b['cx']},{b['cy']})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: UI-UG grounding on home screen (Chinese queries)
# ──────────────────────────────────────────────────────────────────────────────

def test_uiug_on_home_screen():
    sep("TEST 3: UI-UG grounding on home screen (錢包/Wallet)")
    from mdb.screen import spec_from_screenshot
    from agents.ui_agent import UIAgent

    if not HOME_SCREENSHOT.exists():
        print(f"✗ Home screenshot not found: {HOME_SCREENSHOT}")
        return

    w, h = _img_wh(HOME_SCREENSHOT)
    spec = spec_from_screenshot(w, h, "iPhone 16 Pro")
    print(f"Screenshot: {w}×{h}px  spec={spec.pt_w}×{spec.pt_h}pt @{spec.scale}x")

    agent = UIAgent(server_url=UI_SERVER_URL)

    queries = [
        "Wallet app icon",      # English
        "錢包 app icon",         # Chinese
        "Settings app icon",
        "設定 app icon",
    ]

    for q in queries:
        print(f"\n  Query: {q!r}")
        try:
            result = _call_ui_server(HOME_SCREENSHOT, q)
            raw = result.get("raw", "")
            elements = agent._parse_grounding(raw)
            print(f"  Parsed: {len(elements)} element(s)  raw={raw[:120]!r}")
            for el in elements[:3]:
                if el.description == "norm1000":
                    cx_n = (el.bbox[0] + el.bbox[2]) / 2
                    cy_n = (el.bbox[1] + el.bbox[3]) / 2
                    cx_pt, cy_pt = spec.norm1000_to_pt(cx_n, cy_n)
                    print(f"    {el.label!r:35s}  norm=({cx_n:.0f},{cy_n:.0f})  pt=({cx_pt},{cy_pt})")
                else:
                    print(f"    {el.label!r:35s}  bbox={el.bbox}")
        except Exception as e:
            print(f"  ✗ Error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Accessibility label bridge (Chinese → English)
# ──────────────────────────────────────────────────────────────────────────────

def test_accessibility_cn_bridge():
    sep("TEST 4: Accessibility label bridge (Chinese → English)")
    from mdb.backends.idb_backend import IdbBackend
    idb = IdbBackend()

    # First go home so we can test home screen element lookup
    import subprocess
    subprocess.run(["idb", "ui", "button", "HOME", "--udid", UDID],
                   capture_output=True)
    import time; time.sleep(1)

    queries_cn = [
        ("錢包", "Wallet"),
        ("設定", "Settings"),
        ("相片", "Photos"),
        ("地圖", "Maps"),
        ("錢包 app icon", "Wallet"),
        ("Settings app icon", "Settings"),
    ]

    for query, expected in queries_cn:
        result = idb.find_element_by_label(UDID, query)
        if result:
            print(f"  ✓ {query!r:25s} → found: {result['label']!r:20s} at ({result['cx']},{result['cy']})")
        else:
            print(f"  ✗ {query!r:25s} → NOT FOUND  (expected: {expected!r})")


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Reasoning responsibility analysis
# ──────────────────────────────────────────────────────────────────────────────

def test_reasoning_responsibility():
    sep("TEST 5: Who should reason? — Screen state analysis")
    from mdb.backends.idb_backend import IdbBackend
    idb = IdbBackend()

    raw = idb.dump_ui(UDID)
    data = json.loads(raw)
    elements = data if isinstance(data, list) else [data]

    types = {}
    for el in elements:
        t = el.get("type", "?")
        types[t] = types.get(t, 0) + 1

    print(f"  Current screen elements: {len(elements)} total")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:20s} × {count}")

    # Heuristics for screen state
    labels = [el.get("AXLabel", "") or "" for el in elements]
    types  = [el.get("type", "") for el in elements]

    has_alert       = "Alert" in types
    has_dialog_btns = any(
        l.lower().replace("\u2019", "'") in {"allow once", "don't allow", "ok", "cancel"}
        for l in labels
    )
    # Home screen: iOS represents app icons as Buttons with app-name labels
    # A home screen typically has many Buttons and very few other types
    n_buttons = types.count("Button")
    n_non_app = sum(1 for t in types if t not in ("Button", "Application", "Slider", "StaticText"))
    has_home_screen = n_buttons >= 4 and n_non_app == 0
    has_nav_bar     = "NavigationBar" in types
    heading         = next((l for l in labels if l and l.strip()), "(none)")
    app_name        = next((el.get("AXLabel","") for el in elements
                            if el.get("type") == "Application"), "")

    print(f"\n  Screen classification heuristics:")
    print(f"    has_alert_type   = {has_alert}")
    print(f"    has_dialog_btns  = {has_dialog_btns}")
    print(f"    has_home_screen  = {has_home_screen}  (n_buttons={n_buttons})")
    print(f"    has_nav_bar      = {has_nav_bar}")
    print(f"    app_name         = {app_name!r}")
    print(f"    heading/title    = {heading!r}")

    print(f"\n  Decision:")
    if has_alert or has_dialog_btns:
        print("    → ORCHESTRATOR handles: system dialog, auto-tap dismiss button")
    elif has_home_screen:
        print("    → ORCHESTRATOR handles: home screen, use accessibility tree for all taps (no UI-UG needed)")
        btns = [(el.get("AXLabel",""),
                 round(el.get("frame",{}).get("x",0)+el.get("frame",{}).get("width",0)/2),
                 round(el.get("frame",{}).get("y",0)+el.get("frame",{}).get("height",0)/2))
                for el in elements if el.get("type")=="Button"]
        print("      Available buttons:", [(l, cx, cy) for l, cx, cy in btns[:8]])
    elif has_nav_bar:
        print("    → QWEN reasons: inside app with navigation hierarchy")
    else:
        print("    → QWEN reasons: complex/unknown screen")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os; os.chdir(Path(__file__).parent)
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 70)
    print("  auto-simctl: UI Agent / Dialog Understanding Test")
    print("=" * 70)

    test_accessibility_dialog()
    test_uiug_on_dialog_screenshot()
    test_uiug_on_home_screen()
    test_accessibility_cn_bridge()
    test_reasoning_responsibility()

    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)
