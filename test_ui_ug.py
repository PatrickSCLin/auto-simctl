"""
Standalone test for UI-UG-7B-2601 via mlx-vlm.

Usage:
    python3 test_ui_ug.py                    # uses /tmp/test_ui_ug.png
    python3 test_ui_ug.py path/to/image.png  # custom image
    python3 test_ui_ug.py --screenshot       # take fresh screenshot from booted simulator
"""
import sys
import time
import subprocess
import tempfile
import os
from pathlib import Path

MODEL_PATH = str(Path.home() / ".cache/huggingface/hub/ui-ug-7b-2601-4bit")
DEFAULT_IMAGE = "/tmp/test_ui_ug.png"

# ── args ──────────────────────────────────────────────────────────────────────
take_screenshot = "--screenshot" in sys.argv
image_path = DEFAULT_IMAGE
for arg in sys.argv[1:]:
    if not arg.startswith("--") and Path(arg).exists():
        image_path = arg

if take_screenshot:
    print("[..] Taking screenshot from booted simulator...")
    result = subprocess.run(
        ["xcrun", "simctl", "io", "booted", "screenshot", DEFAULT_IMAGE],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERR] simctl screenshot failed: {result.stderr}")
        sys.exit(1)
    image_path = DEFAULT_IMAGE
    print(f"[ok] Screenshot saved: {image_path}")

if not Path(image_path).exists():
    print(f"[ERR] Image not found: {image_path}")
    print("Run with --screenshot to take a fresh one, or provide a path.")
    sys.exit(1)

print(f"[..] Image: {image_path}  ({Path(image_path).stat().st_size // 1024} KB)")

# ── load model ────────────────────────────────────────────────────────────────
print(f"\n[..] Loading UI-UG-7B from {MODEL_PATH}...")
t0 = time.time()
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
model, processor = load(MODEL_PATH)
print(f"[ok] Model loaded in {time.time()-t0:.1f}s")

# ── helper ────────────────────────────────────────────────────────────────────
def infer(prompt: str, max_tokens: int = 1024) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    formatted = apply_chat_template(processor, config=model.config, prompt=messages)
    t = time.time()
    print(f"  [..] Running inference (max_tokens={max_tokens})...")
    output = generate(model, processor, image=image_path, prompt=formatted,
                      max_tokens=max_tokens, verbose=False)
    print(f"  [ok] Done in {time.time()-t:.1f}s")
    return output

# ── test 1: grounding ─────────────────────────────────────────────────────────
print("\n─── Test 1: Grounding (List all UI items) ───────────────────────────────")
raw = infer("List all the UI items.", max_tokens=512)
print(f"\nRAW OUTPUT:\n{raw}\n")

# ── test 2: short describe ────────────────────────────────────────────────────
print("─── Test 2: Describe entire screen ─────────────────────────────────────")
raw2 = infer("Describe what you see on this screen in one sentence.", max_tokens=128)
print(f"\nRAW OUTPUT:\n{raw2}\n")

print("─── Done ────────────────────────────────────────────────────────────────")
