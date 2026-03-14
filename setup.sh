#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[ok]${NC}  $*"; }
info() { echo -e "${YELLOW}[..] ${NC} $*"; }
err()  { echo -e "${RED}[err]${NC} $*"; }

QWEN_MODEL_DIR="$HOME/.cache/huggingface/hub/qwen3.5-2b-mlx-4bit"
UIUG_MODEL_DIR="$HOME/.cache/huggingface/hub/ui-ug-7b-2601"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     auto-simctl setup installer      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Detect OS ──────────────────────────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    info "Detected macOS"
else
    OS="linux"
    info "Detected Linux"
fi

# ── 2. Homebrew packages (macOS only) ─────────────────────────────────────────
if [[ "$OS" == "macos" ]]; then
    if ! command -v brew &>/dev/null; then
        err "Homebrew not found. Install from https://brew.sh then re-run."
        exit 1
    fi

    info "Installing android-platform-tools (adb)..."
    if command -v adb &>/dev/null; then
        ok "adb already installed ($(adb version | head -1))"
    else
        brew install android-platform-tools
        ok "adb installed"
    fi

    info "Installing idb-companion (iOS Simulator control)..."
    if command -v idb_companion &>/dev/null; then
        ok "idb-companion already installed"
    else
        brew tap facebook/fb 2>/dev/null || true
        brew install idb-companion
        ok "idb-companion installed"
    fi
fi

# ── 3. Python deps ────────────────────────────────────────────────────────────
info "Installing Python dependencies..."
pip3 install --quiet --upgrade \
    "fb-idb>=1.1.0" \
    "pure-python-adb>=0.2.2.dev0" \
    "mlx-openai-server>=1.6.0" \
    "mlx-vlm>=0.4.0" \
    "openai>=1.0.0" \
    "typer>=0.9.0" \
    "rich>=13.0.0" \
    "fastmcp>=0.1.0" \
    "huggingface_hub>=0.20.0"
ok "Python dependencies installed"

# ── 4. Check / download Qwen3.5-2B-4bit ──────────────────────────────────────
info "Checking Qwen3.5-2B-4bit model..."
if [[ -f "$QWEN_MODEL_DIR/config.json" ]]; then
    ok "Qwen3.5-2B-4bit found at $QWEN_MODEL_DIR"
else
    info "Downloading Qwen3.5-2B-4bit (this may take a few minutes)..."
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os
local_dir = os.path.expanduser("~/.cache/huggingface/hub/qwen3.5-2b-mlx-4bit")
snapshot_download(repo_id="Qwen/Qwen3.5-2B-Instruct-MLX-4bit", local_dir=local_dir)
print("Download complete.")
PYEOF
    ok "Qwen3.5-2B-4bit downloaded"
fi

# ── 5. Check / download UI-UG-7B-2601 ────────────────────────────────────────
info "Checking UI-UG-7B-2601 model..."
if [[ -f "$UIUG_MODEL_DIR/config.json" ]]; then
    ok "UI-UG-7B-2601 found at $UIUG_MODEL_DIR"
else
    info "Downloading UI-UG-7B-2601 (~15GB, this will take a while)..."
    python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os
local_dir = os.path.expanduser("~/.cache/huggingface/hub/ui-ug-7b-2601")
snapshot_download(
    repo_id="neovateai/UI-UG-7B-2601",
    local_dir=local_dir,
    ignore_patterns=["*.md", "*.gitattributes"],
)
print("Download complete.")
PYEOF
    ok "UI-UG-7B-2601 downloaded"
fi

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║         Setup complete!              ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Models:"
echo "    Qwen3.5-2B-4bit : $QWEN_MODEL_DIR"
echo "    UI-UG-7B-2601   : $UIUG_MODEL_DIR"
echo ""
echo "  Usage:"
echo "    python3 cli.py devices          # list connected devices"
echo "    python3 cli.py run '<task>'     # run an AI task"
echo "    python3 cli.py server start     # start Qwen inference server"
echo ""
