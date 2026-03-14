# auto-simctl

**Intelligent Mobile Simulator Control** — the missing piece for vibe coding on mobile. An AI agent that controls real/simulated Android and iOS devices: screenshot → UI understanding → reasoning → action → report.

## What it does

- **Unified device bridge (MDB)**: One API over `adb` (Android) and `idb` (iOS Simulator). All coordinates clamped to screen bounds before execution.
- **Accessibility-first UI understanding**: `idb ui describe-all` provides precise logical-point `(cx, cy)` for every element. Qwen picks *which* element; the element table supplies *where*.
- **Qwen-as-director**: Qwen3.5-9B reasons from the accessibility tree + screenshot, decides actions, and bridges language semantics (e.g. Chinese task → English UI labels) without hardcoded translation tables.
- **Fast-paths (no LLM)**: For "打開 X app" tasks, the orchestrator short-circuits Qwen entirely — if the app icon is visible in elements, it taps directly; if elements already show in-app UI (Tab Bar + No Recents), it signals done.
- **Pre-flight home reset**: Before each task, presses HOME to exit any foreground app, then detects and corrects Today View / side launcher pages by swiping left to page 0.
- **UI-UG fallback**: UI-UG-7B-2601 via a background HTTP server handles custom-drawn views where accessibility labels are unavailable (games, canvas, WebView).
- **ReAct loop with navigation stack**: Screenshot → acc elements → fast-paths → Qwen → action → execute → update nav stack → repeat until done or max steps.
- **Navigation & scroll awareness**: Maintains a `NavFrame` stack (depth, screen label, scroll offset) so the agent always knows which page it's on and how far it has scrolled.
- **Dialog & keyboard handling**: Auto-detects and dismisses system permission dialogs; detects on-screen keyboard and switches to `input_text()` automatically.
- **Future**: MCP server so Cursor/Claude can call `run_task(task, device)` from the vibe coding loop.

## Quick start

```bash
# One-time setup: install adb, idb, Python deps, download models
./setup.sh

# Start both model servers (Qwen on :8080, UI-UG on :8081)
python3 cli.py server start

# Run a task on the first available booted simulator
python3 cli.py run "Open Settings"

# Run with verbose step-by-step output
python3 cli.py run "找看看有沒有資料夾" --verbose

# List connected devices
python3 cli.py devices

# Stop servers
python3 cli.py server stop
```

## Requirements

- macOS (Apple Silicon) — MLX-based models
- Python 3.10+
- Xcode + `idb-companion` (Homebrew) for iOS
- `android-platform-tools` (Homebrew) or Android Studio for Android
- Models in `~/.cache/huggingface/hub/`:
  - `qwen3.5-9b-mlx-4bit` (reasoning, ~8s/step with thinking)
  - `neovateai/UI-UG-7B-2601` (UI grounding fallback)

## How the agent thinks

```
Pre-flight (once per task):
  1. Press HOME if not on SpringBoard
  2. Detect Today View (>12 elements) → swipe left to page 0

For each step (max 20):
  1. Take screenshot
  2. List all accessibility elements (visible + off-screen)
  3. Detect keyboard open (single-letter buttons present)
  4. Get scroll boundary info (content above/below/left/right)
  5. Detect & auto-dismiss system dialogs
  6. Fast-paths (no Qwen):
     a. Elements show Tab Bar + No Recents + app label → done
     b. MDB foreground = target app + elements show in-app → done
     c. App icon Button visible in elements → direct tap(cx, cy)
  7. Qwen phase-1: analyze screenshot + elements → decide action
     └─ If action = "ground": accessibility elements passed to Qwen phase-2
        (UI-UG called only if no accessibility labels at all)
  8. Snap out-of-bounds tap to nearest accessible element
  9. Execute action via MDB (coords clamped to screen bounds)
  10. Update navigation stack (NavFrame + ScrollState)
  11. Dead-end check: same action 3× → force HOME
```

**Key design principle**: Accessibility elements carry `(cx, cy)` in logical points — Qwen decides *which* element, the element table provides *where*. Qwen never needs to estimate coordinates from the screenshot for standard iOS apps.

## Project layout

```
auto-simctl/
├── cli.py                  # Entry point: run / devices / server start|stop|status
├── ui_server.py            # UI-UG-7B HTTP server (port 8081)
├── logger.py               # Structured logging helpers
│
├── mdb/                    # Mobile Device Bridge
│   ├── bridge.py           # DeviceBridge unified API + coord clamping
│   ├── screen.py           # ScreenSpec: pixel ↔ logical-pt ↔ norm1000
│   ├── models.py           # DeviceInfo, Action, Screenshot dataclasses
│   └── backends/
│       ├── idb_backend.py  # iOS: screenshot, tap, swipe, input_text,
│       │                   #   list_elements, get_foreground_app,
│       │                   #   get_scroll_info, detect_system_dialog
│       └── adb_backend.py  # Android: same interface via adb
│
├── agents/
│   ├── qwen_agent.py       # Qwen3.5-9B via mlx-openai-server; adaptive thinking
│   ├── ui_agent.py         # UI-UG-7B-2601 client (HTTP → port 8081)
│   └── prompts.py          # SYSTEM_PROMPT + build_user_message
│
├── orchestrator/
│   ├── loop.py             # Pre-flight, fast-paths, ReAct loop, nav stack
│   └── result.py           # TaskResult, StepLog, NavFrame, ScrollState
│
├── mcp_server/
│   └── server.py           # FastMCP skeleton (future)
│
├── PLAN.md                 # Full architecture and design decisions
├── setup.sh                # Auto-installer
└── .cursor/skills/
    └── auto-simctl-navigation/SKILL.md   # Navigation patterns & failure modes
```

See [PLAN.md](PLAN.md) for full architecture, coordinate systems, and design decisions.

## Third-party Models

This project downloads and uses the following models at runtime (not bundled):

| Model | License | Source |
|---|---|---|
| `qwen3.5-9b-mlx-4bit` | Apache 2.0 | [Qwen / Alibaba Cloud](https://huggingface.co/Qwen) |
| `neovateai/UI-UG-7B-2601` | Apache 2.0 | [neovateai/UI-UG-7B-2601](https://huggingface.co/neovateai/UI-UG-7B-2601) |

Models are downloaded separately (via `setup.sh`) and are not redistributed with this project.

## License

MIT — auto-simctl source code only.
The downloaded models are governed by their respective Apache 2.0 licenses.
