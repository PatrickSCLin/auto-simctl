# auto-simctl

**Intelligent Mobile Simulator Control** — the missing piece for vibe coding on mobile. An AI agent that controls real/simulated Android and iOS devices: screenshot → UI understanding → reasoning → action → report.

## What it does

- **Unified device bridge (MDB)**: One API over `adb` (Android) and `idb` (iOS Simulator). All coordinates clamped to screen bounds before execution.
- **Accessibility-first UI understanding**: `idb ui describe-all` provides precise logical-point `(cx, cy)` for every element. Qwen picks *which* element; the element table supplies *where*.
- **Qwen-as-director**: Qwen3.5-9B reasons from the accessibility tree + screenshot, decides actions, and bridges language semantics (e.g. Chinese task → English UI labels) without hardcoded translation tables.
- **Fast-paths (no LLM)**: The orchestrator short-circuits Qwen for deterministic cases — gesture keywords ("swipe right", "back", …), open-keyboard `input_text`, app-icon visible, foreground matches target app.
- **`act` — stateful one-shot mode**: Does NOT reset to HOME. Executes exactly one atomic action and returns. tap / swipe / scroll / pan are always considered done after one execution. Designed for the MCP vibe-coding loop.
- **`run` — full autonomous mode**: Pre-flight HOME reset, then multi-step ReAct loop until the goal is complete or max steps reached.
- **`screen` — instant screen snapshot**: Returns foreground app, visible elements with tap coordinates, scroll state, and keyboard state. With `-s` saves a screenshot.
- **Keyboard-open fast-path**: When `act("input_text X")` is called and the keyboard is already visible, the text is typed immediately — no Qwen call needed.
- **Compound input**: `act("type https://… on address textfield")` taps the field, waits for the keyboard, then types — all in one call.
- **Pre-flight home reset** (`run` only): Before each task, presses HOME then corrects Today View / side launcher pages by swiping left to page 0.
- **UI-UG fallback**: UI-UG-7B-2601 via a background HTTP server handles custom-drawn views where accessibility labels are unavailable (games, canvas, WebView).
- **ReAct loop with navigation stack**: Screenshot → acc elements → fast-paths → Qwen → action → execute → update nav stack → repeat until done or max steps.
- **Navigation & scroll awareness**: Maintains a `NavFrame` stack (depth, screen label, scroll offset) so the agent always knows which page it's on and how far it has scrolled.
- **Dialog & keyboard handling**: Auto-detects and dismisses system permission dialogs; detects on-screen keyboard and switches to `input_text()` automatically.
- **MCP server**: `mcp_server/server.py` exposes `list_devices`, `get_screen_state`, `act`, and `run_task` as FastMCP tools — plug directly into Cursor or Claude Desktop.

## Quick start

```bash
# One-time setup: install adb, idb, Python deps, download models
./setup.sh

# Start both model servers (Qwen on :8080, UI-UG on :8081)
python3 cli.py server start

# ── Full autonomous task (HOME reset, multi-step) ─────────────────────────────
python3 cli.py run "Open Settings"
python3 cli.py run "find any folders" --verbose

# ── One-shot act (no HOME reset, continues from current screen) ───────────────
python3 cli.py act "swipe right"                                # gesture fast-path
python3 cli.py act "tap Watch app"                              # tap
python3 cli.py act "tap address bar"                            # tap a field (keyboard opens)
python3 cli.py act "input_text https://google.com"              # type (keyboard must be open)
python3 cli.py act "press enter"                                # submit — input_text does NOT auto-press Enter
python3 cli.py act "type https://google.com on address bar"     # tap + type in one call (still needs press enter after)
python3 cli.py act "back"                                       # back

# ── Screen snapshot (for the vibe-coding brain) ───────────────────────────────
python3 cli.py screen                  # rich text summary of current screen
python3 cli.py screen --json           # machine-readable JSON
python3 cli.py screen -s shot.png      # save screenshot file

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
run mode — pre-flight (once per task):
  1. Press HOME if not on SpringBoard
  2. Detect Today View (>12 elements) → swipe left to page 0

act mode — no pre-flight (continues from current screen)

For each step (max 20):
  1. Take screenshot
  2. List all accessibility elements (visible + off-screen)
  3. Detect keyboard open (single-letter Button elements present)
  4. Get scroll boundary info (content above/below/left/right)
  5. Detect & auto-dismiss system dialogs
  6. Fast-paths (no Qwen):
     a. Keyboard open + input task (input_text / type)
        → input_text(X) directly, skip Qwen
     b. Gesture keyword (swipe right, swipe left, back, scroll up, scroll down, …)
        → deterministic swipe/press, skip Qwen  [act only: also skips step loop]
     c. Elements show Tab Bar + No Recents + app label → done
     d. MDB foreground = target app + elements show in-app → done
     e. App icon Button visible in elements → direct tap(cx, cy)
  7. Qwen phase-1: analyze screenshot + elements → decide action
     └─ If action = "ground": accessibility elements passed to Qwen phase-2
        (UI-UG called only if no accessibility labels at all)
  8. Snap out-of-bounds tap to nearest accessible element
  9. Execute action via MDB (coords clamped to screen bounds)
  10. act mode one-shot rules:
      - tap / swipe / scroll / pan → return done immediately (no verify loop)
      - input task + step 1 was tap → sleep 0.8s for keyboard, type, return done
  11. Update navigation stack (NavFrame + ScrollState)
  12. Dead-end check: same action 3× → force HOME  [run mode only]
```

**Key design principles**:
- Accessibility elements carry `(cx, cy)` in logical points — Qwen decides *which* element, the element table provides *where*. Qwen never needs to estimate coordinates from the screenshot for standard iOS apps.
- `act` is strictly one-shot: tap / swipe / scroll / pan complete after one execution. No post-action verification — the MCP caller observes via `get_screen_state`.
- Keyboard detection drives the `input_text` fast-path: if a keyboard is visible when `input_text X` is issued, the text is typed immediately without any Qwen call.
- `input_text` does **not** press Enter/Return automatically. If the action requires submission (URL navigation, search, form submit), follow up with a separate `act("press enter")`.

## Project layout

```
auto-simctl/
├── cli.py                  # Entry point: run / act / screen / devices / server
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
│   ├── loop.py             # Pre-flight, fast-paths, ReAct loop, nav stack,
│   │                       #   act one-shot rules, keyboard/input fast-path
│   └── result.py           # TaskResult, StepLog, NavFrame, ScrollState
│
├── mcp_server/
│   └── server.py           # FastMCP server: list_devices, get_screen_state,
│                           #   act (one-shot), run_task (multi-step)
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
