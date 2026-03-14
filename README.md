# auto-simctl

**Intelligent Mobile Simulator Control** — the missing piece for vibe coding on mobile. An AI agent that controls real/simulated Android and iOS devices: screenshot → UI understanding → reasoning → action → report.

## What it does

- **Unified device bridge (MDB)**: One API over `adb` (Android) and `idb` (iOS Simulator).
- **Accessibility-first UI understanding**: `idb ui describe-all` provides precise logical-point coordinates for all on-screen elements, including off-screen (scrollable) content.
- **Qwen-as-director**: Qwen3.5-2B reasons from the accessibility tree + screenshot, decides actions, and bridges language semantics (e.g. Chinese task → English UI labels) without hardcoded translation tables.
- **UI-UG fallback**: UI-UG-7B-2601 via a background HTTP server handles custom-drawn views where accessibility labels are unavailable.
- **ReAct loop with navigation stack**: Screenshot → acc elements → Qwen → action → execute → update nav stack → repeat until done or max steps.
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
  - `qwen3.5-2b-mlx-4bit` (reasoning)
  - `neovateai/UI-UG-7B-2601` (UI grounding fallback)

## How the agent thinks (ReAct loop)

```
For each step (max 20):
  1. Take screenshot
  2. List all accessibility elements (visible + off-screen)
  3. Detect keyboard open (single-letter buttons present)
  4. Get scroll boundary info (content above/below/left/right)
  5. Detect & auto-dismiss system dialogs
  6. Qwen phase-1: analyze screenshot + elements → decide action
     └─ If action = "ground": accessibility elements passed to Qwen phase-2
        (UI-UG called only if no accessibility labels at all)
  7. Execute action via MDB
  8. Update navigation stack (NavFrame + ScrollState)
  9. Dead-end check: same action 3× → force HOME
```

## Project layout

```
auto-simctl/
├── cli.py                  # Entry point: run / devices / server start|stop|status
├── ui_server.py            # UI-UG-7B HTTP server (port 8081)
├── logger.py               # Structured logging helpers
│
├── mdb/                    # Mobile Device Bridge
│   ├── bridge.py           # DeviceBridge unified API
│   ├── screen.py           # ScreenSpec: pixel ↔ logical-pt ↔ norm1000
│   ├── models.py           # DeviceInfo, Action, Screenshot dataclasses
│   └── backends/
│       ├── idb_backend.py  # iOS: screenshot, tap, swipe, input_text,
│       │                   #   list_elements, get_scroll_info, detect_system_dialog
│       └── adb_backend.py  # Android: same interface via adb
│
├── agents/
│   ├── qwen_agent.py       # Qwen3.5-2B via mlx-openai-server (phase-1 + phase-2)
│   ├── ui_agent.py         # UI-UG-7B-2601 client (HTTP → port 8081)
│   └── prompts.py          # SYSTEM_PROMPT + build_user_message (rules + context injection)
│
├── orchestrator/
│   ├── loop.py             # ReAct loop, navigation stack, dead-end detection
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

## License

MIT.
