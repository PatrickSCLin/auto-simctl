"""
Orchestrator — ReAct control loop with navigation stack.

Flow per step:
  1. screenshot(device)
  2. qwen.decide(screenshot, task, history, nav_stack)
     → if "ground": ui_agent.grounding_targeted → qwen.decide phase-2
     → else: direct action
  3. Update nav_stack based on action type:
     - tap / swipe / launch_app  → push new frame (navigated deeper)
     - press_key(BACK)           → pop one frame
     - press_key(HOME)           → clear stack to root
  4. Execute, repeat.

Navigation stack gives Qwen full context:
  [0] Home Screen
  [1] Settings       (via tap(195,453))
  [2] Wi-Fi          (via tap(390,128))  ← current
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

from logger import get_logger
from agents.qwen_agent import QwenAgent
from agents.ui_agent import UIAgent
from mdb.bridge import DeviceBridge
from mdb.models import Action
from vision_screenshot_server import get_screenshot_url, set_current_screenshot

from .result import NavFrame, ScrollState, StepLog, TaskResult

log = get_logger("orchestrator")

MAX_GROUND_PER_STEP = 2
MAX_SAME_ACTION_REPEAT = 3


def _best_tap_from_elements(elements: list[dict], query: str) -> Action:
    """
    Auto-pick the best element to tap when Qwen refuses to decide.
    Handles both logical-point coords (from accessibility) and 0-1000 norm (from UI-UG).
    """
    keywords = {w.lower() for w in query.split() if len(w) > 2}

    def score(el: dict) -> int:
        label = el.get("label", "").lower()
        return sum(1 for kw in keywords if kw in label)

    best = max(elements, key=score)
    from_acc = best.get("description") == "from_accessibility"

    # Use the pre-computed center if available (from accessibility it's already logical pts)
    center = best.get("center", [])
    if center and len(center) == 2:
        cx, cy = center[0], center[1]
    else:
        bbox = best.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

    if from_acc:
        return Action(
            action_type="tap", x=cx, y=cy,
            reasoning=f"accessibility auto-picked {query!r}: {best.get('label')!r} at ({cx},{cy})",
        )
    return Action(
        action_type="tap", x=cx, y=cy,
        reasoning=f"norm1000 auto-picked {query!r}: {best.get('label')!r} at norm({cx},{cy})",
    )

def _best_acc_match(elements: list[dict], query: str) -> Action:
    """
    Auto-pick the best accessibility element when Qwen can't decide.
    Elements are already in logical point coordinates (cx, cy).
    Uses simple word-overlap scoring — good enough as a last resort.
    """
    q_words = {w.lower() for w in query.split() if len(w) > 1}

    def score(el: dict) -> int:
        label = el.get("label", "").lower()
        return sum(1 for w in q_words if w in label)

    best = max(elements, key=score)
    cx = best.get("cx", 0)
    cy = best.get("cy", 0)
    return Action(
        action_type="tap", x=cx, y=cy,
        reasoning=f"acc auto-picked for {query!r}: '{best.get('label')}' at ({cx},{cy})",
    )


# Task "打開 X app" / "open X app" → match foreground by name or bundle_id
_OPEN_APP_NORMALIZE = {"檔案": "files", "設定": "settings", "相片": "photos", "日曆": "calendar"}
_OPEN_APP_BUNDLE_SUBSTR = {
    "files": ["documentsapp", "files"],
    "settings": ["preferences", "settings"],
    "photos": ["mobileslideshow", "photos"],
    "calendar": ["mobilecal", "calendar"],
}


def _open_app_tap_if_visible(task: str, acc_elements: list) -> Optional[Action]:
    """
    If task is '打開 X app' and the X icon is visible as a Button in elements,
    return a direct tap action — no Qwen needed.
    """
    if not acc_elements:
        return None
    m = re.search(r"打開\s*([^\s]+)(?:\s*app)?", task) or re.search(r"open\s+(\w+)\s+app", task, re.I)
    if not m:
        return None
    x = m.group(1).strip().lower()
    x_norm = _OPEN_APP_NORMALIZE.get(x, x)
    if len(x_norm) < 2:
        return None
    keywords = [x_norm] + _OPEN_APP_BUNDLE_SUBSTR.get(x_norm, [])
    visible_buttons = [
        e for e in acc_elements
        if e.get("visible", True) and e.get("type") == "Button"
    ]
    for el in visible_buttons:
        label = el.get("label", "").lower()
        if any(kw in label for kw in keywords):
            cx, cy = el.get("cx", 0), el.get("cy", 0)
            log.info(
                f"App icon '{el.get('label')}' visible in elements → "
                f"direct tap({cx},{cy}), skip Qwen"
            )
            return Action(
                action_type="tap", x=cx, y=cy,
                reasoning=f"App icon '{el.get('label')}' visible in elements, tapped directly.",
            )
    return None


def _open_app_done_from_elements(task: str, acc_elements: list) -> Optional[Action]:
    """
    Pure-elements fast-path for "打開 X app".
    If elements have Tab Bar + (No Recents | in-app empty state) AND an "X | Application"
    label that matches the target app → done, no MDB needed.

    Example elements when inside Files:
      Files | Application
      No Recents | StaticText
      Tab Bar | Group
    """
    if not acc_elements:
        return None
    task = task.strip()
    m = re.search(r"打開\s*([^\s]+)(?:\s*app)?", task) or re.search(r"open\s+(\w+)\s+app", task, re.I)
    if not m:
        return None
    x = m.group(1).strip().lower()
    x_norm = _OPEN_APP_NORMALIZE.get(x, x)
    if len(x_norm) < 2:
        return None

    visible = [e for e in acc_elements if e.get("visible", True)]
    labels_lower = " ".join(e.get("label", "") for e in visible).lower()

    has_tab_bar = "tab bar" in labels_lower
    has_in_app_state = "no recents" in labels_lower or "recently opened" in labels_lower
    if not (has_tab_bar and has_in_app_state):
        return None

    # Look for "X | Application" where X matches the target app name
    keywords = [x_norm] + _OPEN_APP_BUNDLE_SUBSTR.get(x_norm, [])
    app_label = next(
        (e.get("label", "") for e in visible
         if "application" in e.get("type", "").lower() or
            (e.get("label", "") and any(kw in e.get("label", "").lower() for kw in keywords))),
        None
    )
    if app_label and any(kw in app_label.lower() for kw in keywords):
        return Action(
            action_type="done",
            result=f"Elements confirm '{app_label}' is open (Tab Bar + No Recents detected).",
            reasoning="Accessibility elements show in-app UI for target app; done without LLM or MDB.",
        )
    return None


def _open_app_done_if_foreground(
    task: str, foreground_app: dict, acc_elements: list
) -> Optional[Action]:
    """
    Fast-path for "打開 X app": elements-first, MDB as fallback confirmation.
    Priority: pure elements check (no stale MDB issues) → MDB + elements combined.
    """
    # Try pure-elements path first (doesn't depend on MDB at all)
    action = _open_app_done_from_elements(task, acc_elements)
    if action:
        return action

    # Fallback: MDB foreground + elements showing in-app
    if not foreground_app:
        return None
    labels_lower = " ".join(
        e.get("label", "") for e in acc_elements if e.get("visible", True)
    ).lower()
    has_tab_bar = "tab bar" in labels_lower
    has_in_app = "no recents" in labels_lower or "application" in labels_lower
    if not (has_tab_bar and has_in_app):
        return None

    task_s = task.strip()
    m = re.search(r"打開\s*([^\s]+)(?:\s*app)?", task_s) or re.search(r"open\s+(\w+)\s+app", task_s, re.I)
    if not m:
        return None
    x = m.group(1).strip().lower()
    x_norm = _OPEN_APP_NORMALIZE.get(x, x)
    if len(x_norm) < 2:
        return None
    name = (foreground_app.get("name") or "").lower()
    bid = (foreground_app.get("bundle_id") or "").lower()
    keywords = [x_norm] + _OPEN_APP_BUNDLE_SUBSTR.get(x_norm, [])
    if any(kw in name or kw in bid for kw in keywords):
        return Action(
            action_type="done",
            result=f"App '{foreground_app.get('name', '')}' in foreground and elements show in-app (MDB+AX).",
            reasoning="Foreground matches task and accessibility shows in-app UI; done without LLM.",
        )
    return None


def _extract_screen_label(action: Action) -> str:
    """Extract a short screen label from Qwen's reasoning or action."""
    if action.reasoning:
        # Take first sentence, truncate
        first = action.reasoning.split(".")[0].strip()
        return first[:60] if first else "unknown screen"
    return f"after {action}"



def _swipe_direction(action: Action) -> str:
    """
    Classify a swipe action:
      "scroll_down"  — finger moves up (y decreases) → content scrolls down
      "scroll_up"    — finger moves down → content scrolls up
      "scroll_right" — finger moves left → content scrolls right (next page)
      "scroll_left"  — finger moves right → content scrolls left (prev page)
      "navigate"     — large horizontal swipe (page/carousel navigation)
    """
    if action.action_type != "swipe":
        return ""
    dx = (action.x2 or 0) - (action.x or 0)
    dy = (action.y2 or 0) - (action.y or 0)
    if abs(dy) >= abs(dx):
        return "scroll_down" if dy < 0 else "scroll_up"
    # Horizontal swipe — treat as navigation (page swipe) if large, else horizontal scroll
    return "navigate" if abs(dx) > 150 else ("scroll_right" if dx < 0 else "scroll_left")


def _is_navigation_action(action: Action) -> bool:
    """True if this action navigates to a new screen (not a scroll)."""
    if action.action_type == "tap":
        return True
    if action.action_type == "launch_app":
        return True
    if action.action_type == "swipe":
        return _swipe_direction(action) == "navigate"
    return False


def _is_back_action(action: Action) -> bool:
    return action.action_type == "press_key" and action.key in ("BACK", "HOME")


class Orchestrator:
    def __init__(
        self,
        mdb: DeviceBridge,
        qwen: QwenAgent,
        ui_agent: UIAgent,
        max_steps: int = 20,
        step_delay_ms: int = 800,
        on_step: Optional[Callable[[StepLog], None]] = None,
    ) -> None:
        self.mdb = mdb
        self.qwen = qwen
        self.ui_agent = ui_agent
        self.max_steps = max_steps
        self.step_delay_ms = step_delay_ms
        self.on_step = on_step

    def _is_on_home_screen(self, device_udid: str) -> bool:
        """
        True if SpringBoard is the foreground app (home screen / app grid).
        SpringBoard bundle ID: com.apple.springboard
        """
        try:
            fg = self.mdb.get_foreground_app(device_udid)
            if fg:
                bid = (fg.get("bundle_id") or "").lower()
                return "springboard" in bid
        except Exception:
            pass
        return False

    def run(self, task: str, device_udid: str) -> TaskResult:
        logs: list[StepLog] = []
        history: list[dict] = []

        log.info(f"Starting task: {task!r} on device {device_udid[:12]}...")
        task_plan: Optional[str] = None

        # ── Pre-flight: ensure we start from the MAIN home screen page ──────────
        # Main home page: has app icon Buttons (Files, Contacts…) AND ≤ 12 elements.
        # Today View: also has dock but has 13+ elements (widgets). Distinguish by count.
        def _on_main_home() -> bool:
            try:
                els = self.mdb.list_elements(device_udid)
                visible = [e for e in els if e.get("visible", True)]
                labels = {e.get("label", "").lower() for e in visible}
                # Main home: small element count + at least one app icon Button
                has_app_icons = any(
                    e.get("type") == "Button" and len(e.get("label", "")) > 2
                    for e in visible
                )
                return has_app_icons and len(visible) <= 12
            except Exception:
                return False

        try:
            if not self._is_on_home_screen(device_udid):
                log.info("Not on home screen — pressing HOME to exit foreground app.")
                self.mdb.press_key(device_udid, "HOME")
                time.sleep(0.9)

            if not _on_main_home():
                # Today View is to the LEFT of page 0; swipe LEFT (350→50) to go to page 0.
                log.info("On a side page (Today View?) — swiping left to main home page.")
                self.mdb.swipe(device_udid, 350, 437, 50, 437, 300)
                time.sleep(0.6)

            if not _on_main_home():
                log.info("Still not on main home — pressing HOME once more.")
                self.mdb.press_key(device_udid, "HOME")
                time.sleep(0.6)

            if not _on_main_home():
                log.info("Trying swipe left once more to reach page 0.")
                self.mdb.swipe(device_udid, 350, 437, 50, 437, 300)
                time.sleep(0.5)
        except Exception as e:
            log.warning(f"Pre-flight HOME/swipe failed (non-fatal): {e}")

        # Navigation stack: always starts with root frame
        nav_stack: list[NavFrame] = [
            NavFrame(depth=0, screen_label="Root / Home Screen", action_taken=None, step=0)
        ]
        last_action_str: Optional[str] = None
        same_action_count = 0

        for step in range(1, self.max_steps + 1):
            step_log = StepLog(step=step, action=Action(action_type="error"))
            log.info(f"── Step {step}/{self.max_steps} ──────────────────")
            log.info(f"Nav depth: {len(nav_stack)-1}  path: {' → '.join(f.screen_label for f in nav_stack)}")

            # ── 1. Screenshot ─────────────────────────────────────────────────
            log.info("Taking screenshot...")
            t_shot = time.time()
            try:
                shot = self.mdb.screenshot(device_udid)
                step_log.screenshot_base64 = shot.base64
                log.info(f"Screenshot OK  {len(shot.png_bytes)//1024}KB  ({time.time()-t_shot:.2f}s)")
            except Exception as e:
                step_log.error = f"Screenshot failed: {e}"
                logs.append(step_log)
                return TaskResult(
                    success=False, task=task, steps_taken=step,
                    conclusion=f"Screenshot failed: {e}",
                    plan=task_plan, logs=logs, blocked_reason=str(e),
                    device_udid=device_udid,
                )

            # ── 1b. Dialog detection + auto-dismiss ───────────────────────────
            # Check before every step. If a system dialog is blocking:
            #   - Determine if the task NEEDS the permission
            #   - If not, auto-dismiss with the recommended button (no Qwen needed)
            #   - If yes, pass dialog info to Qwen to decide
            dialog_info: Optional[dict] = None
            try:
                dialog_info = self.mdb.detect_system_dialog(device_udid)
                if dialog_info:
                    btn_names = [b["label"] for b in dialog_info["buttons"]]
                    log.info(f"Dialog detected: {dialog_info['type']!r} — "
                             f"msg={dialog_info['message'][:60]!r}  "
                             f"buttons={btn_names}")

                    # Keywords that suggest the task REQUIRES the permission
                    _NEEDS_PERMISSION = {
                        "location": ["location", "map", "gps", "地圖", "位置", "地點"],
                        "camera":   ["camera", "photo", "video", "相機", "拍照", "相片"],
                        "mic":      ["microphone", "record", "voice", "麥克風", "錄音"],
                        "contact":  ["contact", "通訊錄"],
                        "notify":   ["notification", "通知"],
                    }
                    task_l = task.lower()
                    needs = any(
                        kw in task_l
                        for keywords in _NEEDS_PERMISSION.values()
                        for kw in keywords
                    )

                    if not needs:
                        # Auto-dismiss: find the dismiss button and tap it
                        dismiss = next(
                            (b for b in dialog_info["buttons"]
                             if b["label"] == dialog_info["dismiss_label"]),
                            dialog_info["buttons"][-1],
                        )
                        log.info(f"Auto-dismissing dialog (task doesn't need permission): "
                                 f"tap '{dismiss['label']}' at ({dismiss['cx']}, {dismiss['cy']})")
                        dismiss_action = Action(
                            action_type="tap",
                            x=dismiss["cx"], y=dismiss["cy"],
                            reasoning=f"Auto-dismiss dialog: '{dismiss['label']}'",
                        )
                        step_log.action = dismiss_action
                        logs.append(step_log)
                        history.append({"step": step, "action": str(dismiss_action)})
                        if self.on_step:
                            self.on_step(step_log)
                        try:
                            self.mdb.execute(device_udid, dismiss_action)
                            log.info("Dialog dismissed OK")
                        except Exception as e:
                            log.warning(f"Dialog dismiss failed: {e}")
                        time.sleep(0.5)
                        dialog_info = None  # cleared — proceed to Qwen
                        # Fall through to Qwen so it can proceed with the original task
            except Exception as e:
                log.debug(f"Dialog detection error (non-fatal): {e}")

            # ── 1c. Fast-path for pure navigation tasks ────────────────────────
            # If task is "go home" and no dialog is present, skip Qwen and just HOME.
            _GOTO_HOME_KEYWORDS = [
                "回到home", "回主畫面", "回首頁", "回home", "回到首頁",
                "go home", "home screen", "launcher", "主畫面", "首頁",
                "press home", "按home",
            ]
            if not dialog_info and any(kw in task.lower() for kw in _GOTO_HOME_KEYWORDS):
                if step == 1:
                    log.info("Fast-path: 'go home' task → pressing HOME directly")
                    action = Action(
                        action_type="press_key", key="HOME",
                        reasoning="Go-home task: press HOME directly",
                    )
                    step_log.action = action
                    logs.append(step_log)
                    history.append({"step": step, "action": str(action)})
                    if self.on_step:
                        self.on_step(step_log)
                    log.info(f"Executing: {action}")
                    try:
                        self.mdb.execute(device_udid, action)
                        log.info("Action executed OK")
                    except Exception as e:
                        log.error(f"Execute failed: {e}")
                    # HOME for go-home task = done
                    log.info("HOME pressed — task complete.")
                    return TaskResult(
                        success=True, task=task, steps_taken=step,
                        conclusion="Returned to home screen successfully.",
                        plan=task_plan, logs=logs, device_udid=device_udid,
                    )

            # ── 1d. Accessibility elements + scroll info snapshot ─────────────
            # Fetch ALL labeled elements (including off-screen) so Qwen can:
            #   1. Semantic-match task language to screen labels (no translation table)
            #   2. See off-screen elements and know scrolling is needed
            acc_elements: list[dict] = []
            scroll_info: dict = {}
            keyboard_open: bool = False
            try:
                acc_elements = self.mdb.list_elements(device_udid)
                visible_count = sum(1 for e in acc_elements if e.get("visible", True))
                offscreen_count = len(acc_elements) - visible_count
                # Keyboard is open if single-letter Button elements are present
                keyboard_open = any(
                    e.get("type") == "Button" and len(e.get("label", "").strip()) == 1
                    for e in acc_elements
                )
                if keyboard_open:
                    log.info(f"Accessibility: {visible_count} visible + {offscreen_count} off-screen "
                             f"(keyboard open)")
                else:
                    log.info(f"Accessibility: {visible_count} visible + {offscreen_count} off-screen elements")
                if acc_elements:
                    scroll_info = self.mdb.get_scroll_info(device_udid)
                    if scroll_info.get("has_content_below") or scroll_info.get("has_content_above"):
                        log.info(f"Scroll: content_below={scroll_info.get('has_content_below')} "
                                 f"content_above={scroll_info.get('has_content_above')} "
                                 f"total_h≈{scroll_info.get('content_height_pt')}pt")
            except Exception as e:
                log.debug(f"list_elements error (non-fatal): {e}")

            # ── 2. Qwen phase-1 (or MDB fast-path for "打開 X app") ─────────────
            set_current_screenshot(shot.png_bytes)
            foreground_app = self.mdb.get_foreground_app(device_udid)
            if foreground_app:
                log.debug(f"Foreground app: {foreground_app.get('bundle_id')} ({foreground_app.get('name')})")
            action = _open_app_done_if_foreground(task, foreground_app or {}, acc_elements)
            if action is not None:
                log.info(f"MDB foreground matches task → done (skip Qwen): {action.result}")
            elif (tap_action := _open_app_tap_if_visible(task, acc_elements)) is not None:
                action = tap_action
            else:
                log.info("Qwen analyzing screen...")
                t_qwen = time.time()
                try:
                    action = self.qwen.decide(
                        task=task,
                        screenshot_data_url=shot.data_url,
                        screenshot_url=get_screenshot_url(),
                        ui_elements=acc_elements,
                        history=history,
                        step=step,
                        max_steps=self.max_steps,
                        nav_stack=nav_stack,
                        dialog_info=dialog_info,
                        scroll_info=scroll_info,
                        keyboard_open=keyboard_open,
                        foreground_app=foreground_app,
                    )
                    log.info(f"Qwen phase-1 ({time.time()-t_qwen:.1f}s): {action}")
                except Exception as e:
                    logs.append(step_log)
                    log.error(f"Qwen phase-1 failed: {e}")
                    return TaskResult(
                        success=False, steps_taken=step,
                        conclusion=f"Qwen reasoning failed: {e}",
                        plan=task_plan, logs=logs, blocked_reason=str(e),
                        device_udid=device_udid, task=task,
                    )

            # ── 2b. Snap tap to nearest accessible element ────────────────────
            # Qwen sometimes hallucinates coordinates from the screenshot.
            # When we have accessibility elements (which already carry correct
            # cx/cy from the system), snap any tap that is out-of-bounds or
            # suspiciously far from every element to the closest match by label.
            if (action.action_type == "tap" and acc_elements and
                    (action.x is None or action.y is None or
                     action.x > 402 or action.y > 874)):
                visible_els = [e for e in acc_elements if e.get("visible", True)]
                if visible_els:
                    best = min(
                        visible_els,
                        key=lambda e: (
                            (e.get("cx", 201) - (action.x or 201)) ** 2 +
                            (e.get("cy", 437) - (action.y or 437)) ** 2
                        ),
                    )
                    old_coords = (action.x, action.y)
                    action = Action(
                        action_type="tap",
                        x=best["cx"], y=best["cy"],
                        reasoning=action.reasoning,
                    )
                    log.warning(
                        f"Tap coords {old_coords} out of bounds — "
                        f"snapped to nearest element '{best.get('label')}' "
                        f"at ({best['cx']},{best['cy']})"
                    )

            # ── 3. Ground loop ────────────────────────────────────────────────
            # Qwen emits ground(query) when it cannot determine exact coordinates.
            #
            # Resolution priority:
            #   A. acc_elements already fetched in step 1d
            #      → pass to Qwen phase-2 as context; let Qwen pick semantically
            #      → this covers language bridging (設定→Settings) without hardcoding
            #   B. acc_elements is empty (game / canvas / custom view)
            #      → fall back to UI-UG visual grounding + Qwen phase-2
            ground_count = 0
            while action.action_type == "ground" and ground_count < MAX_GROUND_PER_STEP:
                ground_count += 1
                query = action.ground_query or "visible UI elements"
                log.info(f"Ground {ground_count}/{MAX_GROUND_PER_STEP}: {query!r}")

                if acc_elements:
                    # ── Strategy A: Qwen picks from accessibility elements ─────
                    # Accessibility elements are already in logical point coordinates.
                    # We have the full list — let Qwen do semantic matching.
                    log.info(f"Accessibility has {len(acc_elements)} elements — "
                             "asking Qwen to select the best match")
                    set_current_screenshot(shot.png_bytes)
                    t2 = time.time()
                    try:
                        action = self.qwen.decide(
                            task=task,
                            screenshot_data_url=shot.data_url,
                            screenshot_url=get_screenshot_url(),
                            ui_elements=acc_elements,
                            history=history,
                            step=step,
                            max_steps=self.max_steps,
                            grounding_result=acc_elements,
                            nav_stack=nav_stack,
                            ground_query=query,
                            foreground_app=foreground_app,
                        )
                        log.info(f"Qwen phase-2/acc ({time.time()-t2:.1f}s): {action}")
                    except Exception as e:
                        log.warning(f"Qwen phase-2 exception: {e}")
                        action = Action(action_type="error", result=str(e))

                    if action.action_type in ("ground", "error"):
                        action = _best_tap_from_elements(acc_elements, query)
                        log.warning(f"Phase-2 acc fallback — auto-picked: {action}")

                else:
                    # ── Strategy B: UI-UG visual grounding (no accessibility) ──
                    log.info(f"No accessibility elements — using UI-UG for {query!r}")
                    set_current_screenshot(shot.png_bytes)
                    try:
                        elements = self.ui_agent.grounding_targeted(
                            shot.png_bytes, query, screenshot_url=get_screenshot_url()
                        )
                        step_log.ui_elements = elements
                        log.info(f"UI-UG: {len(elements)} element(s) for {query!r}")
                    except Exception as e:
                        elements = []
                        log.warning(f"UI-UG failed: {e}")
                    element_dicts = [el.to_dict() for el in elements]

                    # Qwen phase-2 picks from UI-UG results
                    log.info("Qwen phase-2/visual: deciding from UI-UG grounding...")
                    set_current_screenshot(shot.png_bytes)
                    t2 = time.time()
                    try:
                        action = self.qwen.decide(
                            task=task,
                            screenshot_data_url=shot.data_url,
                            screenshot_url=get_screenshot_url(),
                            ui_elements=element_dicts,
                            history=history,
                            step=step,
                            max_steps=self.max_steps,
                            grounding_result=element_dicts,
                            nav_stack=nav_stack,
                            ground_query=query,
                            foreground_app=foreground_app,
                        )
                        log.info(f"Qwen phase-2/visual ({time.time()-t2:.1f}s): {action}")
                    except Exception as e:
                        log.warning(f"Qwen phase-2 exception: {e}")
                        action = Action(action_type="error", result=str(e))

                    if action.action_type in ("ground", "error") and element_dicts:
                        action = _best_tap_from_elements(element_dicts, query)
                        log.warning(f"Phase-2 visual fallback — auto-picked: {action}")

            if action.action_type == "ground":
                # Ground loop exhausted — force backtrack to parent screen
                log.warning("Ground loop limit — forcing BACK to previous screen")
                action = Action(
                    action_type="press_key", key="BACK",
                    reasoning="Could not find target. Going back to try a different path.",
                )

            # ── 4. Dead-end detection: same action repeated too many times ────
            current_action_str = str(action)
            if current_action_str == last_action_str:
                same_action_count += 1
            else:
                same_action_count = 1
                last_action_str = current_action_str

            if same_action_count >= MAX_SAME_ACTION_REPEAT and not action.done:
                log.warning(
                    f"Same action repeated {same_action_count}x ({action}) — "
                    "forcing HOME to escape dead end"
                )
                action = Action(
                    action_type="press_key", key="HOME",
                    reasoning=f"Stuck: {action} repeated {same_action_count} times without progress.",
                )
                same_action_count = 0

            # ── 5. Log step ───────────────────────────────────────────────────
            step_log.action = action
            logs.append(step_log)
            history.append({
                "step": step,
                "action": str(action),
                "nav_depth": len(nav_stack) - 1,
            })

            if self.on_step:
                self.on_step(step_log)

            # ── 6. Terminal actions ────────────────────────────────────────────
            if action.action_type == "done":
                log.info(f"Task done: {action.result}")
                return TaskResult(
                    success=True,
                    task=task,
                    plan=task_plan,
                    steps_taken=step,
                    conclusion=action.result or "Task completed.",
                    logs=logs,
                    device_udid=device_udid,
                )
            if action.action_type == "error":
                log.warning(f"Task error: {action.result}")
                return TaskResult(
                    success=False, task=task, steps_taken=step,
                    conclusion=action.result or "Task failed.",
                    plan=task_plan, logs=logs, blocked_reason=action.result,
                    device_udid=device_udid,
                )

            # ── 7. Execute ────────────────────────────────────────────────────
            log.info(f"Executing: {action}")
            try:
                self.mdb.execute(device_udid, action)
                log.info("Action executed OK")
            except Exception as e:
                log.error(f"Execute failed: {e}")
                step_log.error = str(e)
                # Don't terminate — let Qwen see the failure and recover
                history[-1]["error"] = str(e)
                time.sleep(0.3)
                continue

            # ── 8. Update navigation stack ────────────────────────────────────
            if _is_navigation_action(action):
                # Pushed deeper — record this screen transition
                label = _extract_screen_label(action)
                nav_stack.append(NavFrame(
                    depth=len(nav_stack),
                    screen_label=label,
                    action_taken=action,
                    step=step,
                ))
                log.info(f"Nav push → depth {len(nav_stack)-1}: {label!r}")

            elif action.action_type == "swipe":
                # Scroll within the same screen — update scroll offset in current frame
                direction = _swipe_direction(action)
                current = nav_stack[-1]
                dx = (action.x2 or 0) - (action.x or 0)
                dy = (action.y2 or 0) - (action.y or 0)
                if direction in ("scroll_down", "scroll_up"):
                    current.scroll.scroll_y -= dy   # dy<0 → scrolled down → offset increases
                    current.scroll.at_top    = not scroll_info.get("has_content_above", False)
                    current.scroll.at_bottom = not scroll_info.get("has_content_below", True)
                    current.scroll.at_top    = current.scroll.scroll_y <= 0
                    if scroll_info.get("content_height_pt"):
                        current.scroll.content_height_hint = scroll_info["content_height_pt"]
                    log.info(f"Scroll {direction}: offset_y now ~{current.scroll.scroll_y}pt  "
                             f"{'(at bottom)' if current.scroll.at_bottom else ''}")
                elif direction in ("scroll_left", "scroll_right"):
                    current.scroll.scroll_x -= dx
                    current.scroll.at_left  = current.scroll.scroll_x <= 0
                    if scroll_info.get("content_width_pt"):
                        current.scroll.content_width_hint = scroll_info["content_width_pt"]
                    log.info(f"Scroll {direction}: offset_x now ~{current.scroll.scroll_x}pt")

            elif action.action_type == "press_key":
                if action.key == "BACK" and len(nav_stack) > 1:
                    popped = nav_stack.pop()
                    log.info(f"Nav pop ← back from {popped.screen_label!r}, now depth {len(nav_stack)-1}")
                    # Reset scroll on the screen we return to (it may have scrolled before)
                    nav_stack[-1].scroll = ScrollState()
                elif action.key == "HOME":
                    nav_stack = [nav_stack[0]]   # keep root only
                    log.info("Nav reset → HOME (depth 0)")

                    # For "go home" tasks, pressing HOME is the definitive success.
                    # iOS HOME always navigates to the home screen — no need for Qwen to verify.
                    _GOTO_HOME_KEYWORDS = [
                        "回到home", "回主畫面", "回首頁", "回home", "回到首頁",
                        "go home", "home screen", "launcher", "主畫面", "首頁",
                        "press home", "按home",
                    ]
                    if any(kw in task.lower() for kw in _GOTO_HOME_KEYWORDS):
                        log.info("HOME key confirms task completion for 'go home' task.")
                        return TaskResult(
                            success=True, task=task, steps_taken=step,
                            conclusion="Returned to home screen successfully.",
                            plan=task_plan, logs=logs, device_udid=device_udid,
                        )

            # ── 9. Post-action screen context injection ────────────────────
            # After tap/launch, quickly read the new screen's key elements
            # and inject into history so AI can detect "done" on next step.
            if action.action_type in ("tap", "launch_app"):
                try:
                    time.sleep(0.3)
                    post_elements = self.mdb.list_elements(device_udid)
                    # Extract headings/nav bar labels as a screen summary
                    screen_hints = []
                    for el in post_elements:
                        if el.get("type") in ("Heading", "NavigationBar") and el.get("label"):
                            screen_hints.append(f"{el['type']}: {el['label']}")
                    if screen_hints:
                        history[-1]["screen_after"] = ", ".join(screen_hints[:3])
                except Exception:
                    pass

            # ── 10. Wait before next step ──────────────────────────────────────
            if self.step_delay_ms > 0:
                time.sleep(self.step_delay_ms / 1000.0)

        return TaskResult(
            success=False, task=task, steps_taken=self.max_steps,
            conclusion=f"Reached max steps ({self.max_steps}) without completing the task.",
            plan=task_plan, logs=logs, blocked_reason="max_steps_reached",
            device_udid=device_udid,
        )
