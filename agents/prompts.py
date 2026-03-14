"""
Prompts for the Qwen reasoning agent (vision + text).

Design: Qwen MUST see the screenshot (vision) as the universal solution.
Accessibility elements / MDB identifier precise tap are optional context or fast-paths.
"""

SYSTEM_PROMPT = """\
You are a mobile UI automation agent. The SCREEN IMAGE is the source of truth — use it to understand the current state and complete the task.

Output ONLY one JSON action per step.

ACTIONS:
{"action_type":"tap","x":<int>,"y":<int>,"reasoning":"<why>"}
{"action_type":"swipe","x":<int>,"y":<int>,"x2":<int>,"y2":<int>,"duration_ms":400,"reasoning":"<why>"}
{"action_type":"input_text","text":"<string>","reasoning":"<why>"}
{"action_type":"press_key","key":"HOME|BACK|ENTER","reasoning":"<why>"}
{"action_type":"launch_app","app_id":"<bundle_id>","reasoning":"<why>"}
{"action_type":"ground","ground_query":"<what to find>","reasoning":"<why>"}
{"action_type":"done","result":"<what was done>","reasoning":"<why>"}
{"action_type":"error","result":"<reason>","reasoning":"<why>"}

RULES:
1. ELEMENTS = IN-APP (trust over image): If the elements table contains "Tab Bar" AND ("No Recents" or "X | Application"), you are INSIDE the app, NOT on the home screen. Home screen has many Buttons (Fitness, Watch, Contacts, Files as icons). So: 4 elements with Tab Bar + No Recents = in-app. For task 打開 X app when elements show in-app → output done. Do NOT say "home screen" when elements show Tab Bar + No Recents.
2. FOREGROUND + IN-APP → done: If "Current foreground app: ... (X)" and task is 打開 X app, and elements show in-app (Tab Bar, No Recents, X | Application), → output done. (If elements show many icon Buttons = home grid, do not output done.)
3. IMAGE: Home = grid of app icons. In-app = Tab Bar, "No Recents", in-app content. When elements already show in-app, trust elements.
4. DONE — generic: If the screen/elements clearly show the task result, output done.
5. Elements (tap): ALWAYS use the (cx,cy) shown in the elements table. NEVER estimate coordinates from the image — the table coordinates are exact. "X | Application" + Tab Bar/No Recents = inside X.
6. Bridge languages: 設定→Settings, 檔案→Files, 相片→Photos.
7. KEYBOARD OPEN: Use input_text() directly. Do NOT tap letter keys.
8. SCROLL (vertical): swipe(201,700,201,200) = scroll down; swipe(201,200,201,700) = scroll up. PAGE (horizontal): swipe(50,437,350,437) = swipe right (go to previous page); swipe(350,437,50,437) = swipe left (go to next page). All coordinates in 402×874pt space.
9. DIALOGS: Handle system dialogs first. Dismiss unless task needs that permission.
10. DEAD-END: Same action repeated → go BACK or try new path.
11. Screen: iPhone 16 Pro 402×874. Top-left origin. Status bar y<55.
12. If you cannot find the target in the image, use ground("query"); otherwise decide from the screenshot.

Output ONLY the JSON. No explanation."""


def build_user_message(
    task: str,
    screenshot_data_url: str,
    ui_elements: list[dict],
    history: list[dict],
    step: int,
    max_steps: int,
    grounding_result: "list[dict] | None" = None,
    nav_stack: "list | None" = None,
    dialog_info: "dict | None" = None,
    ground_query: "str | None" = None,
    scroll_info: "dict | None" = None,
    keyboard_open: bool = False,
    screenshot_url: "str | None" = None,
    foreground_app: "dict | None" = None,
) -> list[dict]:
    """
    Build the user message content for the Vision API: image (screenshot) + text context.

    Prefer screenshot_url (binary fetch) over screenshot_data_url (base64) to avoid
    large request bodies and extra encoding. Qwen sees the screenshot first (universal).
    foreground_app from MDB (idb list-apps) helps done detection: e.g. task 打開 files app
    + foreground_app bundle_id is com.apple.DocumentsApp → we're in Files → done.
    """
    parts: list[dict] = []

    # Vision-first: send screenshot so Qwen can see the screen.
    # Prefer URL (server fetches binary) over data URL (base64 in body).
    image_url = screenshot_url if screenshot_url else (
        screenshot_data_url if (screenshot_data_url and screenshot_data_url.startswith("data:")) else None
    )
    if image_url:
        parts.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

    lines: list[str] = [
        f"Step {step}/{max_steps}  |  Task: {task}",
    ]

    # ── Foreground app (from MDB: idb list-apps process_state=Running) ─────────
    if foreground_app:
        bid = foreground_app.get("bundle_id", "")
        name = foreground_app.get("name", "")
        lines.append(f"\nCurrent foreground app: {bid} ({name})")
        lines.append("  Use this to decide done: e.g. task 打開 files app + foreground is Files app → done.")

    # ── Active dialog (highest priority context) ───────────────────────────────
    if dialog_info:
        lines.append("\n⚠️  ACTIVE SYSTEM DIALOG — handle this FIRST:")
        lines.append(f"  Type: {dialog_info['type']}")
        lines.append(f"  Message: {dialog_info['message']}")
        lines.append("  Buttons:")
        for btn in dialog_info["buttons"]:
            lines.append(f"    • \"{btn['label']}\"  tap({btn['cx']}, {btn['cy']})")
        dl = dialog_info["dismiss_label"]
        dismiss_btn = next((b for b in dialog_info["buttons"] if b["label"] == dl), None)
        if dismiss_btn:
            lines.append(f"  Suggested dismiss: tap({dismiss_btn['cx']}, {dismiss_btn['cy']}) "
                         f"→ \"{dl}\"")
        lines.append("  → Does this task require that permission? If not, dismiss it.")

    # ── Keyboard status ────────────────────────────────────────────────────────
    if keyboard_open:
        lines.append(
            "\n⌨️  KEYBOARD IS OPEN — a text field is focused and ready for input.\n"
            "   Use input_text(\"your text\") to type. "
            "Do NOT tap individual letter keys."
        )

    # ── Navigation breadcrumbs + scroll position ──────────────────────────────
    if nav_stack:
        lines.append(f"\nNavigation (depth {len(nav_stack) - 1}):")
        for frame in nav_stack:
            prefix = "  →" if frame.depth < len(nav_stack) - 1 else "  ★"
            action_str = f"  (via {frame.action_taken})" if frame.action_taken else ""
            scroll_str = ""
            if frame.scroll.scroll_y != 0 or frame.scroll.scroll_x != 0:
                scroll_str = f"  [scroll: {frame.scroll.summary()}]"
            lines.append(f"{prefix} [{frame.depth}] {frame.screen_label}{action_str}{scroll_str}")
    else:
        lines.append("\nNavigation: home screen (depth 0)")

    # ── Scroll boundaries ──────────────────────────────────────────────────────
    if scroll_info:
        si = scroll_info
        scroll_parts = []
        if si.get("has_content_above"):
            scroll_parts.append("content ABOVE viewport (scroll up to see)")
        if si.get("has_content_below"):
            scroll_parts.append("content BELOW viewport (scroll down to see)")
        if si.get("has_content_left"):
            scroll_parts.append("content to the LEFT")
        if si.get("has_content_right"):
            scroll_parts.append("content to the RIGHT")
        if scroll_parts:
            lines.append("Scroll boundaries: " + "  |  ".join(scroll_parts))
            if si.get("content_height_pt"):
                lines.append(f"  Total content height ≈ {si['content_height_pt']}pt "
                             f"(screen = 874pt)")

    # ── Recent action history ──────────────────────────────────────────────────
    if history:
        lines.append("\nLast actions:")
        for h in history[-4:]:
            line = f"  step {h.get('step','?')}: {h.get('action','?')}"
            if h.get("screen_after"):
                line += f" → screen: [{h['screen_after']}]"
            if h.get("error"):
                line += f" ⚠ ERROR: {h['error']}"
            lines.append(line)

    # ── Phase-2: picking from provided elements ────────────────────────────────
    if grounding_result is not None:
        is_phase2_acc = (
            grounding_result and
            isinstance(grounding_result[0], dict) and
            "cx" in grounding_result[0] and
            "bbox" not in grounding_result[0]
        )
        if ground_query:
            lines.append(f"\nGROUND QUERY: \"{ground_query}\"")

        if is_phase2_acc:
            # Accessibility elements — already in logical points
            lines.append(f"\nAccessibility elements on screen ({len(grounding_result)} total):")
            lines.append("  label                        | type            | tap(x,y)")
            lines.append("  " + "-" * 55)
            for el in grounding_result[:30]:
                label = el.get("label", "")[:30]
                etype = el.get("type", "")[:14]
                cx    = el.get("cx", 0)
                cy    = el.get("cy", 0)
                lines.append(f"  {label:<30} | {etype:<14} | tap({cx},{cy})")
            lines.append(
                "\nSemantically match the ground_query to the label above "
                "(task may be in Chinese, labels in English — you bridge them). "
                "Output a DIRECT tap/press_key/done action. Do NOT output ground."
            )
        else:
            # Visual grounding (UI-UG) result
            lines.append(f"\nVisual grounding result ({len(grounding_result)} element(s)):")
            for el in grounding_result[:10]:
                cx_list = el.get("center", [0, 0])
                cx, cy  = (cx_list[0], cx_list[1]) if len(cx_list) >= 2 else (0, 0)
                lines.append(
                    f"  • [{el.get('type','')}] \"{el.get('label','')}\"  "
                    f"center=({cx},{cy})  bbox={el.get('bbox',[])}"
                )
            if grounding_result:
                lines.append(
                    "\nOutput a DIRECT action using these coordinates. "
                    "Do NOT output ground."
                )
            else:
                lines.append(
                    "\nNothing found visually. Try BACK, a different ground query, or error."
                )

    # ── Phase-1: accessibility elements as context ─────────────────────────────
    elif ui_elements:
        # Filter out keyboard keys (single-letter buttons) — they clutter context
        _is_key = lambda e: (e.get("type") == "Button" and
                             len(e.get("label", "").strip()) == 1)
        visible = [el for el in ui_elements
                   if el.get("visible", True) and not _is_key(el)]
        offscreen = [el for el in ui_elements
                     if not el.get("visible", True) and not _is_key(el)]

        lines.append(f"\nVisible elements ({len(visible)}):")
        lines.append("  label                        | type            | tap(x,y)")
        lines.append("  " + "-" * 55)
        for el in visible[:25]:
            label = el.get("label", "")[:30]
            etype = el.get("type", "")[:14]
            cx    = el.get("cx", 0)
            cy    = el.get("cy", 0)
            lines.append(f"  {label:<30} | {etype:<14} | tap({cx},{cy})")

        if offscreen:
            lines.append(f"\nOff-screen elements ({len(offscreen)}, need scrolling to reach):")
            for el in offscreen[:15]:
                label = el.get("label", "")[:30]
                etype = el.get("type", "")[:14]
                cy    = el.get("cy", 0)
                direction = "↓ below" if cy > 874 else "↑ above"
                lines.append(f"  {label:<30} | {etype:<14} | {direction} viewport")

        # Hint when elements clearly indicate in-app (Tab Bar + No Recents / Application)
        labels_lower = " ".join(el.get("label", "") for el in visible).lower()
        has_tab_bar = "tab bar" in labels_lower
        has_in_app = "no recents" in labels_lower or "application" in labels_lower
        task_lower = task.lower()
        open_app_task = "打開" in task or "open" in task_lower
        if has_tab_bar and has_in_app and open_app_task:
            lines.append(
                "\n→ IN-APP: Elements show Tab Bar and in-app UI (No Recents / Application). "
                "You are INSIDE the app, not on home screen. If task is to open this app → output done."
            )

        lines.append(
            "\nMatch task semantically to labels (Chinese→English). "
            "Tap visible elements directly. "
            "For off-screen elements: scroll toward them first, then tap."
        )
    else:
        lines.append(
            "\nNo accessibility elements available. "
            "Use ground() to locate elements visually, or swipe to reveal content."
        )

    lines.append("\nOutput ONE JSON action object.")
    parts.append({"type": "text", "text": "\n".join(lines)})
    return parts
