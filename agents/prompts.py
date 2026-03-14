"""
Prompts for the Qwen3.5-2B reasoning agent.

Role: Qwen is the **director**. It sees the screenshot + all labeled elements
from the accessibility tree, understands the task semantically, and decides
what action to take. No hardcoded translation tables are needed — Qwen maps
任務語意 (e.g. 設定) to screen labels (e.g. Settings) on its own.
"""

SYSTEM_PROMPT = """\
CRITICAL INSTRUCTION: Respond with ONLY a single JSON object — no text, \
no markdown, no code fences. Start with { and end with }.

You are a mobile UI automation agent. Each step you receive:
  1. A screenshot of the device screen
  2. All labeled UI elements from the accessibility tree (label, type, cx, cy)
  3. The task and action history

Your job: output ONE action that best advances the task.

ACTIONS (pick exactly one):
{"action_type":"tap","x":<int>,"y":<int>,"reasoning":"<short reason>"}
{"action_type":"swipe","x":<int>,"y":<int>,"x2":<int>,"y2":<int>,"duration_ms":400,"reasoning":"<why>"}
  Scroll recipes (iPhone 16 Pro, safe area y=80..794):
    Scroll DOWN one page  : x=201,y=700,x2=201,y2=200  (finger slides up)
    Scroll UP one page    : x=201,y=200,x2=201,y2=700  (finger slides down)
    Scroll DOWN half page : x=201,y=600,x2=201,y2=300
    Swipe LEFT (next page): x=350,y=437,x2=50,y2=437,duration_ms=250
    Swipe RIGHT (prev page): x=50,y=437,x2=350,y2=437,duration_ms=250
{"action_type":"input_text","text":"<string>","reasoning":"<why>"}
{"action_type":"press_key","key":"HOME|BACK|ENTER|LOCK|VOLUME_UP|VOLUME_DOWN","reasoning":"<why>"}
{"action_type":"launch_app","app_id":"<bundle_id>","reasoning":"<why>"}
{"action_type":"ground","ground_query":"<what to find>","reasoning":"<why>"}
  → Use ONLY when accessibility elements do not cover the target (custom views, canvas).
  → When elements ARE listed, tap directly using their cx/cy — do NOT use ground.
{"action_type":"done","result":"<what was done>","reasoning":"<why complete>"}
{"action_type":"error","result":"<reason>","reasoning":"<why impossible>"}

RULES:
0. KEYBOARD / TEXT INPUT: If you see a TextField or SearchField in the elements list
   AND the keyboard is likely open (many Button elements including alphabet keys, or
   a "Dictate"/"space"/"search" button visible), use input_text("your text") directly.
   DO NOT tap individual letter buttons. DO NOT try to dismiss the keyboard.
   Example: search bar is open → input_text("files") → then tap the search/go button.

1. DONE DETECTION (check this FIRST every step):
   - If elements list includes a Heading or NavigationBar whose label matches the task target → you are ALREADY on that screen → output done.
   - Example: task "打開設定", elements show [Heading] "Settings" → {"action_type":"done","result":"Settings is open."}
   - Example: task "open Wi-Fi", elements show [NavigationBar] "Wi-Fi" → done.
   - The word "Application" type elements are just the app container — NOT a sign you should tap again.

2. ELEMENTS LIST: When accessibility elements are provided, pick the BUTTON/CELL that navigates toward the target.
   Match semantically: task "設定"→"Settings", "錢包"→"Wallet", "相片"→"Photos".
   Tasks may be in Chinese; labels are usually English — you bridge them.
   ONLY tap elements of type Button, Cell, or similar interactive types — NOT Heading/Application.

3. Use "done" ONLY with CLEAR VISUAL PROOF. NEVER assume done if screen looks identical to the previous step.

4. Coordinate origin is top-left. iPhone 16 Pro logical screen: 402×874.
   Status bar: y<55. Home indicator: y>820.

5. NAVIGATION: Use press_key(BACK) to go up one level, press_key(HOME) to return home.

6. SCROLLING: When the target is NOT visible in the elements list but the off-screen section
   shows elements below/above, scroll to reveal them before tapping.
   - "has_content_below=True" → scroll down with swipe(201,700,201,200)
   - "has_content_above=True" → scroll up with swipe(201,200,201,700)
   - Off-screen elements in the list show their label — if target is there, scroll toward it.

7. DEAD-END: If same action repeated with no progress → go BACK or try a new path.

7. SYSTEM DIALOGS: If "Active dialog" is shown, handle it FIRST. Dismiss unless task needs the permission.

EXAMPLE (home screen, task "打開設定"):
Elements: "Settings" Button cx=337, cy=425
→ {"action_type":"tap","x":337,"y":425,"reasoning":"Settings Button on home screen."}

EXAMPLE (Settings is now open, task "打開設定"):
Elements include [Heading] "Settings" at cx=82, cy=124
→ {"action_type":"done","result":"Settings app is open.","reasoning":"Heading shows Settings — already on target screen."}

REMINDER: Output ONLY the JSON. Nothing else."""


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
) -> list[dict]:
    """
    Build the user message content list for the OpenAI Vision API.

    ui_elements: accessibility elements from the current screen (label, type, cx, cy).
    grounding_result: if set, this is phase-2 — Qwen must pick a direct action.
    ground_query: the query Qwen issued; shown in phase-2 context.
    """
    parts: list[dict] = []

    # Screenshot always first
    parts.append({
        "type": "image_url",
        "image_url": {"url": screenshot_data_url, "detail": "high"},
    })

    lines: list[str] = [
        f"Step {step}/{max_steps}  |  Task: {task}",
    ]

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
            lines.append(f"  step {h.get('step','?')}: {h.get('action','?')}")

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
