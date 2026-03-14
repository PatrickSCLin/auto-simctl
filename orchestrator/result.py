"""
Result dataclasses for the orchestrator ReAct loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from mdb.models import Action, UIElement


@dataclass
class ScrollState:
    """
    Tracks accumulated scroll offset and boundary flags for a screen.
    Units: logical points (same coordinate space as idb tap/swipe).
    scroll_y > 0  means scrolled DOWN (content has moved up); < 0 means scrolled UP.
    scroll_x > 0  means scrolled RIGHT (paged right); < 0 means scrolled LEFT.
    """
    scroll_y: int = 0          # accumulated vertical scroll in logical points
    scroll_x: int = 0          # accumulated horizontal scroll
    at_top:    bool = True
    at_bottom: bool = False
    at_left:   bool = True
    at_right:  bool = False
    content_height_hint: int = 0   # estimated total content height in logical pts (0=unknown)
    content_width_hint:  int = 0

    def summary(self) -> str:
        parts = []
        if self.scroll_y == 0 and self.at_top:
            parts.append("top of content")
        elif self.at_bottom:
            parts.append(f"bottom of content (scrolled ~{self.scroll_y}pt down)")
        else:
            parts.append(f"scrolled ~{self.scroll_y}pt from top")
        if self.scroll_x != 0 or not self.at_left:
            if self.at_right:
                parts.append("rightmost page")
            else:
                parts.append(f"~{self.scroll_x}pt from left edge")
        return "  |  ".join(parts)


@dataclass
class NavFrame:
    """
    One entry in the navigation stack.
    Pushed whenever the agent navigates INTO a new screen.
    Popped when BACK is pressed; cleared when HOME is pressed.
    """
    depth: int
    screen_label: str        # Qwen's description of this screen (from reasoning)
    action_taken: Action     # the action that navigated into this screen (None for root)
    step: int
    scroll: ScrollState = field(default_factory=ScrollState)


@dataclass
class StepLog:
    step: int
    action: Action
    ui_elements: list[UIElement] = field(default_factory=list)
    screenshot_base64: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "action": str(action := self.action),
            "action_detail": action.to_dict(),
            "ui_elements_count": len(self.ui_elements),
            "screenshot": self.screenshot_base64,
            "error": self.error,
        }

    def to_history_entry(self) -> dict:
        return {"step": self.step, "action": str(self.action)}


@dataclass
class TaskResult:
    success: bool
    steps_taken: int
    conclusion: str
    logs: list[StepLog] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    device_udid: str = ""
    task: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "steps_taken": self.steps_taken,
            "conclusion": self.conclusion,
            "blocked_reason": self.blocked_reason,
            "evidence": [log.to_dict() for log in self.logs],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
