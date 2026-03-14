"""
Shared dataclasses for MDB — Mobile Device Bridge.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from .screen import ScreenSpec


class DeviceType(str, Enum):
    ANDROID = "android"
    IOS = "ios"


class DeviceState(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BOOTED = "booted"      # iOS Simulator booted
    SHUTDOWN = "shutdown"  # iOS Simulator shutdown


@dataclass
class DeviceInfo:
    udid: str
    name: str
    device_type: DeviceType
    state: DeviceState
    os_version: str = ""
    model: str = ""              # e.g. "iPhone17,1"
    screen: "Optional[ScreenSpec]" = field(default=None, repr=False)

    def __str__(self) -> str:
        screen_str = ""
        if self.screen:
            screen_str = f" [{self.screen.pt_w}×{self.screen.pt_h}pt @{self.screen.scale}x]"
        return (f"{self.name} [{self.device_type.value}]"
                f"{screen_str} ({self.udid[:12]}...) — {self.state.value}")


@dataclass
class UIElement:
    label: str
    bbox: list[int]          # [x1, y1, x2, y2]
    element_type: str = ""   # button, text, image, icon, input, etc.
    description: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x = (self.bbox[0] + self.bbox[2]) // 2
        y = (self.bbox[1] + self.bbox[3]) // 2
        return (x, y)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "type": self.element_type,
            "bbox": self.bbox,
            "center": list(self.center),
            "description": self.description,
        }


ActionType = Literal[
    "tap", "swipe", "input_text", "press_key",
    "launch_app", "screenshot",
    "ground",   # Qwen asks UI-UG for precise coordinates before acting
    "done", "error",
]

KeyName = Literal["HOME", "BACK", "ENTER", "LOCK", "VOLUME_UP", "VOLUME_DOWN"]


@dataclass
class Action:
    action_type: ActionType
    # tap / swipe
    x: Optional[int] = None
    y: Optional[int] = None
    x2: Optional[int] = None
    y2: Optional[int] = None
    duration_ms: int = 300
    # input_text
    text: Optional[str] = None
    # press_key
    key: Optional[KeyName] = None
    # launch_app
    app_id: Optional[str] = None
    # ground — Qwen asks UI-UG for a specific element's coordinates
    ground_query: Optional[str] = None
    # done / error
    result: Optional[str] = None
    reasoning: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.action_type in ("done", "error")

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            action_type=d.get("action_type", "error"),
            x=d.get("x"),
            y=d.get("y"),
            x2=d.get("x2"),
            y2=d.get("y2"),
            duration_ms=d.get("duration_ms", 300),
            text=d.get("text"),
            key=d.get("key"),
            app_id=d.get("app_id"),
            ground_query=d.get("ground_query"),
            result=d.get("result"),
            reasoning=d.get("reasoning"),
        )

    def to_dict(self) -> dict:
        d: dict = {"action_type": self.action_type}
        for attr in ("x", "y", "x2", "y2", "text", "key", "app_id", "ground_query", "result", "reasoning"):
            v = getattr(self, attr)
            if v is not None:
                d[attr] = v
        if self.action_type == "swipe":
            d["duration_ms"] = self.duration_ms
        return d

    def __str__(self) -> str:
        if self.action_type == "tap":
            return f"tap({self.x}, {self.y})"
        if self.action_type == "swipe":
            return f"swipe({self.x},{self.y} → {self.x2},{self.y2}, {self.duration_ms}ms)"
        if self.action_type == "input_text":
            return f'input_text("{self.text}")'
        if self.action_type == "press_key":
            return f"press_key({self.key})"
        if self.action_type == "launch_app":
            return f"launch_app({self.app_id})"
        if self.action_type == "ground":
            return f"ground(query={self.ground_query!r})"
        if self.action_type == "done":
            return f"DONE: {self.result}"
        if self.action_type == "error":
            return f"ERROR: {self.result}"
        return self.action_type


@dataclass
class Screenshot:
    png_bytes: bytes
    device_udid: str
    width: int = 0
    height: int = 0

    @property
    def base64(self) -> str:
        return base64.b64encode(self.png_bytes).decode()

    @property
    def data_url(self) -> str:
        return f"data:image/png;base64,{self.base64}"
