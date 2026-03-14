"""
QwenAgent — reasoning engine using Qwen3.5-2B-4bit via mlx-openai-server.

Calls the local OpenAI-compatible server started by mlx-openai-server.
Server must be running at http://localhost:8080/v1 (see server_manager.py).
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from logger import get_logger
from mdb.models import Action
from agents.prompts import SYSTEM_PROMPT, build_user_message

log = get_logger("qwen_agent")

DEFAULT_MODEL_PATH = str(
    Path.home() / ".cache" / "huggingface" / "hub" / "qwen3.5-2b-mlx-4bit"
)
DEFAULT_SERVER_URL = "http://localhost:8080/v1"
DEFAULT_MODEL_NAME = "qwen3.5-2b"   # model name sent to the API (any string works)


class QwenAgent:
    """
    Reasoning engine: given a task + screenshot + UI elements + history,
    outputs the next Action to execute.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: str = DEFAULT_MODEL_PATH,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.server_url = server_url
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self.server_url, api_key="local")
            except ImportError as e:
                raise RuntimeError("openai package not installed. Run: pip install openai") from e
        return self._client

    # ── Server management ──────────────────────────────────────────────────────

    def server_running(self) -> bool:
        """Check if the mlx-openai-server is responding."""
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.server_url}/models", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def start_server(self, port: int = 8080, wait_secs: int = 30) -> subprocess.Popen:
        """Start mlx-openai-server in background and wait for it to be ready."""
        if self.server_running():
            return None

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Qwen model not found at {self.model_path}. Run ./setup.sh first."
            )

        import sys
        python_bin = Path(sys.executable).parent
        server_bin = python_bin / "mlx-openai-server"

        cmd = [
            str(server_bin), "launch",
            "--model-path", self.model_path,
            "--model-type", "multimodal",
            "--port", str(port),
            "--host", "127.0.0.1",
            "--max-tokens", "4096",
        ]
        log.info(f"Starting mlx-openai-server on port {port} with model {self.model_path}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        deadline = time.time() + wait_secs
        while time.time() < deadline:
            if self.server_running():
                log.info(f"mlx-openai-server ready at {self.server_url}")
                return proc
            time.sleep(1)

        raise TimeoutError(f"mlx-openai-server did not start within {wait_secs}s")

    # ── Core reasoning ─────────────────────────────────────────────────────────

    def decide(
        self,
        task: str,
        screenshot_data_url: str,
        ui_elements: list[dict],
        history: list[dict],
        step: int = 1,
        max_steps: int = 20,
        grounding_result: Optional[list[dict]] = None,
        nav_stack: Optional[list] = None,
        dialog_info: Optional[dict] = None,
        ground_query: Optional[str] = None,
        scroll_info: Optional[dict] = None,
        keyboard_open: bool = False,
    ) -> Action:
        """
        Phase 1 (grounding_result=None):
          Qwen sees screenshot + task → may return a 'ground' action
          (meaning: "ask UI-UG to find <ground_query>")

        Phase 2 (grounding_result=[...]):
          Qwen receives UI-UG's precise element locations → must return
          a direct action (tap/swipe/etc.)

        Returns an Action. May return action_type='ground' in phase 1.
        """
        if not self.server_running():
            raise RuntimeError(
                "mlx-openai-server is not running. "
                "Start with: python3 cli.py server start"
            )

        phase = "phase-2/direct" if grounding_result is not None else "phase-1/analyze"
        log.info(f"Qwen decide: step={step}/{max_steps}  {phase}  ui_elements={len(ui_elements)}")
        client = self._get_client()
        user_content = build_user_message(
            task=task,
            screenshot_data_url=screenshot_data_url,
            ui_elements=ui_elements,
            history=history,
            step=step,
            max_steps=max_steps,
            grounding_result=grounding_result,
            nav_stack=nav_stack,
            dialog_info=dialog_info,
            ground_query=ground_query,
            scroll_info=scroll_info,
            keyboard_open=keyboard_open,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        t0 = time.time()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            # Disable Qwen3 thinking mode so it outputs JSON directly
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.time() - t0

        raw = response.choices[0].message.content.strip()
        log.debug(f"Qwen raw ({elapsed:.1f}s): {raw!r:.400}")

        # Strip <think>...</think> blocks emitted by Qwen3 thinking mode
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        action = self._parse_action(raw)

        # If still prose after stripping: try prose → action extraction
        if action.action_type == "error" and raw and not raw.strip().startswith("{"):
            log.warning("Qwen output was prose — attempting prose extraction")
            action_from_prose = self._extract_from_prose(raw)
            if action_from_prose is not None:
                log.info(f"Prose extraction succeeded: {action_from_prose}")
                action = action_from_prose
            else:
                # Last resort: ask again with the shortest possible prompt
                log.warning("Prose extraction failed — retrying with minimal prompt")
                retry_messages = [
                    {"role": "system", "content": "Output ONLY a JSON object. No text."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": screenshot_data_url, "detail": "low"}},
                            {"type": "text", "text": (
                                f"Task: {task}\n"
                                + (f"UI elements: {json.dumps(ui_elements[:5])}\n" if ui_elements else "")
                                + "Output ONE JSON action: tap/swipe/input_text/press_key/launch_app/ground/done/error."
                            )},
                        ],
                    },
                ]
                t1 = time.time()
                try:
                    retry_resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=retry_messages,
                        max_tokens=256,
                        temperature=0.0,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    raw2 = re.sub(
                        r"<think>.*?</think>", "",
                        retry_resp.choices[0].message.content.strip(),
                        flags=re.DOTALL,
                    ).strip()
                    log.debug(f"Retry raw ({time.time()-t1:.1f}s): {raw2!r:.300}")
                    action = self._parse_action(raw2)
                except Exception as e:
                    log.error(f"Retry failed: {e}")

        log.info(f"Qwen action: {action}")
        return action

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_action(self, raw: str) -> Action:
        """
        Parse Qwen output into an Action.
        Tries multiple strategies before giving up:
          1. Entire output is JSON
          2. JSON embedded inside prose (find first { ... last })
          3. Fallback error action
        """
        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

        # Strategy 1: whole thing is JSON
        if cleaned.startswith("{"):
            try:
                d = json.loads(cleaned)
                return Action.from_dict(d)
            except json.JSONDecodeError:
                pass

        # Strategy 2: find outermost { ... } in the text
        start = cleaned.find("{")
        if start != -1:
            # Walk from end to find matching closing brace
            depth = 0
            end = -1
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end != -1:
                candidate = cleaned[start:end]
                try:
                    d = json.loads(candidate)
                    log.debug("Extracted JSON from mixed-text response")
                    return Action.from_dict(d)
                except json.JSONDecodeError:
                    pass

        # Strategy 3: repair truncated JSON
        # Qwen sometimes outputs {"action_type":"tap","x":195,"y":453,"reasoning":"very long text...
        # that gets cut off. Extract the known fields via regex before giving up.
        if start != -1:
            fragment = cleaned[start:]
            repaired = self._repair_truncated_json(fragment)
            if repaired:
                log.debug("Repaired truncated JSON")
                return repaired

        return Action(
            action_type="error",
            result=f"Could not parse action from model output: {raw[:200]}",
        )

    def _repair_truncated_json(self, fragment: str) -> Optional[Action]:
        """
        Try to extract a valid action from a truncated JSON string like:
          {"action_type":"tap","x":195,"y":453,"reasoning":"long text that got cut off
        """
        # Extract action_type first (always required)
        at_m = re.search(r'"action_type"\s*:\s*"([^"]+)"', fragment)
        if not at_m:
            return None
        action_type = at_m.group(1)

        def _get_str(key: str) -> Optional[str]:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', fragment)
            return m.group(1) if m else None

        def _get_int(key: str) -> Optional[int]:
            m = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+)', fragment)
            return int(m.group(1)) if m else None

        try:
            d: dict = {"action_type": action_type}
            for k in ("key", "text", "app_id", "ground_query", "result"):
                v = _get_str(k)
                if v is not None:
                    d[k] = v
            for k in ("x", "y", "x2", "y2", "duration_ms"):
                v = _get_int(k)
                if v is not None:
                    d[k] = v
            action = Action.from_dict(d)
            if action.action_type != "error":
                return action
        except Exception:
            pass
        return None

    def _extract_from_prose(self, text: str) -> Optional[Action]:
        """
        Last-resort: extract an action intent from natural language.
        Handles cases like:
          "I'll tap on the Settings icon at coordinates (195, 453)"
          "I need to tap x=195 y=453"
          "swipe from 100,300 to 100,800"
          "press the HOME button"
          "the task is complete / done"
        """
        t = text.lower()

        # Done / error keywords
        if any(k in t for k in ("task is complete", "task is done", "successfully opened",
                                 "settings is now open", "task accomplished")):
            return Action(action_type="done",
                          result="Task appears complete based on model analysis.",
                          reasoning=text[:120])

        # HOME / BACK key phrases
        if any(k in t for k in ("press home", "go to home", "home screen",
                                 "home button", "press_key home")):
            return Action(action_type="press_key", key="HOME", reasoning=text[:80])
        if any(k in t for k in ("press back", "go back", "navigate back", "press_key back")):
            return Action(action_type="press_key", key="BACK", reasoning=text[:80])

        # Tap with coordinates — various formats
        tap_patterns = [
            r'tap[^\d]*\((\d+)[,\s]+(\d+)\)',           # tap(195, 453)
            r'tap[^\d]*x[=:\s]*(\d+)[^\d]+y[=:\s]*(\d+)',  # tap x=195 y=453
            r'at[^\d]+\((\d+)[,\s]+(\d+)\)',             # at (195, 453)
            r'coordinates?[^\d]+(\d+)[,\s]+(\d+)',       # coordinates 195, 453
            r'position[^\d]+(\d+)[,\s]+(\d+)',           # position 195 453
            r'\((\d{2,3})[,\s]+(\d{3,4})\)',             # any (xx, xxx) pair
        ]
        for pat in tap_patterns:
            m = re.search(pat, t)
            if m:
                x, y = int(m.group(1)), int(m.group(2))
                # Basic sanity: within plausible screen bounds
                if 0 < x < 1200 and 0 < y < 2800:
                    return Action(action_type="tap", x=x, y=y, reasoning=text[:80])

        # Swipe — "swipe from x1,y1 to x2,y2"
        swipe_m = re.search(
            r'swipe[^\d]*(\d+)[,\s]+(\d+)[^\d]+(\d+)[,\s]+(\d+)', t
        )
        if swipe_m:
            x1, y1, x2, y2 = (int(g) for g in swipe_m.groups())
            return Action(action_type="swipe", x=x1, y=y1, x2=x2, y2=y2,
                          reasoning=text[:80])

        # Ground query — "I need to find / locate ..."
        ground_m = re.search(
            r'(?:need to (?:find|locate)|looking for|search for)\s+["\']?([^"\'.\n]{5,60})["\']?',
            t,
        )
        if ground_m:
            return Action(action_type="ground",
                          ground_query=ground_m.group(1).strip(),
                          reasoning=text[:80])

        return None
