"""
auto-simctl CLI — typer + rich entry point.

Commands:
  devices          List connected devices
  run <task>       Run an AI task on a device (requires servers running)
  screenshot       Take a screenshot and save it
  boot             Boot an iOS Simulator
  server start     Start both Qwen + UI-UG inference servers (background)
  server stop      Stop both servers
  server status    Show server status + PIDs
  server logs      Stream live logs from both servers (Ctrl-C to stop)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="auto-simctl",
    help="Intelligent mobile simulator control — AI-driven device testing.",
    add_completion=False,
)

# Global verbose state — set before any command runs
_verbose = False

@app.callback()
def _global_options(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed debug logs"),
):
    global _verbose
    _verbose = verbose
    from logger import setup
    setup(verbose=verbose)
server_app = typer.Typer(help="Manage the Qwen inference server.")
app.add_typer(server_app, name="server")

console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_orchestrator(max_steps: int, on_step=None):
    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent
    from mdb.bridge import DeviceBridge
    from orchestrator.loop import Orchestrator

    mdb = DeviceBridge()
    qwen = QwenAgent()
    ui = UIAgent()
    return Orchestrator(mdb=mdb, qwen=qwen, ui_agent=ui, max_steps=max_steps, on_step=on_step)


def _device_table(devices) -> Table:
    table = Table(title="Connected Devices", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", min_width=20)
    table.add_column("Type", width=10)
    table.add_column("State", width=12)
    table.add_column("OS", width=12)
    table.add_column("UDID", style="dim")
    for i, d in enumerate(devices, 1):
        state_color = "green" if d.state.value in ("booted", "online") else "yellow"
        table.add_row(
            str(i),
            d.name,
            f"[blue]{d.device_type.value}[/blue]",
            f"[{state_color}]{d.state.value}[/{state_color}]",
            d.os_version,
            d.udid,
        )
    return table


def _step_panel(log) -> Panel:
    from mdb.models import Action
    action: Action = log.action
    color = "green" if action.action_type == "done" else (
        "red" if action.action_type == "error" else "cyan"
    )
    ui_count = len(log.ui_elements) if log.ui_elements else 0
    content = (
        f"[bold]Step {log.step}[/bold]  "
        f"[{color}]{action}[/{color}]\n"
        f"[dim]UI elements detected: {ui_count}[/dim]"
        + (f"\n[red]{log.error}[/red]" if log.error else "")
    )
    return Panel(content, border_style=color)


# ── Commands ───────────────────────────────────────────────────────────────────

@app.command()
def devices():
    """List all connected Android devices and iOS Simulators."""
    from mdb.bridge import DeviceBridge
    with console.status("[cyan]Scanning for devices...[/cyan]"):
        bridge = DeviceBridge()
        devs = bridge.list_devices()

    if not devs:
        rprint("[yellow]No devices found.[/yellow]")
        rprint("  Android: connect via USB with debugging enabled, or start an emulator.")
        rprint("  iOS: open Xcode → Simulators → Boot a simulator.")
        raise typer.Exit(1)

    console.print(_device_table(devs))


@app.command()
def boot(
    device: str = typer.Option("auto", "--device", "-d", help="Device UDID or 'auto' (picks first iOS Simulator)"),
):
    """Boot an iOS Simulator and open the Simulator.app window."""
    from mdb.bridge import DeviceBridge
    from mdb.models import DeviceState, DeviceType

    bridge = DeviceBridge()

    if device == "auto":
        devs = bridge.list_devices()
        ios = [d for d in devs if d.device_type == DeviceType.IOS]
        if not ios:
            rprint("[red]No iOS Simulators found.[/red]")
            raise typer.Exit(1)
        # Prefer already-booted; otherwise pick first available
        booted = [d for d in ios if d.state == DeviceState.BOOTED]
        dev_info = booted[0] if booted else ios[0]
        device = dev_info.udid
    else:
        dev_info = bridge.get_device(device)

    rprint(f"[dim]Target: {dev_info}[/dim]")

    if dev_info.state == DeviceState.BOOTED:
        with console.status("[cyan]Opening Simulator.app window...[/cyan]"):
            bridge.boot_simulator(device)
        rprint(f"[green]Simulator.app opened for {dev_info.name}.[/green]")
    else:
        with console.status(f"[yellow]Booting {dev_info.name} (this may take ~30s)...[/yellow]"):
            try:
                bridge.boot_simulator(device, wait_secs=90)
            except Exception as e:
                rprint(f"[red]Boot failed: {e}[/red]")
                raise typer.Exit(1)
        rprint(f"[green]{dev_info.name} is booted and ready.[/green]")


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to perform, e.g. 'Open Settings and enable Dark Mode'"),
    device: str = typer.Option("auto", "--device", "-d", help="Device UDID or 'auto'"),
    max_steps: int = typer.Option(20, "--max-steps", "-n", help="Maximum steps before giving up"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save full result JSON to file"),
    no_server: bool = typer.Option(False, "--no-server", help="Skip auto-starting Qwen server"),
):
    """Run an AI task on a mobile device."""
    from agents.qwen_agent import QwenAgent
    from mdb.bridge import DeviceBridge

    console.print(Panel(
        f"[bold cyan]Task:[/bold cyan] {task}\n"
        f"[dim]Device: {device} | Max steps: {max_steps}[/dim]",
        title="auto-simctl",
        border_style="cyan",
    ))

    # Cancel any running task
    prev_pid = _read_pid(_TASK_PID_FILE)
    if prev_pid and _pid_alive(prev_pid):
        import signal as _sig
        rprint(f"[yellow]Cancelling previous task (PID {prev_pid})…[/yellow]")
        os.kill(prev_pid, _sig.SIGTERM)
        time.sleep(0.5)
    _write_pid(_TASK_PID_FILE, os.getpid())

    # Resolve device
    bridge = DeviceBridge()
    if device == "auto":
        dev = bridge.first_device()
        if dev is None:
            rprint("[red]No devices available.[/red] Connect a device or start a Simulator.")
            raise typer.Exit(1)
        device = dev.udid

    dev_info = bridge.get_device(device)
    rprint(f"[dim]Using device: {dev_info}[/dim]")

    # Auto-boot iOS Simulator if not ready
    from mdb.models import DeviceState, DeviceType
    if dev_info.device_type == DeviceType.IOS:
        if dev_info.state == DeviceState.SHUTDOWN:
            with console.status(f"[yellow]Booting {dev_info.name} (may take ~30s)...[/yellow]"):
                try:
                    bridge.boot_simulator(device, wait_secs=90)
                    rprint("[green]Simulator booted and ready.[/green]")
                except Exception as e:
                    rprint(f"[red]Failed to boot simulator: {e}[/red]")
                    raise typer.Exit(1)
        else:
            # Already booted — ensure window is visible
            bridge.boot_simulator(device)

    # Check both servers are ready
    from agents.ui_agent import UIAgent
    qwen = QwenAgent()
    ui_check = UIAgent()
    if not no_server:
        qwen_ok = qwen.server_running()
        uiug_ok = ui_check.server_running()
        if not qwen_ok or not uiug_ok:
            missing = []
            if not qwen_ok: missing.append("Qwen (port 8080)")
            if not uiug_ok: missing.append("UI-UG (port 8081)")
            rprint(f"[red]Servers not running:[/red] {', '.join(missing)}")
            rprint("Start both with: [cyan]python3 cli.py server start[/cyan]")
            raise typer.Exit(1)

    # Run with live step display
    step_logs = []

    def on_step(log):
        step_logs.append(log)
        console.print(_step_panel(log))

    orch = _make_orchestrator(max_steps=max_steps, on_step=on_step)

    start = time.time()
    result = orch.run(task=task, device_udid=device)
    elapsed = time.time() - start

    # Final result
    border = "green" if result.success else "red"
    icon = "✓" if result.success else "✗"
    console.print(Panel(
        f"[bold]{icon} {'Success' if result.success else 'Failed'}[/bold]\n\n"
        f"{result.conclusion}\n\n"
        f"[dim]Steps: {result.steps_taken}/{max_steps}  |  Time: {elapsed:.1f}s[/dim]"
        + (f"\n[yellow]Blocked: {result.blocked_reason}[/yellow]" if result.blocked_reason else ""),
        title="Result",
        border_style=border,
    ))

    if output:
        output.write_text(result.to_json(), encoding="utf-8")
        rprint(f"[dim]Full result saved to {output}[/dim]")

    raise typer.Exit(0 if result.success else 1)


@app.command()
def screenshot(
    device: str = typer.Option("auto", "--device", "-d", help="Device UDID or 'auto'"),
    output: Path = typer.Option(Path("screenshot.png"), "--output", "-o", help="Output PNG path"),
):
    """Take a screenshot from a device and save to file."""
    from mdb.bridge import DeviceBridge
    bridge = DeviceBridge()

    if device == "auto":
        dev = bridge.first_device()
        if dev is None:
            rprint("[red]No devices available.[/red]")
            raise typer.Exit(1)
        device = dev.udid

    from mdb.models import DeviceState, DeviceType
    dev_info = bridge.get_device(device)
    if dev_info.device_type == DeviceType.IOS and dev_info.state == DeviceState.SHUTDOWN:
        with console.status(f"[yellow]Booting {dev_info.name}...[/yellow]"):
            bridge.boot_simulator(device, wait_secs=45)
        rprint("[green]Simulator booted.[/green]")

    with console.status("[cyan]Taking screenshot...[/cyan]"):
        shot = bridge.screenshot(device)

    output.write_bytes(shot.png_bytes)
    rprint(f"[green]Screenshot saved to {output}[/green]  ({len(shot.png_bytes) // 1024} KB)")


# ── Server subcommands ─────────────────────────────────────────────────────────

_SERVER_DIR     = Path(__file__).parent
_QWEN_PID_FILE  = _SERVER_DIR / ".qwen_server.pid"
_UIUG_PID_FILE  = _SERVER_DIR / ".uiug_server.pid"
_TASK_PID_FILE  = _SERVER_DIR / ".current_task.pid"
_QWEN_LOG_FILE  = _SERVER_DIR / ".qwen_server.log"
_UIUG_LOG_FILE  = _SERVER_DIR / ".uiug_server.log"


def _write_pid(path: Path, pid: int) -> None:
    path.write_text(str(pid))


def _read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _stream_logs(follow: bool = True) -> None:
    """
    Stream .qwen_server.log and .uiug_server.log to the terminal.
    Qwen lines are cyan, UI-UG lines are magenta.
    Blocks until Ctrl-C when follow=True.
    """
    import select, threading

    styles = {
        _QWEN_LOG_FILE: ("cyan",    "Qwen "),
        _UIUG_LOG_FILE: ("magenta", "UI-UG"),
    }
    handles = {}
    for path, (color, label) in styles.items():
        if path.exists():
            f = open(path, "r", encoding="utf-8", errors="replace")
            f.seek(0, 2)  # seek to end — only new lines
            handles[f.fileno()] = (f, color, label)

    if not handles:
        rprint("[dim]No server log files yet. Start servers first.[/dim]")
        return

    stop = threading.Event()

    def _drain() -> None:
        while not stop.is_set():
            try:
                readable, _, _ = select.select(list(handles.keys()), [], [], 0.1)
            except ValueError:
                break
            for fd in readable:
                f, color, label = handles[fd]
                line = f.readline()
                if line:
                    console.print(f"[{color}][{label}][/{color}] {line.rstrip()}")
            if not follow:
                break

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    if follow:
        rprint("[dim]Streaming server logs — Ctrl-C to stop[/dim]\n")
        try:
            t.join()
        except KeyboardInterrupt:
            stop.set()
    else:
        t.join(timeout=2)

    for f, _, _ in handles.values():
        f.close()


def _start_proc(
    cmd: list[str],
    log_file: Path,
    pid_file: Path,
) -> "subprocess.Popen":
    """Launch a subprocess, tee stdout+stderr to log_file, record PID."""
    import subprocess
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=fh, stderr=fh,
        start_new_session=True,  # detach from terminal
    )
    _write_pid(pid_file, proc.pid)
    return proc


@server_app.command("start")
def server_start(
    qwen_model: str = typer.Option(
        str(Path.home() / ".cache/huggingface/hub/qwen3.5-9b-mlx-4bit"),
        "--qwen-model", help="Qwen model path",
    ),
    uiug_model: str = typer.Option(
        str(Path.home() / ".cache/huggingface/hub/ui-ug-7b-2601-4bit"),
        "--uiug-model", help="UI-UG model path",
    ),
    qwen_port: int = typer.Option(8080, "--qwen-port"),
    uiug_port: int = typer.Option(8081, "--uiug-port"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                 help="Stream server logs after starting"),
):
    """Start both Qwen (reasoning) and UI-UG (vision) servers in the background."""
    import subprocess, sys
    python = sys.executable
    python_bin = Path(sys.executable).parent

    # ── Qwen server ────────────────────────────────────────────────────────────
    from agents.qwen_agent import QwenAgent
    qwen = QwenAgent(model_path=qwen_model)
    if qwen.server_running():
        rprint(f"[green]Qwen already running[/green] at http://localhost:{qwen_port}/v1")
    else:
        rprint(f"[yellow]Starting Qwen server[/yellow] (port {qwen_port}) …")
        server_bin = python_bin / "mlx-openai-server"
        _start_proc(
            [str(server_bin), "launch",
             "--model-path", qwen_model,
             "--model-type", "multimodal",  # VLM: Qwen sees screenshot
             "--port", str(qwen_port),
             "--host", "127.0.0.1",
             "--max-tokens", "4096"],
            log_file=_QWEN_LOG_FILE,
            pid_file=_QWEN_PID_FILE,
        )
        with console.status("[cyan]Waiting for Qwen server…[/cyan]"):
            deadline = time.time() + 120
            while time.time() < deadline:
                if qwen.server_running():
                    break
                time.sleep(1)
        if qwen.server_running():
            rprint(f"[green]Qwen ready[/green] at http://localhost:{qwen_port}/v1")
        else:
            rprint("[red]Qwen server did not start in time.[/red]")
            rprint(f"  Check logs: [dim]python3 cli.py server logs[/dim]  or  [dim]cat {_QWEN_LOG_FILE}[/dim]")
            raise typer.Exit(1)

    # ── UI-UG server ───────────────────────────────────────────────────────────
    from agents.ui_agent import UIAgent
    ui = UIAgent(server_url=f"http://127.0.0.1:{uiug_port}")
    if ui.server_running():
        rprint(f"[green]UI-UG already running[/green] at http://localhost:{uiug_port}")
    else:
        rprint(f"[yellow]Starting UI-UG server[/yellow] (port {uiug_port}, loading ~7 GB model…)")
        _start_proc(
            [python, str(Path(__file__).parent / "ui_server.py"),
             "--port", str(uiug_port),
             "--model-path", uiug_model],
            log_file=_UIUG_LOG_FILE,
            pid_file=_UIUG_PID_FILE,
        )
        with console.status("[cyan]Waiting for UI-UG (loading model ~30 s)…[/cyan]"):
            deadline = time.time() + 120
            while time.time() < deadline:
                if ui.server_running():
                    break
                time.sleep(2)
        if ui.server_running():
            rprint(f"[green]UI-UG ready[/green] at http://localhost:{uiug_port}")
        else:
            rprint("[red]UI-UG server did not start in time.[/red]")
            rprint(f"  Check logs: [dim]python3 cli.py server logs[/dim]  or  [dim]cat {_UIUG_LOG_FILE}[/dim]")
            raise typer.Exit(1)

    rprint("\n[bold green]Both servers ready.[/bold green]")
    rprint("  Run tasks with:   [cyan]python3 cli.py run '<task>'[/cyan]")
    rprint("  Watch live logs:  [cyan]python3 cli.py server logs[/cyan]")

    if verbose:
        rprint()
        _stream_logs(follow=True)


@server_app.command("restart")
def server_restart(
    qwen_model: str = typer.Option(
        str(Path.home() / ".cache/huggingface/hub/qwen3.5-9b-mlx-4bit"),
        "--qwen-model", help="Qwen model path",
    ),
    uiug_model: str = typer.Option(
        str(Path.home() / ".cache/huggingface/hub/ui-ug-7b-2601-4bit"),
        "--uiug-model", help="UI-UG model path",
    ),
    qwen_port: int = typer.Option(8080, "--qwen-port"),
    uiug_port: int = typer.Option(8081, "--uiug-port"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                 help="Stream server logs after starting"),
):
    """Stop both servers, then start them again (same as stop + start)."""
    server_stop()
    time.sleep(2)  # let ports release
    server_start(
        qwen_model=qwen_model,
        uiug_model=uiug_model,
        qwen_port=qwen_port,
        uiug_port=uiug_port,
        verbose=verbose,
    )


@server_app.command("stop")
def server_stop():
    """Stop both Qwen and UI-UG servers."""
    import signal as _sig
    stopped = 0
    for label, pid_file in [("Qwen", _QWEN_PID_FILE), ("UI-UG", _UIUG_PID_FILE)]:
        pid = _read_pid(pid_file)
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, _sig.SIGTERM)
                rprint(f"[green]Stopped {label} server (PID {pid})[/green]")
                stopped += 1
            except Exception as e:
                rprint(f"[yellow]{label}: {e}[/yellow]")
        else:
            rprint(f"[dim]{label} server not running[/dim]")
        pid_file.unlink(missing_ok=True)
    if stopped == 0:
        rprint("[yellow]No servers were running.[/yellow]")


@server_app.command("logs")
def server_logs(
    tail: bool = typer.Option(True, "--tail/--no-tail", "-f/-F",
                              help="Keep streaming (tail -f). Use --no-tail for a one-shot dump."),
    lines: int = typer.Option(40, "--lines", "-n",
                              help="Number of past lines to show before streaming."),
):
    """Stream live logs from both Qwen and UI-UG servers.

    \b
    Examples:
      python3 cli.py server logs           # follow forever (Ctrl-C to stop)
      python3 cli.py server logs --no-tail # dump last 40 lines and exit
      python3 cli.py server logs -n 100    # dump last 100 lines and follow
    """
    import threading

    styles = {
        _QWEN_LOG_FILE: ("cyan",    "Qwen "),
        _UIUG_LOG_FILE: ("magenta", "UI-UG"),
    }

    if not any(p.exists() for p in styles):
        rprint("[yellow]No log files found. Start servers with:[/yellow] python3 cli.py server start")
        raise typer.Exit(1)

    # Print historical lines first
    for path, (color, label) in styles.items():
        if not path.exists():
            continue
        all_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in all_lines[-lines:]:
            console.print(f"[{color}][{label}][/{color}] {line}")

    if not tail:
        return

    rprint("\n[dim]Streaming server logs — Ctrl-C to stop[/dim]\n")
    stop = threading.Event()

    def _follow(path: Path, color: str, label: str) -> None:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while not stop.is_set():
                    line = f.readline()
                    if line:
                        console.print(f"[{color}][{label}][/{color}] {line.rstrip()}")
                    else:
                        stop.wait(0.1)
        except Exception:
            pass

    threads = []
    for path, (color, label) in styles.items():
        if path.exists():
            t = threading.Thread(target=_follow, args=(path, color, label), daemon=True)
            t.start()
            threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()


@server_app.command("status")
def server_status():
    """Show status of both Qwen and UI-UG servers."""
    from agents.qwen_agent import QwenAgent
    from agents.ui_agent import UIAgent

    qwen = QwenAgent()
    ui   = UIAgent()

    qwen_ok = qwen.server_running()
    uiug_ok = ui.server_running()

    table = Table(show_header=False, box=None)
    table.add_column(width=12)
    table.add_column()
    table.add_row(
        "Qwen",
        f"[green]running[/green] at http://localhost:8080/v1" if qwen_ok
        else "[red]stopped[/red]",
    )
    table.add_row(
        "UI-UG",
        f"[green]running[/green] at http://localhost:8081" if uiug_ok
        else "[red]stopped[/red]",
    )

    qwen_pid = _read_pid(_QWEN_PID_FILE)
    uiug_pid = _read_pid(_UIUG_PID_FILE)
    if qwen_pid or uiug_pid:
        table.add_row("", "")
        if qwen_pid:
            alive = "[green]alive[/green]" if _pid_alive(qwen_pid) else "[red]dead[/red]"
            table.add_row("[dim]Qwen PID[/dim]", f"[dim]{qwen_pid}[/dim] ({alive})")
        if uiug_pid:
            alive = "[green]alive[/green]" if _pid_alive(uiug_pid) else "[red]dead[/red]"
            table.add_row("[dim]UI-UG PID[/dim]", f"[dim]{uiug_pid}[/dim] ({alive})")

    console.print(table)

    if not qwen_ok or not uiug_ok:
        rprint("\nStart with: [cyan]python3 cli.py server start[/cyan]")
        rprint("View logs:  [cyan]python3 cli.py server logs[/cyan]")


if __name__ == "__main__":
    app()
