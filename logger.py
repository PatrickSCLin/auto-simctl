"""
Centralised logging for auto-simctl.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.debug("detailed message")
    log.info("normal message")

Verbose mode is enabled by calling setup(verbose=True) once from cli.py.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager

try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False

_configured = False


def setup(verbose: bool = False) -> None:
    global _configured
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler]
    if _RICH:
        handlers = [RichHandler(
            level=level,
            show_time=True,
            show_path=verbose,
            markup=True,
            rich_tracebacks=True,
        )]
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        handlers = [handler]

    logging.basicConfig(level=level, handlers=handlers, force=True)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup(verbose=False)
    return logging.getLogger(name)


@contextmanager
def timer(log: logging.Logger, label: str, level: int = logging.DEBUG):
    """Context manager that logs elapsed time for a block."""
    t0 = time.time()
    log.log(level, f"[dim]→ {label}...[/dim]")
    try:
        yield
    finally:
        elapsed = time.time() - t0
        log.log(level, f"[dim]✓ {label} done ({elapsed:.2f}s)[/dim]")
