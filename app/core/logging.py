"""Logging setup: one readable format for the whole application.

Level from the LOG_LEVEL environment variable (default INFO), e.g.
    $env:LOG_LEVEL="DEBUG"; python -m uvicorn app.main:app
"""

import logging
import os
import sys

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if not any(getattr(h, "_ldi", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
        handler._ldi = True                     # marker: never add our handler twice
        root.addHandler(handler)
    root.setLevel(level)
    # libraries that are too talkative at INFO
    for noisy in ("httpx", "httpcore", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)