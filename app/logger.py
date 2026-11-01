"""One rotating log file per pipeline stage -> logs/<stage>.log (plus console)."""
import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
_FMT = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

def get_logger(stage: str) -> logging.Logger:
    log = logging.getLogger(f"attendance.{stage}")
    if log.handlers:
        return log
    log.setLevel(logging.DEBUG)
    fh = RotatingFileHandler(LOG_DIR / f"{stage}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(_FMT)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(f"[{stage}] %(message)s"))
    log.addHandler(fh); log.addHandler(ch)
    log.propagate = False
    return log
