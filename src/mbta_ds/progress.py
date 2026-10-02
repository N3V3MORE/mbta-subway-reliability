"""Reusable progress reporting for long-running stages.

Built on the logging framework rather than a third-party progress bar, for three
reasons: output lands in the same stream as everything else (so a redirect to a
file captures it), it respects the ``-v`` flag, and the project stays free of an
extra dependency.

Every stage that loops over more than a handful of items reports ``[n/total]``
with elapsed time and an ETA, so a run that takes minutes shows how far along it
is instead of appearing to hang.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger("mbta_ds.progress")


def format_duration(seconds: float) -> str:
    """Render a duration compactly: ``9s``, ``2m 14s``, ``1h 03m``.

    Returns ``"?"`` for NaN or infinite input, which is what an ETA works out to
    before the first step has completed.
    """
    if seconds is None or math.isnan(seconds) or math.isinf(seconds):
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


@dataclass
class Progress:
    """Counts through a known number of steps and reports pace and ETA.

    Usage::

        with Progress(len(models), "regression models") as progress:
            for name, model in models.items():
                ...fit and score...
                progress.step(name, extra=f"MAE {mae:.1f}s")

    ``extra`` is appended to the line so the running metric for each step is
    visible as the stage proceeds, not only in the final summary.
    """

    total: int
    label: str
    #: Emit a line every N steps. The final step is always reported.
    log_every: int = 1
    #: Also report the start, so a slow first step does not look like a hang.
    announce_start: bool = True

    started: float = field(default=0.0, init=False)
    done: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 1)
        self.log_every = max(int(self.log_every), 1)

    def __enter__(self) -> Progress:
        self.started = time.monotonic()
        self.done = 0
        if self.announce_start:
            log.info("%s: 0/%d", self.label, self.total)
        return self

    def step(self, name: str = "", *, extra: str = "") -> int:
        """Record one completed step and log it if it is due."""
        self.done += 1
        if self.done < self.total and self.done % self.log_every:
            return self.done

        elapsed = time.monotonic() - self.started
        per_step = elapsed / self.done if self.done else float("nan")
        remaining = (self.total - self.done) * per_step
        suffix = f"  {extra}" if extra else ""
        detail = f" {name}" if name else ""
        log.info(
            "[%d/%d] %s%s  (elapsed %s, eta %s)%s",
            self.done, self.total, self.label, detail,
            format_duration(elapsed), format_duration(remaining), suffix,
        )
        return self.done

    def __exit__(self, *exc_info: object) -> None:
        elapsed = time.monotonic() - self.started
        log.info(
            "%s: %d step(s) in %s", self.label, self.done, format_duration(elapsed)
        )
