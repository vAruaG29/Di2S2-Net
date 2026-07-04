"""
Step-wise timing helper for the inference / stitching / evaluation stages.

Usage:

    timer = StepTimer()

    with timer.step("Stage 1: inference"):
        ...                       # arbitrary work
        with timer.step("read tile index", indent=6):
            ...
        with timer.step("forward pass", indent=6):
            ...

    timer.print_summary()         # printed table at the end
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, field


def now_hms() -> str:
    """Wall-clock timestamp HH:MM:SS for log lines."""
    return datetime.now().strftime("%H:%M:%S")


@dataclass
class _Record:
    label: str
    indent: int
    start_iso: str
    end_iso: str
    seconds: float


@dataclass
class StepTimer:
    """Records nested step durations and prints a summary table.

    `step()` is a context manager that:
        - prints `⏱ [HH:MM:SS] START  <label>` on enter
        - prints `⏱ [HH:MM:SS] DONE   <label>  (Δt s)` on exit
        - appends a record to `self.records` for the final summary

    `cumulative()` returns a helper that totals N invocations under one
    label — useful for per-batch loops where you want one summary row,
    not one per batch.
    """

    records: list = field(default_factory=list)

    @contextmanager
    def step(self, label: str, indent: int = 4):
        t0 = time.time()
        ts_in = now_hms()
        print(f"{' ' * indent}⏱  [{ts_in}] START  {label}")
        try:
            yield
        finally:
            dt = time.time() - t0
            ts_out = now_hms()
            print(f"{' ' * indent}⏱  [{ts_out}] DONE   {label}  ({dt:.2f}s)")
            self.records.append(_Record(label, indent, ts_in, ts_out, dt))

    def stamp(self, label: str, indent: int = 4):
        """One-shot HH:MM:SS log line (no duration)."""
        print(f"{' ' * indent}⏱  [{now_hms()}] {label}")

    def cumulative(self, label: str, indent: int = 6):
        """Return a callable that tallies repeated short ops into one record.

        Use:
            tally = timer.cumulative("save_geotiff (x N)")
            for ...:
                with tally:
                    save_geotiff(...)
            tally.flush()   # adds one row to the summary
        """
        return _CumulativeBlock(self, label, indent)

    def print_summary(self, title: str = "Timing summary"):
        if not self.records:
            return
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")
        print(f"  {'Step':<55s} {'Start':>8s} {'End':>8s} {'Δt(s)':>8s}")
        print(f"  {'─' * 80}")
        for r in self.records:
            ind = " " * r.indent
            label_w = max(1, 55 - r.indent)
            label = (r.label[:label_w - 1] + "…") if len(r.label) > label_w else r.label
            print(f"  {ind}{label:<{label_w}s} {r.start_iso:>8s} {r.end_iso:>8s} {r.seconds:>8.2f}")
        total = sum(r.seconds for r in self.records if r.indent <= 4)
        print(f"  {'─' * 80}")
        print(f"  {'TOTAL (top-level)':<55s} {'':>8s} {'':>8s} {total:>8.2f}")
        print(f"{'─' * 80}\n")


class _CumulativeBlock:
    """Accumulate elapsed time across multiple `with` re-entries under one label."""

    def __init__(self, parent: StepTimer, label: str, indent: int):
        self.parent = parent
        self.label = label
        self.indent = indent
        self._total = 0.0
        self._calls = 0
        self._t0_first: str | None = None
        self._t_in: float = 0.0

    def __enter__(self):
        self._t_in = time.time()
        if self._t0_first is None:
            self._t0_first = now_hms()
        return self

    def __exit__(self, *exc):
        self._total += time.time() - self._t_in
        self._calls += 1
        return False

    def flush(self):
        """Emit one summary row for the cumulative total."""
        if self._calls == 0:
            return
        end_iso = now_hms()
        label = f"{self.label}  ({self._calls}×)"
        self.parent.records.append(
            _Record(label, self.indent, self._t0_first or end_iso, end_iso, self._total)
        )
        ind = " " * self.indent
        print(f"{ind}⏱  [{end_iso}] cumulative {self.label}: "
              f"{self._total:.2f}s over {self._calls} call(s)")
