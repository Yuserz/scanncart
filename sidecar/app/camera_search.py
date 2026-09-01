"""Find the best value for a control, given something that measures the result.

Pure and device-free by design: `probe` is a callable the caller supplies, so
these algorithms are tested against synthetic response curves rather than a
camera. That matters because the two curves this codebase searches have
different shapes and each needs its own algorithm — brightness against
exposure (or against brightness) rises monotonically, so the value hitting a
target band can be bracketed; sharpness against focus peaks at the subject's
distance, so the maximum has to be hunted.

Values are snapped to `step` because every control here is an integer grid
(exposure is 14 whole stops, focus 0-1023), and a search that proposes 412.7
would have it silently truncated by the driver.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class SearchResult:
    value: float
    metric: float
    # Device writes actually spent. Each costs settle frames, so callers
    # budget in probes rather than seconds.
    probes: int
    # Spread of metric values seen across the search. Near-zero means the
    # control did nothing measurable — for a peak search that is "nothing was
    # in frame", which is the failure a staged scene exists to prevent.
    span: float
    # Whether the search achieved what it set out to: hit the target band
    # (search_to_target) or found a peak distinguishable from noise
    # (search_for_peak).
    reached: bool


def _snap(value: float, step: float) -> float:
    return round(value / step) * step


def search_to_target(
    probe: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    *,
    tolerance: float,
    step: float = 1.0,
    max_probes: int = 10,
) -> SearchResult:
    """Bracket a monotonically increasing `probe` onto `target`.

    Both bounds are genuine candidates, not just the edges of a bracket a
    plain binary search never quite reaches: if `hi` itself undershoots the
    target, `hi` IS the answer (more of the control does not exist); if
    `lo` itself already overshoots, `lo` IS the answer. Neither of those
    needs bracketing, so they are checked directly before any interior
    point is probed. Only when the target lies strictly between them does
    the loop narrow onto an interior value.

    Returns the closest value found even when the target is out of reach —
    a room too dark for any exposure is still better served by the longest
    exposure than by nothing. `reached` says which happened. Every returned
    `value` was itself probed, and `max_probes` is honoured exactly: a
    caller asking for N probes never gets N+1.
    """
    lo, hi = _snap(lo, step), _snap(hi, step)
    if hi < lo:
        lo, hi = hi, lo

    metrics: list[float] = []
    best_value = lo
    best_metric: float | None = None

    def record(value: float, metric: float) -> None:
        nonlocal best_value, best_metric
        metrics.append(metric)
        if best_metric is None or abs(metric - target) < abs(best_metric - target):
            best_value, best_metric = value, metric

    def finish() -> SearchResult:
        assert best_metric is not None
        return SearchResult(
            value=best_value,
            metric=best_metric,
            probes=len(metrics),
            span=max(metrics) - min(metrics),
            reached=abs(best_metric - target) <= tolerance,
        )

    if max_probes <= 0:
        # No budget at all: nothing was measured, so there is nothing to
        # report as reached.
        return SearchResult(value=best_value, metric=0.0, probes=0, span=0.0, reached=False)

    if hi <= lo:
        # Single point (a degenerate or sub-step range): nothing to bracket.
        record(lo, probe(lo))
        return finish()

    # `hi` first: if even the top of the range falls short of target, `hi`
    # is the closest achievable and bracketing between it and `lo` would
    # only waste probes confirming what monotonicity already guarantees.
    record(hi, probe(hi))
    if best_metric <= target or len(metrics) >= max_probes:
        return finish()

    # `lo`: if the floor already overshoots, `lo` is the closest achievable.
    record(lo, probe(lo))
    if metrics[-1] >= target or len(metrics) >= max_probes:
        return finish()

    # The target lies strictly between the two bounds: bracket it.
    while len(metrics) < max_probes and hi - lo >= step:
        mid = _snap(lo + (hi - lo) / 2, step)
        if mid <= lo or mid >= hi:
            break
        metric = probe(mid)
        record(mid, metric)
        if abs(metric - target) <= tolerance:
            break
        # Monotonic: below target means we need more of the control.
        if metric < target:
            lo = mid
        else:
            hi = mid

    return finish()


def search_for_peak(
    probe: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    step: float = 1.0,
    min_span: float = 0.0,
    max_probes: int = 20,
) -> SearchResult:
    """Ternary search for the maximum of a unimodal `probe`.

    Both endpoints are measured first. A plain ternary search never evaluates
    them, and a camera focused at infinity peaks exactly there — so without
    this the most common fixed-camera answer is the one answer unreachable.
    """
    cache: dict[float, float] = {}

    def at(value: float) -> float:
        if value not in cache:
            cache[value] = probe(value)
        return cache[value]

    lo, hi = _snap(lo, step), _snap(hi, step)

    # Probe endpoints, respecting the budget. Both are essential to detect
    # peaks at range boundaries (e.g. focus at infinity).
    if len(cache) < max_probes:
        at(lo)
    if hi != lo and len(cache) < max_probes:
        at(hi)

    while hi - lo > step * 2 and len(cache) + 2 <= max_probes:
        m1 = _snap(lo + (hi - lo) / 3, step)
        m2 = _snap(hi - (hi - lo) / 3, step)
        if m1 >= m2:
            break
        if at(m1) < at(m2):
            lo = m1
        else:
            hi = m2

    # A zero budget buys no measurement, so report nothing was found.
    if not cache:
        return SearchResult(
            value=lo,
            metric=0.0,
            probes=0,
            span=0.0,
            reached=False,
        )

    best_value = max(cache, key=lambda v: cache[v])
    metrics = list(cache.values())
    span = max(metrics) - min(metrics)

    return SearchResult(
        value=best_value,
        metric=cache[best_value],
        probes=len(cache),
        span=span,
        reached=span >= min_span,
    )
