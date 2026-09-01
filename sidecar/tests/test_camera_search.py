from app.camera_search import search_for_peak, search_to_target


def test_binary_search_lands_on_the_target_of_a_monotone_curve():
    """Brightness rises monotonically with exposure, so the value that hits
    the target band can be bracketed rather than swept exhaustively."""
    # metric(v) = 10*v + 200 -> metric == 130 at v == -7
    result = search_to_target(lambda v: 10 * v + 200, lo=-13, hi=0, target=130.0, tolerance=6.0)

    assert result.value == -7
    assert result.reached is True
    assert result.probes <= 5


def test_binary_search_reports_a_target_it_could_not_reach():
    """A room too dark for any exposure to fix. The best value found is still
    worth recommending; the caller needs to know it fell short."""
    result = search_to_target(lambda v: 20.0, lo=-13, hi=0, target=130.0, tolerance=6.0)

    assert result.reached is False
    assert result.metric == 20.0


def test_binary_search_respects_its_probe_budget():
    calls = []

    def probe(v):
        calls.append(v)
        return 0.0

    search_to_target(probe, lo=0, hi=1023, target=130.0, tolerance=0.1, max_probes=4)
    assert len(calls) <= 4


def test_ternary_search_finds_the_peak_of_a_unimodal_curve():
    """Sharpness against focus peaks at the subject's distance."""
    # An inverted parabola peaking at 400.
    result = search_for_peak(lambda v: 100.0 - ((v - 400) / 100.0) ** 2, lo=0, hi=1023, step=16)

    assert abs(result.value - 400) <= 32
    assert result.reached is True


def test_ternary_search_finds_a_peak_sitting_on_an_endpoint():
    """A camera focused at infinity peaks at one end of the range, where a
    plain ternary search never probes."""
    result = search_for_peak(lambda v: 100.0 - v / 100.0, lo=0, hi=1023, step=16)

    assert result.value == 0


def test_a_flat_curve_reports_no_peak_was_found():
    """Nothing in frame: sharpness does not move, so there is no peak. Saying
    so beats returning a confident wrong distance."""
    result = search_for_peak(lambda v: 42.0, lo=0, hi=1023, step=16, min_span=5.0)

    assert result.reached is False
    assert result.span == 0.0


def test_ternary_search_does_not_re_probe_a_value_it_already_measured():
    """Each probe costs a device write plus settle frames. The bracket shares
    an endpoint between iterations; measuring it twice doubles the cost."""
    calls = []

    def probe(v):
        calls.append(v)
        return 100.0 - ((v - 400) / 100.0) ** 2

    search_for_peak(probe, lo=0, hi=1023, step=16)
    assert len(calls) == len(set(calls))


def test_ternary_search_respects_its_probe_budget():
    """The endpoints are probed before the ternary loop begins, and used to
    bypass the budget entirely — a caller asking for one probe got two."""
    calls = []

    def probe(v):
        calls.append(v)
        return 100.0 - ((v - 400) / 100.0) ** 2

    for budget in (0, 1, 2, 5):
        calls.clear()
        search_for_peak(probe, lo=0, hi=1023, step=16, max_probes=budget)
        assert len(calls) <= budget, f"budget {budget} spent {len(calls)}"
