"""Suite-wide guards for the state a test could otherwise touch on the machine it runs on.

One fixture, and it exists because of one file. The source index (`app/source_index.py`) is
*machine* state - a record of the runs this machine has trained, beside the settings and the
detection log rather than beside the weights - and it is keyed by a weight's **bytes** rather than
by a path. Every fake weight in this tree is built from the same handful of bytes (`b"weights"`),
so a suite that wrote the real index would have one test answering another's question: adding an
entry during a `--install` test is enough to make `test_backfill_provenance.py`'s "a weight no run
produced is left empty" fail, and the failure would depend on test order.

That is not a hypothetical - it happened while this feature was being written, which is why the scan
is here rather than in the tests that noticed. Redirecting it also keeps the promise
`app/source_index.py` makes in its own docstring: the machine's `data/` is written by the tools and
read by the app, never by the suite. The override is the one an operator uses for the same reason
(`SCANNCART_SOURCE_INDEX`, named after `workspace.SCANNCART_DATASET_ROOT`), so the mechanism the
tests exercise is the shipped one rather than a test-only back door.

A test that wants a populated index writes one *through* `source_index.record_source` at
`source_index.index_path()`, which resolves to this file - so the fixture is also what keeps such a
test from depending on where the suite is run.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_source_index(tmp_path, monkeypatch):
    """Point the machine-local source index at a per-test file, for every test in the suite."""
    monkeypatch.setenv("SCANNCART_SOURCE_INDEX", str(tmp_path / "source_index.json"))
