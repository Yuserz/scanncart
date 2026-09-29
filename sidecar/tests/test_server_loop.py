"""The sidecar's event loop, and the two failures the choice exists to avoid.

uvicorn picks the loop for us, and on Windows its pick is `asyncio.ProactorEventLoop`. That loop's
accept path is a self-rearming callback which, on *any* failed accept, closes the listening socket
and never re-arms (`asyncio/proactor_events.py`, `BaseProactorEventLoop._start_serving`: the
`except OSError` branch calls `sock.close()` and skips the `else` that re-arms). The log gets
`Accept failed on a socket` with `OSError: [WinError 64] The specified network name is no longer
available`, and after it the process is still running while nothing is listening - `/api/health`
hangs, every new request and WebSocket connection hangs, and the renderer sits in front of a dead
API until someone restarts the app. An aborted connection is an ordinary event on Windows, not an
attack: a renderer that reloads mid-request, a probe cancelled when a window closes.

Reproduced against this repo's own `run.py` by opening 400 connections and aborting them
(`SO_LINGER 0`) before the handshake completed. On the proactor loop: that traceback, then
`health=UNREACHABLE`, `listening=0`, `alive=True` - the port gone from `netstat`, the pid still in
`tasklist`. The same 400 against a server on the selector loop: `health=idle`, `listening=1`, not one
error line, because the selector's listener is re-armed off the selector and a failed accept is
re-raised into the event loop's own callback handling - CPython's comment is `# The event loop will
catch, log and ignore it.` - which leaves the reader registered.

The second failure is the selector loop's own ceiling, and it is a *count*: `select()` takes a fixed
number of descriptors - 512 on this machine, where 512 in one call returns and 513 raises
`ValueError: too many file descriptors in select()`, out of the event loop's scheduler and not caught
anywhere. The count is of the descriptors registered with the loop, which for a server means its live
connections, so a flood that is *held open* reaches it: measured here, 520 silent connections ended a
real server on the plain selector (`health=refused`, that traceback in its log). That is what
`BatchedSelectSelector` answers, and why these tests check a selector as well as a loop.

What these tests can check on any platform is the *choice*, which is the part that regresses, and the
way it reaches uvicorn, which is subtler than it looks. The socket races are in
`tests/test_server_flood.py` - they need a real server, and the one this file keeps out is the write
that would pass on Linux without exercising anything. `app/loops.py` is where the mechanisms are
written down; this is the guard that keeps the decisions from being quietly undone.
"""

from __future__ import annotations

import ast
import asyncio
import re
import selectors
import socket
import sys
from pathlib import Path

import pytest
import uvicorn

from app import loops

from app_run import server_config

SIDECAR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SIDECAR_DIR.parent

PROACTOR = "the proactor loop, whose accept path closes the listening socket and never re-arms"


def test_uvicorn_resolves_the_setting_to_the_selector_loop():
    """Through uvicorn's own resolver, because that is the half that can silently break.

    It did break while this was written: the setting named a function that *returned* the loop class,
    and uvicorn hands the resolved value straight to `asyncio.Runner(loop_factory=...)`, which calls
    it - so the server died at startup with a missing-argument `TypeError` instead of serving on the
    wrong loop. Asking uvicorn rather than re-importing the string here is what catches that before
    someone launches the app and watches the sidecar exit.
    """
    resolved = uvicorn.Config(None, loop=loops.SERVER_LOOP_SETTING).get_loop_factory()
    assert resolved is loops.SelectorLoop


def test_the_loop_is_the_selector_family_and_not_the_proactor():
    assert issubclass(loops.SelectorLoop, asyncio.SelectorEventLoop), PROACTOR
    if sys.platform == "win32":
        # The platform this choice is *for*: everywhere else uvicorn already picks the selector loop,
        # so Windows is where it is a change rather than a restatement of uvicorn's own default.
        assert not issubclass(loops.SelectorLoop, asyncio.ProactorEventLoop), PROACTOR


def test_the_loop_can_actually_be_constructed():
    """A factory uvicorn can call and get a loop out of - the shape, not only the name."""
    loop = loops.SelectorLoop()
    try:
        assert isinstance(loop, asyncio.AbstractEventLoop)
    finally:
        loop.close()


def test_the_batched_selector_answers_where_the_plain_one_raises():
    """The ceiling itself, without a server: the base class raises here and the batched one does not.

    The count is past Windows' `FD_SETSIZE` (512, the limit 513 descriptors hit) and deliberately
    read as a number rather than probed: the platform's constant is not exported, and a test that
    discovered it would be a test of CPython. The plain selector half is Windows-only for the same
    reason - POSIX `select()` takes `FD_SETSIZE` (1024) descriptors and is not being exercised by
    520 - while the batched half runs everywhere, which is the property that matters: the same list
    is polled rather than refused.
    """
    count = 520
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        if sys.platform == "win32":
            with selectors.SelectSelector() as plain:
                for sock in sockets:
                    plain.register(sock, selectors.EVENT_READ)
                with pytest.raises(ValueError, match="too many file descriptors"):
                    plain.select(0)

        with loops.BatchedSelectSelector() as batched:
            for sock in sockets:
                batched.register(sock, selectors.EVENT_READ)
            # Nothing is ready, and that is the whole assertion: the call returned instead of
            # raising out of the event loop's scheduler.
            assert batched.select(0) == []
    finally:
        for sock in sockets:
            sock.close()


def test_the_batched_selector_reports_a_descriptor_outside_the_first_batch():
    """Batching must not lose a descriptor, and the one it could lose is past the boundary.

    Two listening sockets are made ready, the lowest and the highest descriptor of a set larger than
    one batch, so one of them can only be seen by a later batch. Both have to arrive: a split that
    polled only its first batch - or that returned early and never came back to the rest - would
    report the low one and starve the high one, and a server whose listener was starved in favour of
    its busiest connections would stay up while accepting nothing.
    """
    count = loops.SELECT_BATCH + 3
    idle = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count - 2)]
    listeners = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
    sockets = sorted(idle + listeners, key=lambda s: s.fileno())
    selector = loops.BatchedSelectSelector()
    clients: list[socket.socket] = []
    try:
        for sock in sockets:
            selector.register(sock, selectors.EVENT_READ)
        # `listen()` makes a socket readable as soon as anything connects to it, which is what gives
        # the selector something to report without a byte being sent.
        for listener in (sockets[0], sockets[-1]):
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(listener.getsockname())
            clients.append(client)

        ready = {key.fileobj for key, _ in selector.select(0)}
        assert sockets[0] in ready, "the lowest descriptor was not reported"
        assert sockets[-1] in ready, (
            f"the highest of {count} descriptors was not reported - it is past the batch boundary, "
            "which is where a batching bug shows up"
        )
    finally:
        for sock in clients:
            sock.close()
        for sock in sockets:
            sock.close()
        selector.close()


def test_run_py_serves_the_config_carrying_that_setting():
    """The wiring, end to end: the entrypoint's own config, asked the way the server asks it."""
    config = server_config(None, 1234)
    assert config.get_loop_factory() is loops.SelectorLoop
    assert config.host == "127.0.0.1"
    assert config.port == 1234


# --- and every *other* uvicorn server in this repo ---------------------------------------------

CALL = re.compile(r"uvicorn\.(?:run|Config)\(")
#: The one choice, in either spelling a call site uses: bare (the module that imported it) or through
#: the package (`loops.SERVER_LOOP_SETTING`, which is how this file names it).
#: The one choice, in either spelling a call site uses: bare (the module that imported it) or through
#: the package (`loops.SERVER_LOOP_SETTING`, which is how this file names it).
NAMES_THE_LOOP = re.compile(r"loop=(?:loops\.)?SERVER_LOOP_SETTING\b")
#: The bound, the same way and for the same reason: `limit_concurrency` is what turns a request
#: flood into 503s instead of queued work, and a server that forgot it would be the only one here
#: that can be made to hold connections it is not serving.
NAMES_THE_BOUND = re.compile(r"limit_concurrency=(?:loops\.)?SERVER_CONCURRENCY_LIMIT\b")
#: The servers this guard is *for*. Named rather than counted, so a scan that found nothing at all
#: fails here instead of passing by looking at no code.
ENTRYPOINTS = (
    "sidecar/run.py",
    "sidecar/annotate/run.py",
    "sidecar/local_inference_server.py",
)


def _call_sites() -> list[Path]:
    """Every hand-written `.py` under `sidecar/` - no venvs, no caches, no generated data."""
    skip = {"__pycache__", "node_modules", "data"}
    return [
        path
        for path in SIDECAR_DIR.rglob("*.py")
        if not any(p.startswith(".") or p in skip for p in path.relative_to(SIDECAR_DIR).parts)
    ]


def _uvicorn_calls(text: str) -> list[str]:
    """Each uvicorn `run` / `Config` call as one string, parens balanced.

    Parsed rather than found per line so that a call the formatter wrapped still reads as one call,
    and so that a `loop=` in a neighbouring call cannot vouch for this one. The call is spelled
    here without its opening parenthesis on purpose: this scan reads *text*, so an example of the
    shape it looks for is itself a match, and a docstring that tripped its own guard would be fixed
    by weakening the guard.
    """
    calls = []
    for match in CALL.finditer(text):
        depth = 0
        for i in range(match.end() - 1, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(text[match.start() : i + 1])
                    break
    return calls


def test_the_loop_module_stays_dependency_free():
    """The condition the inference server's import depends on, and the one that can silently break.

    `app/loops.py` is imported by `local_inference_server.py`, which runs in `.venv-inference` -
    a venv that exists precisely because the `inference` package's numpy/opencv pins conflict with
    the ultralytics stack. So it is the one file under `app/` that must import nothing but the
    standard library: an `import cv2` added here for a convenience would be an import that venv
    cannot make, and the failure would land on someone starting the inference server, not on
    whoever added it. `select` and `selectors` are the third party this file may not have and the
    first two it may.
    """
    # Parsed rather than grepped: this module's docstring quotes Python (the accept loop it
    # explains), and a line of prose that happens to start with `from` is not an import.
    tree = ast.parse((SIDECAR_DIR / "app" / "loops.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    allowed = {"asyncio", "select", "selectors", "__future__"}
    assert imported <= allowed, (
        f"app/loops.py imports {sorted(imported - allowed)} - it is imported by the inference "
        "server's venv, which has this project's dependencies deliberately not installed"
    )


def test_the_startup_line_names_the_loop_uvicorn_serves_on():
    """A log line and a resolution, tied together.

    One is a string and the other is what uvicorn's factory returns, so the only way to keep them
    from describing different loops is to ask uvicorn - the same call the test above makes. Without
    this, `EVENT_LOOP=` could name a loop nothing serves on and still be printed everywhere.
    """
    resolved = uvicorn.Config(None, loop=loops.SERVER_LOOP_SETTING).get_loop_factory()
    line = loops.startup_line()

    assert "\n" not in line, f"the startup line has to be one line: {line!r}"
    assert line.startswith("EVENT_LOOP="), line
    assert resolved.__name__ in line, (
        f"the startup line says {line!r} while uvicorn serves on {resolved.__name__} - one of the "
        "two is telling whoever reads the log the wrong thing"
    )


def test_every_uvicorn_server_in_this_repo_names_its_loop_and_its_bound():
    """One choice, three servers, and a fourth that cannot be added without it.

    `run.py` was where this was found, but the Windows default is what every uvicorn server here
    gets, and the other two are worse places to lose it: the annotator's server runs for hours
    through a labeling pass with a browser reloading pages against it, and the no-Docker inference
    server is called during a live capture. Both would keep running with nothing listening, which is
    the failure this whole guard exists for, so "only the app's API names the loop" is not a rule
    worth having.

    Read as text rather than by importing, and that is not laziness: `local_inference_server.py`
    imports the `inference` package, which is deliberately not in this venv (its numpy/opencv pins
    conflict with ultralytics), so an import-based version of this check could not run in CI at all.
    `uvicorn.Server(config)` is not scanned: it takes its loop from the config it is handed, and
    every `Config(...)` call in the tree is one of the calls below.
    """
    found: dict[str, list[str]] = {}
    for path in _call_sites():
        calls = _uvicorn_calls(path.read_text(encoding="utf-8"))
        if calls:
            found[path.relative_to(REPO_ROOT).as_posix()] = calls

    for entrypoint in ENTRYPOINTS:
        assert entrypoint in found, (
            f"{entrypoint} no longer serves over uvicorn, or this scan stopped finding it - the "
            "loop choice it used to name is what this guard is about"
        )
        # The line, not only the choice. `app/loops.py` argues for it: the tests can see the loop
        # and the code can be read for it, but a server that misbehaves on a machine nobody can
        # instrument is diagnosed from its log.
        assert "print(startup_line()" in (REPO_ROOT / entrypoint).read_text(encoding="utf-8"), (
            f"{entrypoint} does not print the loop it serves on. Call "
            "`print(startup_line(), flush=True)` beside the port it already announces."
        )

    for path, calls in found.items():
        # Where a call *serves*, as opposed to resolving a choice: a test builds a Config with no
        # app, to ask what the setting resolves to, and a config nothing connects to has no
        # connections to bound. Those calls still have to name the loop - this file's own tests
        # included, which is why the scan cannot simply skip them - and are not asked for a bound.
        serves = "tests" not in Path(path).parts
        for call in calls:
            assert NAMES_THE_LOOP.search(call), (
                f"{path} serves over uvicorn without naming the loop: {call.strip()!r}. Pass "
                "`loop=SERVER_LOOP_SETTING` from `app/loops.py`: uvicorn's Windows default is the "
                "proactor loop, whose accept path closes the listening socket on the first aborted "
                "connection and never re-arms, leaving the process alive with nothing listening."
            )
            assert NAMES_THE_BOUND.search(call) or not serves, (
                f"{path} serves over uvicorn without naming the concurrency bound: {call.strip()!r}. "
                "Pass `limit_concurrency=SERVER_CONCURRENCY_LIMIT` from `app/loops.py`: past that "
                "bound uvicorn answers 503 and closes the connection, which is the difference "
                "between a flood being shed and a flood being queued behind this server's work."
            )
