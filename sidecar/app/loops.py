"""How every uvicorn server in this repo is served: which loop, which selector, which bound.

The loop, first, and why it is not uvicorn's default on Windows.

uvicorn chooses the loop itself, and on Windows that means `asyncio.ProactorEventLoop` - literally,
in `uvicorn/loops/asyncio.py`: `if sys.platform == "win32" and not use_subprocess: return
asyncio.ProactorEventLoop`. For this app that default has a failure mode that does not look like
one: **the first aborted connection can stop the server accepting for good, while the process stays
alive and healthy.**

The mechanism is CPython's proactor accept loop (`asyncio/proactor_events.py`,
`BaseProactorEventLoop._start_serving`), which is a self-rearming callback:

    def loop(f=None):
        try:
            if f is not None:
                conn, addr = f.result()      # <- raises here, on any accept error
            ...
            f = self._proactor.accept(sock)  # <- re-arms only on the happy path
        except OSError as exc:
            if sock.fileno() != -1:
                self.call_exception_handler({'message': 'Accept failed on a socket', ...})
                sock.close()                 # <- the listening socket is closed
        ...
        else:
            self._accept_futures[sock.fileno()] = f
            f.add_done_callback(loop)

An accept that fails therefore logs `Accept failed on a socket` with `OSError: [WinError 64] The
specified network name is no longer available`, **closes the listening socket**, and never re-arms.
Nothing outlives it: `/api/health` stops answering, every new request and WebSocket connection
hangs, and the renderer sits in front of a dead API while the process keeps running - it is still in
`tasklist`, still holding its camera and its threads. On Windows an aborted connection is an ordinary
event rather than an attack (a renderer that reloads mid-request, a probe cancelled when a window
closes), and over loopback it is easy to produce.

That is not a theory about someone else's stack; it is what this app did. Reproduced against this
repo's own `run.py` by opening 400 connections and aborting them (`SO_LINGER 0`) before the handshake
finished: the log carried exactly that traceback, and the process was left alive with **no listening
socket at all** - the port gone from `netstat`, `/api/health` unreachable, the pid still there. A
sidecar that had been unresponsive on 8765 for a whole session while Electron stayed up had the same
shape: alive, self-pipe sockets open, nothing listening.

The selector loop has no such branch. Its listener re-fires off the selector, and a failed accept is
re-raised out of `_accept_connection` into the event loop's own callback handling - CPython's
comment is `# The event loop will catch, log and ignore it.` - so it is logged and the reader stays
registered, which means the next connection is still accepted. The same 400 aborted connections
against a server on this loop cost nothing but lines in the log.

What this gives up is small, but it is not nothing, and it is written here as measured rather than
as a footnote: `ProactorEventLoop` is the only loop that supports `asyncio` subprocesses and this
app has none (the PowerShell probes in `hardware.py` and `cameras.py` are synchronous), while the
selector loop on Windows is `select()` - and `select()` takes a **count** of descriptors it will not
exceed. Measured on this machine, in this venv: 512 descriptors in one call return normally and 513
raise `ValueError: too many file descriptors in select()`, straight out of the event loop and not
caught anywhere. The count is of *registered* descriptors, which for a server means its live
connections (plus the listener and the self-pipe), so what reaches the ceiling is a **flood**: held
open, 520 silent connections kill the loop and the process with it - measured, `health=refused`
afterwards with `too many file descriptors in select()` in the log. Reaching it takes the whole
flood at once rather than a client count, which is why the same family of storm has been measured to
cost nothing in one server and everything in another: the sidecar's own server shrugged off 32
concurrent connection threads aborting for fifteen seconds, the annotator's 8, and the inference
server died at four because a heavier app holds accepted sockets open longer, so the same storm piles
them up. A *serial* flood of thousands of aborted connections never reaches it at all, and the
proactor by comparison dies on the first aborted accept, which for a server the sidecar calls under a
timeout is an ordinary event (the timeout closes the connection, and the server sees an aborted
accept). So the trade is a rarer and louder ceiling for a routine and silent death. Everywhere else
`SelectorEventLoop` is what uvicorn already uses - Linux gets epoll, which has no such cap - so this
is that same choice made explicitly rather than a new one.

That ceiling is what `BatchedSelectSelector` is for, and why this module is not only a choice of
loop: it polls in batches of `SELECT_BATCH` descriptors, so a list of any length is handed to
`select()` in pieces it accepts, and the loop keeps serving under a flood instead of raising out of
its own scheduler. Only reached where `select()` *is* the platform's best option (`DefaultSelector`
being `SelectSelector` is the test, which is Windows and a stripped-down POSIX build and nothing
else), so a Linux box keeps epoll and the batching costs it nothing; where it is used, the cost is
one extra `select()` call per batch per iteration, with every batch but the last polled at a zero
timeout so the loop's own scheduling is unchanged.

Two things could still swamp a server, and only one of them is ours to refuse. A **request** flood is
handed `limit_concurrency` (`SERVER_CONCURRENCY_LIMIT`): uvicorn answers the connections past that
bound with `503 Service Unavailable` and `connection: close` instead of working on them, so a client
with a runaway retry loop is turned away quickly rather than piling up work. A flood of connections
that never send a request - 520 sockets opened and held, which is the shape measured above - cannot
be answered with anything, because uvicorn decides at headers-complete and there are no headers: that
one is survived by the batched selector rather than shed, and it is the reason both halves exist.

Three servers in this repo serve over uvicorn and every one of them names the choices below - the loop
and its selector, then the bound - `run.py`
(the app's own API, where this was found), `annotate/run.py` (the labeling pass - hours long, with a
browser reloading pages against it, which makes it the likeliest place to abort a connection and the
most expensive place to lose the server), and `local_inference_server.py` (what `local_api` calls
during a capture). `tests/test_server_loop.py` scans for uvicorn call sites and requires each serving
one to name both constants, so a fourth server cannot be added without the choices - and
`tests/test_server_flood.py` is where the two floods above are actually run against a real server,
because neither of them is visible without sockets in front of one.

That is also why this module has to stay dependency-free: it is imported by the inference server,
which runs in its own venv precisely because the `inference` package's numpy/opencv pins conflict
with the ultralytics stack. An import here that pulled in `cv2`, `numpy` or `ultralytics` would be an
import that venv cannot make; `asyncio`, `select` and `selectors` are all this module may have, and
`tests/test_server_loop.py` reads its imports to keep it that way.
"""

from __future__ import annotations

import asyncio
import select
import selectors

#: How many descriptors one `select()` call may be handed. The ceiling is `FD_SETSIZE` - 512 on the
#: machine the numbers above were measured on, where 512 is accepted and 513 raised - and this sits
#: under it rather than on it because the count is the platform's and the constant is ours: a build
#: with a smaller `FD_SETSIZE` should fail a test here, not a server on someone's counter.
SELECT_BATCH = 500


class BatchedSelectSelector(selectors.SelectSelector):
    """`SelectSelector` that never hands `select()` more descriptors than it accepts.

    The base class passes every registered descriptor in one call, so the count that reaches
    `select()` is the number of live connections - and past the platform's `FD_SETSIZE` that call
    raises `ValueError` out of the loop's scheduler, which ends the loop and the server with it.
    Polling in batches makes the same descriptors survive: every batch but the last is polled with a
    zero timeout, so an early batch that has something ready returns immediately (a descriptor that
    was ready stays ready, `select()` being level-triggered) and the rest of the wait is spent on
    the last batch, which is the same schedule the single call would have kept.

    Overridden on every platform rather than only on Windows: `_select` is the one method that
    differs between them, this does what both do, and the place it is *used* is already decided by
    which selector the platform prefers.
    """

    def _select(self, r, w, _, timeout=None):
        readers, writers = sorted(r), sorted(w)
        if len(readers) <= SELECT_BATCH and len(writers) <= SELECT_BATCH:
            return select.select(r, w, w, timeout)

        # Round-robin rather than contiguous slices: whichever batch an fd lands in, it is polled
        # every iteration, so no descriptor can be starved by the batch before it.
        batches = max(1, -(-max(len(readers), len(writers)) // SELECT_BATCH))
        ready_r: list[int] = []
        ready_w: list[int] = []
        for index in range(batches):
            last = index == batches - 1
            r_batch = readers[index::batches]
            w_batch = writers[index::batches]
            r_now, w_now, _ = select.select(r_batch, w_batch, w_batch, timeout if last else 0)
            ready_r += r_now
            ready_w += w_now
            if not last and (r_now or w_now):
                break
        return ready_r, ready_w, []


def _selector_for_this_platform() -> type[selectors.BaseSelector]:
    """`select()` in batches where `select()` is what this platform has, and its own best otherwise.

    One decision with two readers: the loop uses it, and `startup_line()` names it. The alternative
    would be a second copy of "is this Windows" in the line, which is the kind of pair that drifts.
    """
    if selectors.DefaultSelector is selectors.SelectSelector:
        return BatchedSelectSelector
    return selectors.DefaultSelector


class SelectorLoop(asyncio.SelectorEventLoop):
    """The loop every uvicorn server in this repo serves on, with the selector that survives a flood.

    A class, and **not** a function that returns one: uvicorn resolves `SERVER_LOOP_SETTING` and
    passes the result to `asyncio.Runner(loop_factory=...)`, which *calls* it to get the loop - so a
    function that returned the *class* would hand the Runner a class where a loop belongs and kill
    the process at startup with `TypeError: BaseEventLoop.create_task() missing 1 required
    positional argument`. `asyncio_loop_factory` returns `asyncio.ProactorEventLoop` the same way,
    and its own `use_subprocess` argument is the only difference between the two of them.

    It derives from `asyncio.SelectorEventLoop`, so this is uvicorn's own choice of family named
    explicitly; all it adds is the selector above.
    """

    def __init__(self) -> None:
        super().__init__(_selector_for_this_platform()())


#: The same fact in the only shape uvicorn takes it in, spelled out so every entrypoint has something
#: to pass and a test has something to resolve. Kept beside the class it names because the two are
#: one decision: the string is resolved by import, so a rename on either side is a *startup* failure
#: rather than a wrong choice, and `tests/test_server_loop.py` resolves it through uvicorn's own
#: `Config.get_loop_factory()` - the path the server really takes - so the pair cannot drift.
SERVER_LOOP_SETTING = "app.loops:SelectorLoop"

#: The most connections - or in-flight requests - any server here will work on at once. Past it,
#: uvicorn answers `503 Service Unavailable` and closes the connection (`flow_control`'s
#: `service_unavailable`, reached in `on_headers_complete` when `len(connections)` or `len(tasks)` is
#: at the bound), which is the difference between a flood that is *shed* and one that is queued: a
#: runaway retry loop, a port scanner or a browser hammering a stale tab is turned away in
#: milliseconds instead of holding a socket, a parser and a task for as long as it likes.
#:
#: 64 because it is nowhere near any of this app's own traffic - a renderer holds a websocket and a
#: handful of REST reads, and the sidecar makes one workflow call at a time - and far under the count
#: `select()` takes, so a request flood is refused while there is still headroom for the connections
#: that are being served. Deliberately a *policy* bound rather than the platform's: it is a number
#: about this app's traffic, and the platform's 512 is not one a busy server should be measuring
#: itself against.
#:
#: It counts HTTP connections, which is worth knowing before reading a 503 as something it is not: a
#: websocket is not counted (uvicorn checks in its HTTP protocols), and that is the right answer here
#: - the renderer's stream is one connection for the life of the window, which in a bound would be a
#: *reserved* slot rather than a flood control.
SERVER_CONCURRENCY_LIMIT = 64


def startup_line() -> str:
    """One line naming the loop, and the selector under it, which every server here prints as it starts.

    The tests can see the choice, and the code can be read for it, but neither helps the person
    looking at a server that is behaving oddly on a machine they cannot instrument: a *log* is what
    they have, and this line versus a proactor loop is the difference between the bug this module
    exists for and a second bug on top of it. The selector is named beside the loop because it is
    the half that answers the count ceiling above: `selector=BatchedSelectSelector` is a server that
    will keep serving a flood, and `selector=SelectSelector` one that will not. It is printed rather
    than logged because these entrypoints have no logging configuration of their own - one `print`
    to stdout, beside the port each of them already announces.

    Both names come from the objects rather than from spellings of them, so a line cannot describe a
    loop or a selector other than the ones being served on. `tests/test_server_loop.py` holds the
    other half: it resolves `SERVER_LOOP_SETTING` through uvicorn's own `get_loop_factory()` and
    requires *that* class's name to be what this returns, and it scans every uvicorn call site in
    the repo for the `print` below - three servers today, and a fourth cannot start quietly.
    """
    return f"EVENT_LOOP={SelectorLoop.__name__}(selector={_selector_for_this_platform().__name__})"
