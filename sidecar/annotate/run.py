"""Start the annotator: `sidecar/.venv/Scripts/python.exe -m annotate.run [--out ...] [--weights ...]`.

Printed on stdout as `ANNOTATE_PORT=<n>`, the same handshake `sidecar/run.py` uses, so a script or a
human can find the URL without guessing. The port is picked the same way too - an OS-assigned free
port when the default is taken - because the desktop app may already be holding 8765 with the live
sidecar, and an annotator that refused to start while the app was open would be useless in exactly
the workflow it exists for (label a frame, look at it in the app, label the next).

Import path: this package imports `label_classes` from `sidecar/tools/`, which is not on `sys.path`
when Python starts. The tools directory is added here, once, at the entrypoint - and **only** here,
so the app's own modules keep the dependency direction they have now.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

SIDECAR_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = SIDECAR_DIR / "tools"
if str(SIDECAR_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(SIDECAR_DIR))
if str(TOOLS_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(TOOLS_DIR))

# `store` comes with the same import-path requirement as `workspace`, and it is imported here
# rather than inside `main()` because `build_parser` names the labels directory in its help text:
# a constant that only exists after argument parsing would be a NameError in `--help`.
from app.loops import (  # noqa: E402  (same requirement, one choice)
    SERVER_CONCURRENCY_LIMIT,
    SERVER_LOOP_SETTING,
    startup_line,
)
from workspace import DEFAULT_OUT, MANIFEST_NAME, resolve_extras  # noqa: E402  (needs the path above)
from .store import ANNOTATIONS_DIRNAME  # noqa: E402  (same, and it is where the name lives)


def pick_port(preferred: int = 8770) -> int:
    """`preferred`, or whatever the OS hands back when it is busy - reported either way."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free:
        free.bind(("127.0.0.1", 0))
        return int(free.getsockname()[1])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="the staged set to label (has the manifest)")
    ap.add_argument(
        "--extras",
        action="append",
        default=[],
        metavar="DIR",
        # Defaulted rather than left empty on purpose: the hard negatives are their own staged set
        # (`cleaned-negatives/`), so a worklist built from `cleaned-v2/` alone cannot see the 50
        # frames whose whole job is to teach the model what these products are not. Leaving them
        # out is silent, and the dataset would be v1 again. Resolved after `--out` (see `main`),
        # because the default is the set staged beside it and `--extras` adds to it.
        help="another staged set to include (repeatable). Adds to the staged hard negatives beside "
        "--out - pass --no-extras to mean only what you name",
    )
    ap.add_argument(
        "--no-extras",
        action="store_true",
        help="label only the sets named with --extras, not the staged hard negatives beside --out",
    )
    ap.add_argument(
        "--annotations",
        default="",
        # The store owns the name; this is the same constant the tool that *reads* the tree keeps a
        # copy of, and `tests/test_annotate_store.py` pins the two against each other.
        help=f"where the labels live (default: <workspace>/{ANNOTATIONS_DIRNAME})",
    )
    ap.add_argument("--weights", default="", help="which weight may suggest (default: by roster)")
    ap.add_argument("--conf", type=float, default=0.25, help="suggestion confidence floor")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument(
        "--hosted",
        default="",
        help="second-opinion provider for frames the local weight misses: openai | gemini | anthropic "
        "(default: none, and nothing leaves the machine)",
    )
    ap.add_argument(
        "--max-hosted-calls",
        type=int,
        default=100,
        help="hard ceiling on paid calls in one session",
    )
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return ap


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - exercised by hand
    args = build_parser().parse_args(argv)

    from .app import AnnotateState, build_annotate_app, choose_weights
    from .store import LabelStore

    out = Path(args.out).expanduser()
    if not (out / MANIFEST_NAME).is_file():
        raise SystemExit(f"no manifest at {out} - stage a capture set first (clean_v2.py clean)")

    # Resolved after `--out`: the default is the set staged *beside* it, the same rule
    # `--annotations` follows below. Anchoring on the workspace instead would put this
    # workspace's 50 hard negatives in the worklist of a run against a set staged elsewhere.
    #
    # A union, not either/or: naming a second set (a later session's, say) must not shrink the
    # worklist back to `--out` and leave the negatives unlabeled - a frame nobody labels is a
    # frame the merge drops, and the snapshot would report the set as complete.
    extras = resolve_extras(out.parent, args.extras, include_defaults=not args.no_extras)

    models_dir = SIDECAR_DIR / "models"
    weight, note = choose_weights(models_dir, args.weights)
    annotations = (
        Path(args.annotations).expanduser()
        if args.annotations
        else out.parent / ANNOTATIONS_DIRNAME
    )

    store = LabelStore(out=out, annotations=annotations, extras=extras)
    state = AnnotateState(
        store=store,
        weight=weight,
        weight_note=note,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        hosted_name=args.hosted,
        max_hosted_calls=args.max_hosted_calls,
    )

    frames = store.frames()
    done = sum(1 for f in frames if f.state != "unlabeled")
    print(f"images     {out}")
    print(f"labels     {annotations}")
    for extra in extras:
        print(f"also       {extra}")
    print(f"worklist   {len(frames)} frame(s), {done} already decided, {len(frames) - done} outstanding")
    print(f"suggesting from {weight.name} ({weight.resize_mode})" if weight else f"no local weight: {note}")
    if args.hosted:
        print(f"second opinion {args.hosted}, at most {args.max_hosted_calls} paid call(s) this session")
    else:
        print("second opinion off - nothing leaves this machine for suggestions")
    if args.dry_run:
        return 0

    # Only on a real run. `classes.json` is the file `label_progress --source auto` reads as "the
    # annotator has been used against this set", so writing it from a --dry-run would tell that
    # tool a local labeling session exists - and the snapshot would then report 0 decided over a
    # project that has annotations. A dry run writes nothing at all, which is what it says.
    store.write_classes()

    import uvicorn

    port = args.port or pick_port()
    print(f"ANNOTATE_PORT={port}", flush=True)
    print(startup_line(), flush=True)
    # The loop is named rather than left to uvicorn, and this server is the one that needs it most:
    # uvicorn's Windows default is the proactor loop, whose accept path closes the listening socket
    # on the first aborted connection and never re-arms (`app/loops.py` carries the mechanism). A
    # labeling pass runs for hours with a browser reloading pages against it - the likeliest way to
    # abort a connection - and the failure leaves the process alive with nothing listening, so the
    # page would simply stop loading while `python -m annotate.run` sat there in `tasklist`.
    uvicorn.run(
        build_annotate_app(state),
        host="127.0.0.1",
        port=port,
        log_level="warning",
        loop=SERVER_LOOP_SETTING,
        # A labeling pass is hours long with a browser in front of it, and a browser that stops
        # getting answers opens connections rather than waiting: past this many, uvicorn answers
        # 503 and closes instead of piling the work up behind a page nobody is watching.
        limit_concurrency=SERVER_CONCURRENCY_LIMIT,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
