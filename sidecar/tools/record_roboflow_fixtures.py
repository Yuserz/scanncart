#!/usr/bin/env python
"""Record the live Roboflow bodies the dataset tools read, as committed test fixtures.

Every shape assumption in this pipeline was checked against the API by hand, once, in a terminal -
and then it lived in a *comment*. `generate_version.class_order` says the project route answers
`name -> annotation count`; `label_classes.get_project` says the route wraps its payload in a
`project` key; `train_model.export_link` says the export route answers 202 with `progress` and 200
with `export.link`; `label_progress.state_of` says an image with no annotation record has a *list*
where a deliberately-null one has a dict. Each of those is a real measurement, and none of them can
fail when the API changes, because prose is not checked by anything.

`tests/test_roboflow_shapes.py` reads the files this writes. The distinction that makes them worth
committing: the *fakes* in this suite were written by whoever wrote the reader, so they agree with
the reader by construction and would keep agreeing with it after the route changed. A recording
cannot. So this is the half a fake cannot cover, and it is cheap: one command, a few small JSON
files, no network in the tests themselves.

The fixtures are **recordings, not live reads**. They carry the project's state on the day they were
taken - the class list, the image counts, the version numbers - so a test that asserts a *count* off
one is asserting history. What they are for is the *shape* (`classes` is a mapping of name to
annotation count, `versions` is a list of objects with an `id` ending in the number, an export
answer carries `export.link` or `progress`) and for the readings the tools derive from that shape.
Re-record when a route's answer changes; the test names what it pins, so a change that matters says
so in the diff instead of in a comment.

**Reads only.** Every route below is a GET except the project search, which is a POST because that
is how Roboflow's index is queried - `label_progress.fetch_all` sends the same request body, and it
writes nothing. Nothing here creates a version (`generate_version`'s `generate` route), uploads,
deletes, annotates or trains; the export request is the same one `train_model --download` makes, it
spends no version number, and the archive itself is deliberately not downloaded - the fixture is the
JSON body, not 100 MB of JPEGs.

    sidecar/.venv/Scripts/python.exe sidecar/tools/record_roboflow_fixtures.py

Writes to `sidecar/tests/__fixtures__/roboflow/`. Values that are credentials or signed links are
redacted on the way in (see `redact`), and the file it writes is JSON at a fixed `indent=2` with a
trailing newline, so a re-recording that only moved a number is a readable diff.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

import generate_version
from label_classes import WORKSPACE, load_key

# From the owner rather than retyped: this records the *same* route `--download` asks for, and a
# second copy of an identifier here would record the answer to a request no tool makes. That the
# constant might be wrong is not this tool's question - it is exactly what `--format` exists for in
# `train_model`, and what a recording of the answer would settle.
from train_model import EXPORT_FORMAT

API = "https://api.roboflow.com"
# Beside the tests that read them, and named for the route rather than for the tool that reads it:
# one recording can pin a reader in any of them.
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "__fixtures__" / "roboflow"

# A string that carries a credential or a signature. The export link is a redirect to signed
# storage - `https://app.roboflow.com/ds/7Y2YHuMDsN?key=zuux4FVom5` - and its *shape* is the fact
# worth recording (`export.link` is a string) while the token in it is a live one. The first draft
# of this tool matched `api_key` and not a bare `key=`, and duly wrote the signed link of a public
# project's export into a file about to be committed: the pattern is a query parameter list rather
# than a word list because that is what a URL's secret looks like.
SECRET_IN_URL = re.compile(r"[?&](api_key|key|token|sig|signature|expires|X-Amz-[A-Za-z]+)=", re.IGNORECASE)
# An address is personal data with no role in a shape, and the Trash body carries one in
# `deletedByName` (the value, not the key: the key is part of the shape and stays).
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")


def redact(value: object, path: str = "") -> tuple[object, list[str]]:
    """`value` with anything credential-shaped replaced, and where that happened.

    Walks dicts and lists and rewrites two things: a string containing `api_key=` or a signature
    (the export link's query, a signed thumbnail), where the query is replaced by `<redacted>` and
    the scheme, host and path are kept - a reader that finds `export.link` still finds a string; and
    any key whose name says it holds a key or a secret, which is set to `<redacted>` whole.

    Returns the redactions as `<path> (<why>)` lines so the run prints what it removed without
    printing what it removed *from*. A recording that silently dropped a token and a recording that
    silently kept one look the same on disk, and only one of them is safe.
    """
    notes: list[str] = []
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            if isinstance(key, str) and re.search(r"api_?key|secret|token|password", key, re.IGNORECASE):
                out[key] = "<redacted>"
                notes.append(f"{path}/{key} (key-shaped field name)")
                continue
            out[key], sub = redact(item, f"{path}/{key}")
            notes.extend(sub)
        return out, notes
    if isinstance(value, list):
        out_list: list = []
        for i, item in enumerate(value):
            redacted, sub = redact(item, f"{path}/{i}")
            out_list.append(redacted)
            notes.extend(sub)
        return out_list, notes
    if isinstance(value, str) and SECRET_IN_URL.search(value):
        head, _, query = value.partition("?")
        # Only the secret parameter goes: the rest of the query is part of the shape a reader may
        # key on, and a reader that found `export.link` still finds a usable-looking URL.
        kept = "&".join(
            part for part in query.split("&") if not SECRET_IN_URL.match(f"&{part}")
        )
        notes.append(f"{path} (signed URL query)")
        return f"{head}?{'&'.join(filter(None, [kept, '<redacted>']))}", notes
    if isinstance(value, str) and EMAIL.search(value):
        notes.append(f"{path} (email address)")
        return EMAIL.sub("<redacted>", value), notes
    return value, notes


def write_fixture(name: str, body: object, out: Path) -> Path:
    """One recorded body, redacted, written as `out/<name>`."""
    redacted, notes = redact(body)
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redacted, indent=2) + "\n", encoding="utf-8")
    top = ", ".join(sorted(redacted)) if isinstance(redacted, dict) else type(redacted).__name__
    print(f"  {name}  [{top}]")
    for note in notes:
        print(f"    redacted: {note}")
    return path


def get_json(url: str, key: str, **params: object) -> tuple[int, object]:
    """One GET, as the route answered it: the status *and* the body.

    Both, because the tools' own readers say the status is not the whole answer - `export_link`
    exists precisely because "202" and "200" cannot be told apart from the body's keys alone.
    """
    r = httpx.get(url, params={"api_key": key, **params}, timeout=120)
    try:
        body = r.json()
    except ValueError:
        raise SystemExit(f"{url} answered {r.status_code} with a non-JSON body: {r.text[:200]}")
    return r.status_code, body


def project(project_id: str, key: str) -> object:
    """`GET /{workspace}/{project}` - the class list, the counts and the version list.

    The route `generate_version.project` reads and `clean_v2` probes for `type`/`public`.
    """
    status, body = get_json(f"{API}/{WORKSPACE}/{project_id}", key)
    if status != 200:
        raise SystemExit(f"could not read {project_id}: HTTP {status} {str(body)[:200]}")
    return body


def version(project_id: str, number: int, key: str) -> object:
    """`GET /{workspace}/{project}/{n}` - the single version, which `--verify` judges."""
    status, body = get_json(f"{API}/{WORKSPACE}/{project_id}/{number}", key)
    if status != 200:
        raise SystemExit(f"could not read {project_id}/{number}: HTTP {status} {str(body)[:200]}")
    return body


def readable_version(project_id: str, declared: object, key: str) -> tuple[int, object] | None:
    """The newest version of `project_id` the single-version route will actually serve, or None.

    The number is *read* rather than taken from the flag, and the reading is
    `generate_version.next_version`'s - the project's own `versions` list, highest first - because
    that list is not a list of readable versions. This project's only entry is `.../snc-grocery/1`,
    created empty on 2026-09-21 and moved to the Trash, and `GET .../1` answers 404 "does not exist
    or cannot be loaded due to missing permissions" - so a recorder that trusted the list would
    have recorded nothing here and reported a successful run, which is the first thing this tool
    did. Each refused number is printed instead: the hole is a fact about the project, and choosing
    a version quietly is how a recording ends up being of something other than what was asked for.
    """
    start = generate_version.next_version(declared if isinstance(declared, dict) else {}) or 1
    for number in range(start, 0, -1):
        status, body = get_json(f"{API}/{WORKSPACE}/{project_id}/{number}", key)
        if status == 200:
            return number, body
        error = body.get("error") if isinstance(body, dict) else body
        message = error.get("message") if isinstance(error, dict) else str(error)[:80]
        print(f"  {project_id} v{number}: HTTP {status} - {message}")
    return None


def search_page(project_id: str, key: str, limit: int, offset: int = 0) -> object:
    """One page of `POST /{workspace}/{project}/search`, with the fields the tools ask for.

    A POST because that is the route's verb, not because it writes: the body is a query (`limit`,
    `offset`, `fields`), the same one `label_progress.fetch_all` and `clean_v2._index_sweep` send.
    `limit` is this tool's own and deliberately small - the shape of a record is the fact being
    recorded, and a full page is a megabyte of filenames for no more evidence. `total` still comes
    back as the server reported it.
    """
    r = httpx.post(
        f"{API}/{WORKSPACE}/{project_id}/search",
        params={"api_key": key},
        json={"limit": limit, "offset": offset, "fields": ["name", "annotations", "tags", "split"]},
        timeout=180,
    )
    body = r.json()
    if "results" not in body:
        raise SystemExit(f"search failed: HTTP {r.status_code} {str(body)[:200]}")
    return body


def trash(key: str) -> object:
    """`GET /{workspace}/trash` - what `clean_v2._trashed_versions` reads.

    Worth recording for one claim the code makes from it: a trashed version still holds its number,
    which is why this project's next generation is 2 rather than 1.
    """
    _status, body = get_json(f"{API}/{WORKSPACE}/trash", key)
    return body


def export_bodies(project_id: str, number: int, key: str, fmt: str, sleep, timeout_s: int = 900) -> list[tuple[str, dict]]:
    """The export route's two answers, as `train_model.download_export` receives them.

    A loop rather than one request because the route has two *shapes* and both are load-bearing:
    202 with `progress` while it builds, 200 with `export.link` when it is ready - which is the
    reason `export_link` reads the body rather than the status code. The first answer is recorded
    whatever it is; if it is already ready, there is no progress body to record and this says so
    rather than inventing one.

    `nocache=true` is what `download_export` sends, so it is what the recording carries: it asks for
    a fresh export rather than whatever was cached, which is the shape a *first* download sees. The
    archive is not fetched - the link is the fact, and following it would download the dataset.
    """
    url = f"{API}/{WORKSPACE}/{project_id}/{number}/{fmt}"
    recorded: list[tuple[str, dict]] = []
    deadline = time.monotonic() + timeout_s
    while True:
        status, body = get_json(url, key, nocache="true")
        recorded.append((f"export-{project_id}-v{number}-{'link' if status == 200 else 'progress'}.json", {"status": status, "body": body}))
        if status == 200 and isinstance(body, dict) and isinstance(body.get("export"), dict):
            break
        if time.monotonic() > deadline:
            print(f"  (still building after {timeout_s}s - only the progress body was recorded)")
            break
        progress = body.get("progress") if isinstance(body, dict) else None
        print(f"    building{'' if progress is None else f' {float(progress) * 100:.0f}%'} ...")
        sleep(10)
    return recorded


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="snc-grocery", help="the v2 dataset project (default %(default)s)")
    ap.add_argument("--v1-project", default="scanncart-grocery", help="the v1 project (default %(default)s)")
    ap.add_argument(
        "--project-version",
        type=int,
        default=0,
        help="the version to record of --project (default: the newest the route will serve)",
    )
    ap.add_argument(
        "--v1-version",
        type=int,
        default=0,
        help="the version to record of --v1-project (default: the newest the route will serve)",
    )
    ap.add_argument(
        "--export-project",
        default="",
        help=(
            "which project to record the export answer from (default --v1-project: the export route "
            "needs a version the API will serve, and the project under construction may have none)"
        ),
    )
    ap.add_argument(
        "--export-version", type=int, default=0, help="the version to export (default: newest readable)"
    )
    ap.add_argument("--format", default=EXPORT_FORMAT)
    ap.add_argument(
        "--limit",
        type=int,
        default=100,
        help=(
            "search results to record (default %(default)s): a page is the unit the paginating readers "
            "work in, and the labelled/null/unlabelled split only shows up across a few dozen of them"
        ),
    )
    ap.add_argument("--out", default=str(FIXTURES))
    ap.add_argument("--sleep", type=float, default=10.0, help="seconds between export polls")
    ap.add_argument("--no-export", action="store_true", help="skip the export request entirely")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out).expanduser()
    key = load_key(args.project)

    print(f"recording live {WORKSPACE} bodies -> {out}")
    print()

    project_body = project(args.project, key)
    write_fixture(f"project-{args.project}.json", project_body, out)
    v1_body = project(args.v1_project, key)
    write_fixture(f"project-{args.v1_project}.json", v1_body, out)

    # Which version to record: the newest the route will *serve*, which is not the same as the
    # newest the project lists - see `readable_version`.
    wanted: dict[str, tuple[int, object]] = {}
    for project_id, declared, asked in (
        (args.project, project_body, args.project_version),
        (args.v1_project, v1_body, args.v1_version),
    ):
        if asked:
            wanted[project_id] = (asked, version(project_id, asked, key))
            continue
        found = readable_version(project_id, declared, key)
        if found is None:
            print(f"  {project_id}: no version the route will serve - version fixture skipped")
            continue
        wanted[project_id] = found
        print(f"  {project_id}: recording v{found[0]}")
    for project_id, (number, body) in wanted.items():
        write_fixture(f"version-{project_id}-{number}.json", body, out)

    write_fixture(f"trash-{WORKSPACE}.json", trash(key), out)
    first_page = search_page(args.project, key, args.limit)
    write_fixture(f"search-{args.project}-page.json", first_page, out)

    # A second page, aimed at the end of the index, and not out of thoroughness: `state_of` splits
    # an image's three states on the *type* of its `annotations` field - a dict with `count: 0` is a
    # deliberate null (a hard negative) where a *list* means no annotation record at all - and a
    # page of labeled frames carries neither half of that. The background batches are the last thing
    # uploaded to this project, so the last page is where the nulls are; `state_of`'s own docstring
    # calls the misreading "silently re-open every hard negative", which is the sentence this
    # recording is here to make checkable. Skipped when the first page already is the last one.
    total = first_page.get("total") if isinstance(first_page, dict) else None
    if isinstance(total, int) and total > args.limit:
        last_page = search_page(args.project, key, args.limit, offset=total - args.limit)
        write_fixture(f"search-{args.project}-last-page.json", last_page, out)

    if args.no_export:
        print()
        print("export: skipped (--no-export)")
        return 0

    # The export needs a version that exists, and the project under construction is the one most
    # likely not to have one: `snc-grocery`'s only version is the empty trashed v1. So the default
    # is the other project, and the answer recorded is the route's rather than a project's - which
    # is the point, since every project answers this route in the same two shapes.
    export_project = args.export_project or args.v1_project
    export_version = args.export_version
    if not export_version:
        if export_project in wanted:
            export_version = wanted[export_project][0]
        else:
            found = readable_version(export_project, project(export_project, key), key)
            if found is None:
                print()
                print(f"export: skipped - {export_project} has no version the route will serve")
                return 0
            export_version = found[0]

    print()
    print(
        f"asking for {export_project} v{export_version}'s export as {args.format} "
        "(what `--download` asks for; no zip is fetched)"
    )
    for name, recorded in export_bodies(
        export_project, export_version, key, args.format, sleep=time.sleep, timeout_s=900
    ):
        write_fixture(name, recorded["body"], out)

    print()
    print(f"recorded into {out} - `tests/test_roboflow_shapes.py` reads these.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
