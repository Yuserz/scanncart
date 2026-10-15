# Recorded Roboflow bodies

Live answers from `https://api.roboflow.com`, taken once and committed so the dataset tools'
assumptions about them are **checked by `sidecar/tests/test_roboflow_shapes.py`** rather than stated
in a comment. Every fake in this suite was written by the person who wrote the reader it feeds, so a
fake agrees with its reader by construction and would keep agreeing after the route changed; these
are the opposite, which is the half a fake cannot cover.

## Taking them again

```bash
cd sidecar && ./.venv/Scripts/python.exe tools/record_roboflow_fixtures.py
```

Reads only: every route is a GET except the project search, which is a POST because that is how the
index is queried (the same request `label_progress.fetch_all` sends) and which writes nothing. No
version is generated, nothing is uploaded, deleted or annotated, and the export archive is never
downloaded — the fixture is the JSON body. The API key comes from the workspace `.env`
(`label_classes.load_key`), and `--help` lists the flags (`--project`, `--limit`, `--no-export`,
`--out`).

## The files

| File | Route | Read by |
| --- | --- | --- |
| `project-snc-grocery.json` | `GET /{workspace}/{project}` | `generate_version.project`, `label_classes.get_project`, `clean_v2`'s upload preflight, `next_version` |
| `project-scanncart-grocery.json` | the same, for v1's project | the same readers, so a second real class list |
| `version-scanncart-grocery-1.json` | `GET /{workspace}/{project}/{n}` | `generate_version.version` / `--verify`, `compare_preprocessing` |
| `export-scanncart-grocery-v1-link.json` | `GET /{workspace}/{project}/{n}/{format}` | `train_model.export_link`/`export_progress` |
| `search-snc-grocery-page.json` | `POST /{workspace}/{project}/search`, offset 0 | `label_progress.state_of`, `fetch_all`'s pagination |
| `search-snc-grocery-last-page.json` | the same, offset `total - limit` | the same, and it is the page that carries the **null** shape |
| `trash-yusri-caloyloy.json` | `GET /{workspace}/trash` | `clean_v2._trashed_versions` |

Redacted on the way in, and reported by the recorder when it happens: an export `link` is a
*signed* URL (its query is replaced, host and path kept) and the Trash body names the account that
deleted a version (`deletedByName`, an address, whose **key** stays because it is part of the shape).
`test_roboflow_shapes.py` asserts no file carries a credential or an address, so a re-recording that
kept one fails rather than being committed.

## Two things these recordings settled that prose had wrong

- **`snc-grocery` has no version.** Its only entry is the empty `1` of 2026-09-21, moved to the
  Trash — the single-version route answers 404 for it, which is why the recorder reads the version
  it records rather than trusting the project's list. So the next generation this project produces
  is **version 2**, and the v2 weights on disk were trained from the local merge, not from an
  export: their records carry an empty `source`.
- **The project route's class values are annotation counts, not positions.** Palmolive sits at 0,
  `century_tuna` at 150, `555 sardines` at 79 — so ranking them puts the class with the fewest boxes
  first, which is the reading `class_order` was rewritten to refuse.

## What a recording cannot carry

These are recordings, not live reads: the counts, the class list and the version numbers are what
they were on the day, so a test that pinned one would fail on the next upload. The tests pin the
*shape* and the readings derived from it.

Two branches are deliberately not recorded, and the tests name both rather than leaving them implied:
the export route's **202 progress** body (this project's export was already cached, so it answered
200 with the link on the first request — `test_train_model.py`'s polling fakes cover that branch, and
a format this project has never exported would reach it again), and an image with **no annotation
record at all** (`annotations` absent or a list, where a deliberate null is a dict with `count: 0` —
`test_label_progress.py` builds those records by hand).
