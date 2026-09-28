#!/usr/bin/env python
"""Where the dataset tools read and write, kept apart from where they live.

The tools are tracked code (`sidecar/tools/`). What they operate on is not:

    cleaned-v2/         staged JPEGs, manifest, split plans, upload resume state (~2.8 GB)
    cleaned-negatives/  §2's hard-negative frames (checklist Tier C2b)
    v1-source-zips/     v1's six raw per-class capture archives (1,501 images, 3.9 GB)
    .env                ROBOFLOW_API_KEY / ROBOFLOW_PROJECT_ID

v1's local set (`cleaned/`, 5 GB of images carrying no labels) was deleted on 2026-09-22 to
reclaim space on a drive that was 87% full. Nothing referenced it, and it is recoverable two
ways: v1's *annotations* live in the Roboflow project `scanncart-grocery` (1,516 images, one
generated version, so an export works), and the six raw source zips it was built from are
plain files in `v1-source-zips/`.

Those zips were inside `.git` for ~5 weeks, held by the abandoned commit `141c7ad` and then by a
tag pinned to it, because a reflog entry alone expires in ~90 days. A tag turned out to be a poor
home for 3.9 GB of incompressible JPEGs - it was the entire difference between a 3.9 GB and a
60 MB `.git`, and every clone would have paid it. So they were verified against that commit
(each file's blob hash, then a full CRC read of every archive) and pulled out to the workspace;
`gc --prune=now` then dropped the pack from 3.83 GiB to 58 MiB.

`v1-source-zips/README.md` carries the inventory (class, file count, size) and how to delete them
if route 1 is enough for you - at 3.9 GB they are the largest single reclaim left on the disk.

None of that belongs in git, and gigabytes of JPEGs do not belong next to the
sidecar's source either - so the two are separated. Code lives in the sidecar tree;
data lives in the sidecar's data directory, `sidecar/data/datasets/`, which is
gitignored and already where the hard-negative capture lives. Point it elsewhere
with SCANNCART_DATASET_ROOT:

    SCANNCART_DATASET_ROOT=D:/scanncart-datasets python sidecar/tools/clean_v2.py sanity

    import workspace

    workspace.DEFAULT_OUT      # <workspace>/cleaned-v2
    workspace.default_extras() # the staged sets beside it (the hard negatives)
    workspace.default_extras(v2.parent)  # ...beside a set staged somewhere else
    workspace.resolve_extras(v2.parent, args.extras,
                             include_defaults=not args.no_extras)
                               # ...those, plus any named on the command line - a union, because
                               # the default is a second set rather than a fallback value
    workspace.ENV_PATH         # <workspace>/.env
    workspace.MANIFEST_NAME    # the artifact names more than one tool reads or writes

Note this `.env` is the *tools'* credentials file. The sidecar reads its own from
`sidecar/.env` (see app/credentials.py) - they are separate files for separate
programs, and neither is tracked.
"""

from __future__ import annotations

import os
from pathlib import Path

# sidecar/tools/workspace.py -> sidecar/tools -> sidecar -> repo root
HERE = Path(__file__).resolve().parent
SIDECAR_ROOT = HERE.parent
REPO_ROOT = SIDECAR_ROOT.parent

# The dataset workspace: inside the sidecar's gitignored data dir, so code and data
# both live under `sidecar/` while only the code is tracked.
WORKSPACE = Path(
    os.environ.get("SCANNCART_DATASET_ROOT", SIDECAR_ROOT / "data" / "datasets")
).expanduser()
DEFAULT_OUT = WORKSPACE / "cleaned-v2"
ENV_PATH = WORKSPACE / ".env"

# The names of the artifacts more than one tool reads or writes. Spelled once here and imported by
# every writer and reader: a name kept in two places drifts - rename it in the writer and the
# readers still look for the old file, failing only when the tool is run.
MANIFEST_NAME = "manifest.json"  # a staged set's frame list, written by `clean_v2.py clean`
SPLITS_NAME = "splits.json"  # frame -> split, frozen from a chosen plan
MERGE_REPORT_NAME = "merge_report.json"  # a built set's provenance, written by `build_dataset.py`
DATA_YAML_NAME = "data.yaml"  # the dataset declaration ultralytics reads
SCANNCART_DATA_YAML_NAME = "data.scanncart.yaml"  # the normalized one `train_model` writes
PROVENANCE_NAME = "provenance.json"  # who drew each box, in the annotator's tree

# Staged sets that are part of v2 but do not live inside `cleaned-v2/`.
#
# The hard negatives are the reason this exists: `clean_v2 clean --negatives <dir> --out
# <workspace>/cleaned-negatives` stages them into a set of their own, with their own manifest -
# which means every reader that looks at one directory cannot see them. They are the frames that
# teach the model what the products are *not* (25 of 50 empty counters are detected without them),
# and the annotator, the snapshot and the merge would all quietly produce a dataset without them:
# exactly the v1 set this project is trying to improve on, with no error anywhere. Those three
# readers share this list rather than each guessing, so they cannot disagree about what the v2 set
# contains.
# Directory *names*, not paths: `default_extras` resolves them beside whichever staged set it is
# asked about, so landing here as an absolute path would put the workspace anchor back.
DEFAULT_EXTRAS: tuple[str, ...] = ("cleaned-negatives",)


def default_extras(beside: Path | None = None) -> list[Path]:
    """The sets actually staged *beside* `beside` - by default, beside the workspace's own set.

    Filtered by manifest rather than by directory: an empty or half-staged folder is not a set,
    and a reader that accepted one would report frames it cannot open.

    `beside` is the staged set these sit next to (`cleaned-v2`'s parent, i.e. the workspace),
    because a reader resolves `--annotations` off its `--out` the same way. Anchoring on a fixed
    workspace path instead would make `--out`/`--v2` half-honoured: a run against a set staged
    elsewhere - a second attempt, a test's temporary copy - would still pull *this* workspace's
    hard negatives into its worklist, its snapshot and its build, reporting 50 frames the operator
    does not have and dropping the ones that set actually means.
    """
    root = WORKSPACE if beside is None else Path(beside).expanduser()
    return [root / name for name in DEFAULT_EXTRAS if (root / name / MANIFEST_NAME).is_file()]


def resolve_extras(
    beside: Path, explicit: list[str] | tuple[str, ...] = (), include_defaults: bool = True
) -> list[Path]:
    """The staged sets a run has to read: the defaults beside `beside` (the directory the sets sit
    in - `--out`'s parent), **plus** whatever was named.

    A union, because the default here is not a fallback value - it is a second *set*. The hard
    negatives are staged on their own with their own manifest, they are the frames that teach the
    model what the products are not, and a reader that dropped them because somebody named one more
    directory would produce a worklist, a snapshot or a merge that is quietly missing 50 frames.
    That is how `--extras` used to behave in the three readers that take one (`label_progress`,
    `build_dataset`, the annotator): an explicit value *replaced* the default. It is also the
    behaviour `plan_split.include_dirs` was already written to avoid for `--include`, so this is
    that rule moved to where the default list lives, and `include_dirs` now calls it.

    `include_defaults=False` (`--no-extras` on those tools) is how a run that means *only* the sets
    it names says so: replacing the default is a decision rather than something that happens to you
    for naming a second directory. Deduped, defaults first, so a run's printout is stable.

    The defaults are manifest-filtered (`default_extras`'s rule - a half-staged folder is not a
    set); the named directories are passed through as given, because the caller asked for them and
    every reader already skips one that holds no manifest.
    """
    wanted: list[Path] = list(default_extras(beside)) if include_defaults else []
    for extra in explicit:
        path = Path(extra).expanduser()
        if path not in wanted:
            wanted.append(path)
    return wanted
