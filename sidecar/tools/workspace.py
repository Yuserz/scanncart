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

    workspace.DEFAULT_OUT    # <workspace>/cleaned-v2
    workspace.ENV_PATH       # <workspace>/.env

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
