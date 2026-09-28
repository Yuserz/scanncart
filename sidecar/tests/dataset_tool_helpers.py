"""Fixtures shared by the dataset-tools test modules, after one file became several.

Split out of `test_dataset_tools.py` when it was cut along its section banners: these are the
hand-written builders and fakes more than one of the new modules calls - an export, a merged set,
a finished run, the fake ultralytics metrics. A fixture only one module uses stays beside the
tests that use it.

Not a `conftest.py`: these are plain functions and classes tests call, not fixtures, and only the
modules that need them import them.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import build_dataset
import clean_v2
import generations
import plan_split
import workspace

V2 = generations.V2


def _batch(name: str, cls: str, distance: str, n: int, session: str = "s1") -> plan_split.Batch:
    return plan_split.Batch(
        name, cls, distance, session, [f"{name}_{i}.jpg" for i in range(n)]
    )


def _entries(batches: list[plan_split.Batch]) -> list[dict]:
    return [
        {"new_name": img, "class": b.cls, "distance": b.distance, "batch": b.name, "tags": []}
        for b in batches
        for img in b.images
    ]


def _two_session_batches() -> list[plan_split.Batch]:
    """Two sessions over the same two cells - the shape Tier C shoots for."""
    return [
        _batch("milo_close", "milo", "close", 40, "s1"),
        _batch("milo_far", "milo", "far", 40, "s1"),
        _batch("milo_close_s3", "milo", "close", 20, "s3"),
        _batch("milo_far_s3", "milo", "far", 20, "s3"),
    ]


def _staged_local(root: Path, frames: int = 3, slug: str = "milo", distance: str = "mid") -> Path:
    """A staged set plus an annotation tree, in the shapes the pipeline actually writes."""
    out = root / "cleaned-v2"
    (out / slug).mkdir(parents=True)
    entries = []
    for i in range(1, frames + 1):
        name = f"{slug}_{i:04d}.jpg"
        (out / slug / name).write_bytes(b"jpeg")
        entries.append(
            {"new_name": name, "class": slug, "distance": distance, "session": "s2", "batch": f"{slug}_{distance}"}
        )
    (out / "manifest.json").write_text(json.dumps(entries), encoding="utf-8")
    return out


def _local_args(out: Path, **over):
    base = {
        "out": str(out),
        "project": "snc-grocery",
        "source": "local",
        "annotations": "",
        "extras": [],
        "json": False,
    }
    base.update(over)
    import argparse

    return argparse.Namespace(**base)


def _frame(path: Path) -> None:
    """A real, decodable JPEG with content on its edges.

    Two readers need that and neither can be told otherwise: `check_export` samples frames to read
    the geometry (a solid colour is a padded frame), and the doctor fingerprints every train/valid
    frame for near-duplicates - a fixture of three bytes makes that scan unreadable, which the
    doctor now refuses rather than passing off as "no duplicates".
    """
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.frombuffer(os.urandom(64 * 64 * 3), dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(pixels).save(path, quality=60)


def _label_file(directory: Path, stem: str) -> None:
    """One box row. An *absent* label file is a finding to the doctor (the loader would train the
    frame as background, which is a decision nobody recorded), and a real export has a file for
    every image - measured on v1's own: 1,265/1,265 train, 223/223 valid, 327/327 test.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def _fake_export(
    root: Path,
    *,
    names: list[str] | None = None,
    splits: tuple[str, ...] = ("train", "valid", "test"),
    per_split: int = 3,
    nested: bool = False,
) -> Path:
    """A minimal stand-in for an unzipped YOLOv11 PyTorch export.

    `nested` reproduces the shape Roboflow's zip sometimes extracts into, where the
    split folders sit one level down - the reason `find_split_dirs` accepts both.

    Frames and labels both, because the gate on a training run reads them: an export carries a label
    file per image, and one that does not is a finding rather than a shortcut.
    """
    import yaml

    base = root / "export" if nested else root
    for split in splits:
        images = base / split / "images"
        images.mkdir(parents=True)
        for i in range(per_split):
            _frame(images / f"{split}_{i}.jpg")
            _label_file(base / split / "labels", f"{split}_{i}")
    (base / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names or []),
                # The generation's own order by default, not a sorted list: `check_export` compares
                # class names by membership, so any order used to do - but a label row indexes this
                # list, the doctor checks the order against the generation, and v1 and v2 declare
                # the same seven names in different orders.
                "names": names if names is not None else list(V2.classes),
            }
        ),
        encoding="utf-8",
    )
    return root


_RESULTS_CSV = """epoch, time, train/box_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
1, 12.0, 1.20, 0.700, 0.650, 0.710, 0.480
2, 24.0, 0.90, 0.870, 0.860, 0.890, 0.630
3, 36.0, 0.80, 0.890, 0.880, 0.910, 0.620
"""


class _FakeBox:
    """The subset of ultralytics' `Metric` the recall report reads.

    `r` is positional against `ap_class_index` in the real object (metrics.py builds both
    from `unique_classes`, which is derived from the split's *labels*), and that is the
    detail worth reproducing here exactly.
    """

    def __init__(self, index, recall, map50=0.0, map_=0.0, mp=0.0, mr=0.0):
        self.ap_class_index = index
        self.r = recall
        self.map50 = map50
        self.map = map_
        self.mp = mp
        self.mr = mr


class _FakeMetrics:
    def __init__(self, names, box, counts=None):
        self.names = names
        self.box = box
        self.nt_per_class = counts if counts is not None else [10] * len(names)


_NAMES = {0: "bear-brand", 1: "century-tuna", 2: "lucky-me", 3: "milo"}


def _finished_run(root: Path, name: str, *, mtime: float | None = None) -> Path:
    run = root / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"weights")
    (run / "results.csv").write_text(_RESULTS_CSV, encoding="utf-8")
    if mtime is not None:
        import os

        os.utime(run, (mtime, mtime))
    return run


def _manifest_with_distances(tmp_path: Path, pairs: list[tuple[str, str]]) -> Path:
    """A manifest in the shape `clean_v2.py` writes, carrying only what a distance needs.

    Written field-for-field as the tool writes it (`new_name` and `distance`), because the
    reader is filename-keyed and a fixture with invented keys would agree with itself and
    nothing else.
    """
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            [{"class": "x", "distance": d, "new_name": n, "old_name": n} for n, d in pairs]
        ),
        encoding="utf-8",
    )
    return path


def _export_with_names(root: Path, per_split: dict[str, list[str]]) -> Path:
    """A minimal export whose images are *named*, so a manifest can be joined to them.

    Readable frames with a label each, for `_fake_export`'s reason: these fixtures feed runs that
    are now gated on the doctor.
    """
    import yaml

    for split, names in per_split.items():
        images = root / split / "images"
        images.mkdir(parents=True)
        for name in names:
            _frame(images / name)
            _label_file(root / split / "labels", Path(name).stem)
    # The real roster, not a short list: `check_export` refuses an export whose classes are
    # not the v2 eight, and that check is the whole reason a distance pass can trust the yaml
    # `distance_data_yaml` writes. Taken from the generation rather than sorted, because the
    # *order* is load-bearing now that a training run is gated on the doctor: a label row indexes
    # this list, and v1 and v2 declare the same seven names in different orders.
    names = list(V2.classes)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": len(names),
                "names": names,
            }
        ),
        encoding="utf-8",
    )
    return root


def _report_with_distances(
    root: Path, pairs: list[tuple[str, str]], machine_only: dict[str, int] | None = None
) -> Path:
    """A built set's own provenance, in the shape `build_dataset.py` writes it.

    The per-frame `distances` map is what the readers under test here join on - it sits in
    `merge_report.json` beside the `tags` the doctor checks drawings against. The other two fields
    are what makes the set trainable at all: `machine_only_by_split` and the stamp naming the
    `provenance.json` those counts were read over, because `train_model` asks the human pass's gate
    (and `accept_v2` its own) before a run - and a report that cannot say which decisions it was
    built from is refused rather than read as clean.

    Written by hand rather than by running a build, because what is under test is the *reader* - the
    build's own test pins the writer - but the stamp comes from `build_dataset.annotation_state`, so
    a fixture cannot disagree with the writer about its shape.
    """
    annotations = root.parent / "annotations-v2"
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / workspace.PROVENANCE_NAME).write_text(json.dumps({}), encoding="utf-8")
    (root / workspace.MERGE_REPORT_NAME).write_text(
        json.dumps(
            {
                "generation": V2.name,
                "distances": dict(pairs),
                "machine_only_by_split": {} if machine_only is None else dict(machine_only),
                "annotations": build_dataset.annotation_state(annotations),
            }
        ),
        encoding="utf-8",
    )
    return root


def _fake_generation(root: Path) -> generations.Generation:
    """A generation whose export lives in `tmp_path`, so the readers can be tested with no
    workspace on disk. Only `export_dir` and `classes` are ever consulted by the readers below."""
    return generations.Generation(
        name="fake", classes=("a", "b"), export_dir=root, manifest=None,
        resize_mode="stretch", roboflow_project="none",
    )

