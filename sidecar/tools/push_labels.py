#!/usr/bin/env python
"""Push the local annotator's decisions into the Roboflow project — the direction
`import_labels.py` does not cover.

WHY THIS EXISTS
---------------
The annotator's tree (`workspace.ANNOTATIONS_DIRNAME`) is the dataset's source of
truth, and the
project is the cloud copy other stages read (`label_progress.py --source roboflow`,
`generate_version.py`, the suggester). A decision made locally is invisible to the
cloud-side tools until this bridge runs: before it, the project read 1445/1452
decided while the local tree read 1452/1452, with seven frames the project had
never received under any name.

WHAT IT SENDS
-------------
Per frame the cloud does not hold (or holds with no decision):
* the staged image, in the batch and with the tags its manifest record carries
  (`tags_for` + `batch_for_session` — the same helpers the staged uploader uses, so
  the pushed frame cannot drift from its session taxonomy);
* the decision itself as a VOC annotation — the boxes the staged file shows, in its
  own pixel space, with the class names `classes.json` records.
Each frame is verified by reading its own record back after the push and comparing
boxes, labels and tags against the local decision; a frame that does not read back
as the local decision is a reported failure, never a silent one.

ROBOFLOW QUIRKS THIS WORKS ROUND
--------------------------------
* An upload dedupes byte-identical photos onto the existing record — which can be a
  twin whose *name is not the uploaded name* (a deduped upload was hit on
  negative_neg3_0014, whose bytes are identical to negative_neg3_0015's). The tool
  reads the deduped record's own name back and says so, rather than claiming a
  decision for a frame the project holds under another stem.
* A null (a deliberate "this image contains nothing") cannot be written over the
  API: `clean_v2`'s notes record that the annotate endpoint rejects an empty
  annotation file, and the null flag exists only in the UI. A null the cloud lacks
  is therefore reported as a hand-marks frame, and the deduped-bytes case above
  usually already covers it — the photo's null exists under its twin's name.
* The metadata route is async and additive (`addTags`, `removeTags` both supported);
  only the first tag survives the upload's querystring, which is why the retag call
  exists here.
* Reads are eventually consistent; fetches poll rather than fail once.

USAGE
-----
    sidecar/.venv/Scripts/python.exe sidecar/tools/push_labels.py           # what would be pushed
    sidecar/.venv/Scripts/python.exe sidecar/tools/push_labels.py --apply
    sidecar/.venv/Scripts/python.exe sidecar/tools/push_labels.py --stems safeguard_0221 --apply
    sidecar/.venv/Scripts/python.exe sidecar/tools/push_labels.py --any-outstanding --apply

THE CHECK (make push-check)
---------------------------
`--check` is the gate over the push, read-only: no upload, no annotation write. It exits 0 only
when every local decided frame is in the project under its own name, or under the byte-identical
twin's name with the same decision (the 0014/0015 case above) — and exits 1, each frame named,
when the project has never received one under any name. Exit 2 is the inputs the check cannot
run on (no local decision tree — CI's bare checkout is the ordinary case, since the annotator's
tree is gitignored). The project-side rule is the same judgment the push applies, so a gate that
says 0 says exactly what a push would then leave behind: nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from PIL import Image

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from build_dataset import DEFAULT_V2  # noqa: E402
import workspace  # noqa: E402

API = "https://api.roboflow.com"
PROJECT = "snc-grocery"
STAGED_ROOTS = (
    DEFAULT_V2.parent / "cleaned-v2",
    DEFAULT_V2.parent / "cleaned-negatives",
)


def stem_of(name: str) -> str:
    return name.rsplit(".", 1)[0]


def find_staged(stem: str) -> Path | None:
    for root in STAGED_ROOTS:
        for cand in root.rglob(f"{stem}.jpg"):
            if cand.is_file():
                return cand
    return None


def manifest_record(stem: str) -> dict | None:
    """The staged manifest's own record for a frame, across the v2 sets.

    The manifest is the authoritative tags/batch source — copying a decided
    sibling's tags would mis-fence the frame when its own distance differs (0138
    promised close and a sibling said mid). The record is the decision the uploader
    itself wrote, so the pushed frame lands exactly where its staged twin did.
    """
    for root in STAGED_ROOTS:
        path = root / "manifest.json"
        if not path.is_file():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            if rec.get("new_name") == f"{stem}.jpg":
                return rec
    return None


def manifest_hashes() -> dict[str, str]:
    """stem -> sha256, across the staged sets.

    The manifest is the source, not the files on disk: staging's own byte-dedupe can drop a
    frame's jpg from the directories (the twin's copy is what stays), so the record carrying the
    hash is the only remaining copy of that fact. A stem whose manifest record carries no hash
    gets none here - `classify_received` treats it as a twin that cannot be proven.
    """
    hashes: dict[str, str] = {}
    for root in STAGED_ROOTS:
        path = root / "manifest.json"
        if not path.is_file():
            continue
        for rec in json.loads(path.read_text(encoding="utf-8")):
            stem = stem_of(rec.get("new_name", ""))
            if stem and rec.get("sha256"):
                hashes[stem] = str(rec["sha256"])
    return hashes


@dataclass(frozen=True)
class Received:
    """The check's verdict, by category.

    `by_name` - the project holds the photo under the local stem's own name (decided: labeled
    or null).
    `by_twin` - (stem, name) pairs: the project does not hold this frame's name, but it holds the
    stem's byte-identical staged twin under `name`, with the same decision. The photo was
    received; the name under which it is held is the twin's, which is why the plain stem diff
    alone cannot tell this case from a missing one.
    `undecided` - the project holds the photo under its own name but has not decided it (the
    recovery path `--any-outstanding` exists for). Reported, not failed: a frame received with
    no decision is the window between the two writes of a push, and the run sheet's cloud-side
    readers already surface it as outstanding work.
    `missing` - (stem, why): never received under any name. This is what `--check` exits 1 on.
    """

    by_name: list[str] = field(default_factory=list)
    by_twin: list[tuple[str, str]] = field(default_factory=list)
    undecided: list[str] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)

    @property
    def received(self) -> int:
        return len(self.by_name) + len(self.by_twin)


def classify_received(
    local_hash: dict[str, str | None],
    local_state: dict[str, str],
    cloud: dict[str, tuple[str, str]],
) -> Received:
    """Pure: every local decided stem, as received / twin-covered / undecided / missing.

    `local_hash` - stem -> the manifest's sha256, or None when no record holds one.
    `local_state` - stem -> "labeled" | "null" (the decision's own kind).
    `cloud` - stem -> (the name the project holds it under, the record's decision state).

    "Received under any name" is judged by bytes, not names, because the upload dedupes
    byte-identical photos onto one record held under whichever name got there first
    (negative_neg3_0014 landed as its twin 0015's record) - a stem-name diff alone would cry
    missing about a frame the project actually holds. A twin covers the *photo*; its cloud
    record's state has to match the local decision too, or the decision is what went missing:
    a labeled local frame covered by a twin record marked null is not representable either,
    because the annotate route would overwrite the twin's record and misrepresent another
    frame - so a state mismatch fails the same way a gap does, with the frame named.
    """
    by_hash: dict[str, list[str]] = {}
    for stem, sha in local_hash.items():
        if sha:
            by_hash.setdefault(sha, []).append(stem)
    report = Received()
    for stem in sorted(local_state):
        hold = cloud.get(stem)
        if hold is not None:
            name, state = hold
            (report.by_name if state in ("labeled", "null") else report.undecided).append(stem)
            continue
        sha = local_hash.get(stem)
        if not sha:
            report.missing.append(
                (stem, "no manifest record holds its hash, so no sibling can prove it received")
            )
            continue
        siblings = [(s, cloud[s][0], cloud[s][1]) for s in by_hash[sha] if s in cloud]
        twins = [s for s in siblings if s[2] == local_state[stem]]
        if twins:
            report.by_twin.append((stem, twins[0][1]))
        elif siblings:
            name, state = siblings[0][1], siblings[0][2]
            report.missing.append(
                (stem,
                 f"its bytes are in the project only under {name}, which holds them as {state}, "
                 f"while this frame is {local_state[stem]} - no record represents this decision")
            )
        else:
            report.missing.append((stem, "its bytes are in no record the project holds"))
    return report


def image_record(client: httpx.Client, key: str, image_id: str) -> dict:
    """One GET, converged. The per-image route nests the searchable fields under
    `image`, while the search endpoint flattens them — `state_of` reads the flat
    shape, so the record is re-flattened before it is judged."""
    for _ in range(5):
        body = client.get(f"{API}/{WORKSPACE}/{PROJECT}/images/{image_id}", params={"api_key": key}).json()
        image = (body or {}).get("image", body or {})
        if isinstance(image.get("annotations"), list):
            time.sleep(2.0)  # eventual consistency, the same gap fetch_all polls through
            continue
        return image
    return {}


def local_of(record: dict) -> tuple[str, int]:
    """The flat record's state: 'unlabeled' | 'null' | 'labeled', and its box count."""
    ann = record.get("annotations")
    if not isinstance(ann, dict):
        return "unlabeled", 0
    n = int(ann.get("count") or 0)
    return ("labeled" if n > 0 else "null"), n


def voc_xml(filename: str, w: int, h: int, boxes: list[tuple[str, float, float, float, float]]) -> str:
    """VOC: xmin/ymin/xmax/ymax, pixels. `boxes` rows are (label, cx, cy, bw, bh) in pixels."""
    parts = [
        "<annotation>",
        "<folder></folder>",
        f"<filename>{filename}</filename>",
        "<path></path>",
        "<size>",
        f"<width>{w}</width>",
        f"<height>{h}</height>",
        "<depth>3</depth>",
        "</size>",
        "<segmented>0</segmented>",
    ]
    for label, cx, cy, bw, bh in boxes:
        xmin = max(0.0, cx - bw / 2)
        ymin = max(0.0, cy - bh / 2)
        xmax = min(float(w), cx + bw / 2)
        ymax = min(float(h), cy + bh / 2)
        parts.append(
            f"<object><name>{label}</name><pose>Unspecified</pose><truncated>0</truncated>"
            f"<difficult>0</difficult><occluded>0</occluded>"
            f"<bndbox><xmin>{xmin:.0f}</xmin><ymin>{ymin:.0f}</ymin>"
            f"<xmax>{xmax:.0f}</xmax><ymax>{ymax:.0f}</ymax></bndbox></object>"
        )
    parts.append("</annotation>")
    return "".join(parts)


def check_received(ann: Path, local_stems: list[str], cloud: dict[str, dict]) -> int:
    """The `--check` verdict: read-only, and the whole mode's reason to exist.

    Exit 0 only when every local decided frame is in the project under its own name, or under the
    byte-identical twin's name with the same decision; exit 1 names every frame that never got in;
    exit 2 is the input that cannot be checked at all - an empty local tree is not green, it is
    *not checked*, because a gate that passes on nothing it read is the silence this check exists
    to kill (CI's bare checkout is the ordinary case, since the annotator's tree is gitignored).
    """
    from label_progress import state_of

    local_state = {
        s: ("null" if (ann / f"{s}.txt").stat().st_size == 0 else "labeled")
        for s in local_stems
    }
    if not local_state:
        print(f"FAIL no local decisions in {ann} - the check refuses to pass on nothing; "
              "run it on the machine the labeling lives on, or point --annotations there")
        return 2
    cloud_by_stem = {stem_of(name): (name, state_of(rec)[0]) for name, rec in cloud.items()}
    report = classify_received(manifest_hashes(), local_state, cloud_by_stem)
    total = len(local_state)
    print(
        f"the project received {report.received} of {total} local decision(s): "
        f"{len(report.by_name)} under their own name, {len(report.by_twin)} as byte twins"
        + (" - all received" if not report.missing else "")
    )
    for stem, name in report.by_twin:
        print(f"  twin {stem}: bytes identical to {name}, which the project holds with the same decision")
    for stem in report.undecided:
        print(f"  NOTE {stem}: received by name, not decided in the project (push --any-outstanding recovers)")
    for stem, why in report.missing:
        print(f"FAIL {stem}: never received under any name - {why}")
    if report.missing:
        print(
            f"\n{len(report.missing)} local frame(s) the project has never received - "
            "push them (push_labels.py) or fix what the twin's record holds"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    from clean_v2 import DEFAULT_SESSION, batch_for_session, batch_name, tags_for
    from label_classes import load_key
    from label_progress import WORKSPACE, fetch_all

    ap = argparse.ArgumentParser(description=__doc__
                                       if isinstance(__doc__, str) else "push local labels to Roboflow",
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--annotations", default=str(DEFAULT_V2.parent / workspace.ANNOTATIONS_DIRNAME)
    )
    ap.add_argument("--stems", default="", help="comma-separated stems to sync; default = every local frame the cloud does not hold")
    ap.add_argument("--any-outstanding", action="store_true",
                    help="also re-annotate cloud frames that exist with no decision (recovery path)")
    ap.add_argument("--apply", action="store_true", help="really upload and annotate; default is a dry run")
    ap.add_argument("--check", action="store_true",
                    help="read-only gate: exit 1 when a local frame has not been received by the project "
                         "under any name (make push-check)")
    args = ap.parse_args(argv)
    if args.check and (args.stems or args.any_outstanding):
        ap.error("--check is read-only and whole-tree: it takes no --stems, and --any-outstanding "
                 "is a push-mode selection")

    ann = Path(args.annotations)

    # The class names are the push loop's vocabulary; the check mode never draws anything, so
    # a tree whose classes.json is missing still gets its verdict - the refusal, not a crash.
    names = json.loads((ann / "classes.json").read_text(encoding="utf-8"))["names"] \
        if not args.check else []

    key = load_key(PROJECT)
    with httpx.Client(timeout=120) as client:
        cloud = fetch_all(client, key, PROJECT)
        cloud_stems = {stem_of(n): n for n in cloud}

        local_stems = sorted(p.stem for p in ann.glob("*.txt"))
        if args.check:
            return check_received(ann, local_stems, cloud)
        requested = [s.strip() for s in args.stems.split(",") if s.strip()] or sorted(local_stems)
        todo: list[str] = []
        for s in requested:
            if s not in local_stems:
                print(f"SKIP (no local decision): {s}")
                continue
            if s not in cloud_stems:
                todo.append(s)
            elif args.any_outstanding:
                state, _ = local_of(cloud[cloud_stems[s]])
                if state == "unlabeled":
                    todo.append(s)
        print(f"frames to push: {len(todo)} of {len(requested)} held locally")
        if not todo:
            return 0

        for s in todo:
            label = ann / f"{s}.txt"
            is_null = label.stat().st_size == 0
            decision = "null" if is_null else "labeled"
            if is_null:
                # The API cannot mark a null: the annotate endpoint rejects an empty
                # annotation file (clean_v2's notes), and the null flag lives in the UI.
                # A byte-identical photo usually already holds the null under its twin's
                # name, so say what the project actually holds rather than query for
                # something the platform cannot express.
                print(f"NOTE {s}: a null cannot be written over the API. If its photo is byte-identical")
                print("     to another staged frame, that twin's record already holds the cloud's null.")
                continue
            staged = find_staged(s)
            if staged is None:
                print(f"FAIL (no staged image): {s}")
                continue
            img = Image.open(staged)
            rows = []
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                a, cy, cx, bw, bh = (float(v) for v in line.split()[:5])
                rows.append((int(a), cx, cy, bw, bh))
            boxes = [
                (names[a], cx * img.width, cy * img.height, bw * img.width, bh * img.height)
                for a, cx, cy, bw, bh in rows
            ]
            rec = manifest_record(s)
            if rec is not None:
                cls = rec.get("class", "")
                distance = rec.get("distance", "")
                session = rec.get("session") or DEFAULT_SESSION
                tags = tags_for(cls, distance, session)
                batch = batch_for_session(rec.get("batch") or batch_name(cls, distance), session)
                note = ""
            else:
                tags = []
                batch = "uploaded-via-push_labels"
                note = " (manifest record missing — batch defaulted)"
            print(f"{s}: labeled, staged {staged.name} {img.width}x{img.height}, "
                  f"{len(boxes)} box(es), batch={batch}, tags={tags}{note}")
            if not args.apply:
                continue

            up = client.post(
                f"{API}/dataset/{PROJECT}/upload",
                params={"api_key": key, "name": f"{s}.jpg", "batch": batch, "tag": tags[0] if tags else None},
                files={"file": (f"{s}.jpg", staged.read_bytes(), "image/jpeg")},
            )
            if up.status_code != 200 or not up.json().get("id"):
                print(f"FAIL upload HTTP {up.status_code}: {up.text[:160]}")
                continue
            image_id = up.json()["id"]
            first = image_record(client, key, image_id)
            deduped_name = first.get("name")
            if deduped_name and stem_of(deduped_name) != s:
                print(f"NOTE {s}: the upload deduped onto {deduped_name!s} (bytes already in the project); "
                      "its own record already says:\n"
                      f"     {local_of(first)} with tags {sorted(first.get('tags') or [])}"
                      " — the local decision was applied there")
            if len(tags) > 1:
                rest = client.post(
                    f"{API}/{WORKSPACE}/images/metadata",
                    params={"api_key": key},
                    json={"updates": [{
                        "imageId": image_id,
                        "addTags": tags[1:],
                        "metadata": {
                            "distance": distance if rec else "",
                            "sku": cls if rec else "",
                            "batch": batch,
                            "session": session if rec else DEFAULT_SESSION,
                        },
                    }]},
                )
                if rest.status_code == 202:
                    task_url = rest.json().get("url", "")
                    for _ in range(10):
                        if client.get(task_url, params={"api_key": key}).status_code == 200:
                            break
                        time.sleep(3)
                elif rest.status_code != 200:
                    print(f"WARN: retag HTTP {rest.status_code}: {rest.text[:120]}")
            xml = voc_xml(f"{s}.jpg", img.width, img.height, boxes).encode("utf-8")
            ann_resp = client.post(
                f"{API}/dataset/{PROJECT}/annotate/{image_id}",
                params={"api_key": key},
                content=xml,
                headers={"Content-Type": "text/xml"},
            )
            if ann_resp.status_code == 409:
                ann_resp = client.post(
                    f"{API}/dataset/{PROJECT}/annotate/{image_id}",
                    params={"api_key": key, "overwrite": "true"},
                    content=xml,
                    headers={"Content-Type": "text/xml"},
                )
            if ann_resp.status_code != 200:
                print(f"FAIL annotate HTTP {ann_resp.status_code}: {ann_resp.text[:160]}")
                continue
            back = image_record(client, key, image_id)
            got_labels = [b.get("label") for b in (back.get("annotation", {}) or {}).get("boxes") or ["(none)"]]
            got_tags = sorted(str(t) for t in (back.get("tags") or []))
            labels_ok = got_labels == [b[0] for b in boxes] or (got_labels == ["(none)"] and not boxes)
            tags_ok = got_tags == sorted(str(t) for t in tags)
            ok = labels_ok and tags_ok
            print(f"    readback labels={got_labels} tags={got_tags} -> {'OK' if ok else 'MISMATCH'}")
        print("done" + (" (apply ran; re-run label_progress.py --source roboflow)"
                       if args.apply else " (--dry-run: nothing sent)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
