"""Roboflow's answer shapes, pinned against recordings instead of against fakes.

Every fake in this suite was written by the person who wrote the reader it feeds, so it agrees with
that reader by construction - and it would go on agreeing after the route changed. That is fine for
a branch (a 202 progress body, a corrupt record) and useless for a *shape*: `class_order`'s whole
docstring used to be the sentence "the live project route answers `name -> annotation count`", which
was true, unverifiable, and one API change away from being a lie that nothing could catch.

So the bodies are recorded once, from the live API, by
`sidecar/tools/record_roboflow_fixtures.py` and committed under `__fixtures__/roboflow/` - whose
README says what each file is and how to take them again. This module reads them and runs the real
readers over them: `generate_version.project`/`version` (through an httpx stub, so the unwrapping is
what is exercised), `label_classes.get_project`, `clean_v2._trashed_versions`, `class_order`,
`classes_with_distance`, `next_version`, `compare_preprocessing`, `train_model.export_link`/
`export_progress` and `label_progress.state_of`.

What is deliberately not asserted: any count as a fact about the project. These are recordings - the
class list, the image totals and the version numbers are what they were on the day - so a test that
pinned one would fail on the next upload. Pinned instead are the shape and the readings the tools
derive from it, each with the reason the reading exists. Where a recording cannot reach a branch
(the 202 export body, an image with no annotation record at all) the test says so and names the fake
that covers it, because a gap nothing mentions is the failure this module exists to prevent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import clean_v2
import generate_version
import generations
import label_classes
import label_progress
import train_model

FIXTURES = Path(__file__).resolve().parent / "__fixtures__" / "roboflow"
V2_PROJECT = "snc-grocery"
V1_PROJECT = "scanncart-grocery"

# Anything credential-shaped or personal. The recorder redacts both kinds on the way in and reports
# them; this is the other end of that, so a re-recording that kept a signed link or an address fails
# here rather than being committed. The pattern is deliberately a *scan* rather than an equality on
# the known fields: a new route recorded later could carry one under a key nobody thought of.
CREDENTIAL_IN_TEXT = re.compile(r"[?&](api_key|key|token|sig|signature|expires)=|X-Amz-")
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")


def fixture(name: str) -> dict:
    """One recorded body.

    Fails loudly on an absent or non-object file: a helper that answered `{}` would let every
    assertion below pass over nothing, which is the shape this whole module is here to avoid.
    """
    path = FIXTURES / name
    assert path.is_file(), f"no recording at {path} - run tools/record_roboflow_fixtures.py"
    body = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(body, dict), f"{name} is not an object - the recorder writes route bodies"
    return body


class _Response:
    """The two attributes every reader here touches: `status_code` and `.json()`."""

    def __init__(self, body: object, status_code: int = 200):
        self._body = body
        self.status_code = status_code

    def json(self) -> object:
        return self._body


class _Client:
    """A client that answers one recorded body, recording the URLs it was asked for.

    `label_classes.get_project` and `clean_v2._trashed_versions` both take a client and call `.get`,
    which is why they can be run against a recording without a network - and running them is the
    point: asserting that a fixture *has* a `project` key proves nothing about the reader that is
    supposed to unwrap it.
    """

    def __init__(self, body: object, status_code: int = 200):
        self.body, self.status_code = body, status_code
        self.calls: list[str] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(url)
        return _Response(self.body, self.status_code)


# --------------------------------------------------------------------------
# The project route
# --------------------------------------------------------------------------


def test_the_project_route_wraps_its_block_and_the_readers_unwrap_it(monkeypatch):
    """`{project, versions, workspace}`, and the three readers that depend on it being wrapped.

    `generate_version.project` returns the body whole because `next_version` reads the *outer*
    `versions`; `label_classes.get_project` returns the inner block because its callers want
    `classes`; `clean_v2`'s upload preflight reads `type`/`public` off the inner one too. Three
    readings of one shape, so the shape is recorded rather than remembered.
    """
    for name, project_id in ((f"project-{V2_PROJECT}.json", V2_PROJECT), ("project-scanncart-grocery.json", V1_PROJECT)):
        body = fixture(name)
        assert {"project", "versions", "workspace"} <= set(body), name
        assert isinstance(body["project"], dict) and isinstance(body["versions"], list)

        # The reader, over the recording: an httpx stub answers what the route answered.
        monkeypatch.setattr(generate_version, "httpx", _Client(body))
        assert generate_version.project(project_id, "k") == body

        # The other two readers, each over the same body.
        meta = label_classes.get_project(_Client(body), "k", project_id)
        assert meta["classes"] == body["project"]["classes"]
        assert meta["type"] == "object-detection"
        assert isinstance(meta["public"], bool)


def test_the_live_class_list_is_annotation_counts_so_no_order_is_readable():
    """The measurement `class_order` was rewritten for, on the live project rather than on a fake.

    The route's values are how many boxes each class has - `century_tuna` 150, `555 sardines` 79,
    `safeguard` 281 - and not positions. Ranking them put the class with the *fewest* annotations
    first, so the moment a project's names matched its roster the check compared the roster against
    an order assembled out of box counts and refused a project whose order had never been read.
    The gate is "the values are exactly `0..n-1`", and this recording is what says they are not.
    """
    for name in (f"project-{V2_PROJECT}.json", f"project-{V1_PROJECT}.json"):
        classes = fixture(name)["project"]["classes"]
        values = sorted(classes.values())

        assert generate_version.class_order(classes) is None
        assert values != list(range(len(classes))), (
            f"{name}: the route now numbers its classes as positions - `class_order`'s None is no "
            "longer the right answer and the order check is worth re-reading"
        )
        # And the reading it replaced was wrong on real data, not just in principle: ranked by
        # count, this project's first class is not the roster's first.
        ranked_first = min(classes, key=lambda key: classes[key])
        assert ranked_first != generations.V2.classes[0] or ranked_first != generations.V1.classes[0]


def test_the_live_class_lists_raise_no_distance_or_order_finding():
    """The two false alarms that must not fire on the names actually in the project.

    `classes_with_distance` on the live list: none of these names carries a distance word, and a
    finding here would refuse every generation from a project that is correct. `order_mismatch`:
    the recorded list is the roster's *eight* against a seven-name roster, which is a different set
    rather than a reordering - so it is `check_export`'s finding and not this one's, and returning
    an order here would send the operator to drag class rows around for a class that has to be
    deleted.
    """
    classes = fixture(f"project-{V2_PROJECT}.json")["project"]["classes"]

    assert generate_version.classes_with_distance(classes) == {}
    assert generate_version.order_mismatch(list(classes), generations.V2) is None


def test_the_live_project_declares_one_version_and_the_next_generation_is_two():
    """A trashed version keeps its number, which is why the next generation is 2 and not 1.

    The assertion is the *reading* rather than the number: `next_version` takes the highest version
    the project lists and adds one. What it is worth reading for is that this project has never
    generated a version at all - its only entry is the empty v1 of 2026-09-21, which was moved to
    the Trash, where the single-version route answers 404 for it (see the trash recording) - so the
    next generation takes number **2**, and `2` is what a `--download --version` line has to pass.
    Read rather than remembered, because "we deleted v1" does not make the next one v1.
    """
    body = fixture(f"project-{V2_PROJECT}.json")

    assert len(body["versions"]) == 1
    assert body["versions"][0]["id"].rsplit("/", 1)[-1] == "1"
    assert generate_version.next_version(body) == 2


# --------------------------------------------------------------------------
# The single-version route
# --------------------------------------------------------------------------


def test_the_single_version_route_nests_its_body_and_the_shipped_v1_version_matches_both_readers(monkeypatch):
    """Two claims the code makes about an existing version, checked against the version that is.

    `version()` unwraps `{"version": ...}` and says in a comment that the project route returns the
    same object inline instead - so the nesting is a fact worth a recording. And the recorded
    version is `scanncart-grocery/1`, the one whose 98.21 mAP the preprocessing block was frozen
    from: `compare_preprocessing` returns no difference and the augmentation is empty, which is what
    `--verify 1` prints as "matches". That sentence was asserted in prose ("pinned against the
    version that actually shipped") and is now read off the version.
    """
    body = fixture("version-scanncart-grocery-1.json")
    assert {"project", "version", "workspace"} <= set(body)
    assert isinstance(body["version"], dict)

    monkeypatch.setattr(generate_version, "httpx", _Client(body))
    version = generate_version.version(V1_PROJECT, 1, "k")
    assert version == body["version"]

    assert generate_version.compare_preprocessing(version["preprocessing"]) == []
    assert version["augmentation"] == {}
    # The fields `--verify` prints, present on the real body rather than defaulted from a fake.
    for field in ("images", "splits", "created", "generating"):
        assert field in version, field


def test_the_export_answer_carries_the_link_and_the_progress_together():
    """Why `export_link` reads the body instead of the status code.

    The ready answer carries **both** `export.link` and a `progress` of 1, so "the body says
    `progress`" is not a statement about readiness and a reader keyed on it would wait forever.
    `export_link` is asked first and that ordering is the contract; `export_progress` is only
    meaningful when the link is absent - which is also where the recorded evidence stops: the
    project's export was already cached, so the route answered 200 with the link on the first
    request and there is **no 202 progress recording**. That branch is covered by
    `test_train_model.py`'s polling fakes, and either thread is reachable again by asking for a
    format this project has never exported.

    `export.format` is asserted against `train_model.EXPORT_FORMAT` because that identifier is the
    one value in the trainer that could not be verified offline: a wrong one comes back as an error
    body rather than a bad download. This is the answer that says it is right.
    """
    body = fixture("export-scanncart-grocery-v1-link.json")

    assert set(body["export"]) >= {"format", "link", "size"}
    assert train_model.export_link(body) == body["export"]["link"]
    assert train_model.export_progress(body) == 1.0
    assert body["export"]["format"] == train_model.EXPORT_FORMAT
    # Redacted by the recorder, and this is what keeps it that way: the real one is a signed link.
    assert body["export"]["link"].endswith("?<redacted>")
    assert body["export"]["format"] in body["version"]["exports"]

    # And the same version spelled two ways, which is the trap in this body. The single-version
    # route answers `auto-orient: true`; the block embedded in an export answer wraps the same
    # step as `{"enabled": true}`, which is *truthy but not `True`* - so `compare_preprocessing`
    # reports a mismatch on a version that matches. Today's `--verify` reads the single-version
    # route and is right to; this is the assertion that would say so if a reader were ever pointed
    # at the embedded block instead.
    assert body["version"]["preprocessing"]["auto-orient"] == {"enabled": True}
    assert generate_version.compare_preprocessing(body["version"]["preprocessing"]) != []
    assert generate_version.compare_preprocessing(
        fixture("version-scanncart-grocery-1.json")["version"]["preprocessing"]
    ) == []


# --------------------------------------------------------------------------
# The search route
# --------------------------------------------------------------------------


def test_the_search_records_split_the_three_states_on_the_type_of_annotations():
    """`state_of`'s rule, on both recorded pages, because it is the one that re-opens the negatives.

    An image with no annotation record carries a *list* for `annotations`, and a deliberately
    marked-empty one carries a dict with `count: 0`. Reading the dict as "no boxes = not labeled"
    would re-open every hard negative in the project, silently and in the direction of *more*
    training data being missing rather than present - which is why the null shape is recorded
    rather than described. The background batches are the last thing uploaded, so the second page
    (`total - limit`, aimed at the end of the index) is where they are: a page of labelled frames
    carries neither half of the rule, which is what made recording one page not enough.

    **Neither recording holds the third state** - `annotations` absent, or present as a list. That
    is stated rather than glossed: `test_label_progress.py` covers it with hand-built records, and
    the paired assertions above are what the live pages can support.
    """
    page = fixture(f"search-{V2_PROJECT}-page.json")
    last = fixture(f"search-{V2_PROJECT}-last-page.json")

    for body in (page, last):
        assert isinstance(body["total"], int) and isinstance(body["offset"], int)
        assert isinstance(body["results"], list) and body["results"]
        for record in body["results"]:
            assert {"name", "split", "tags", "annotations"} <= set(record)
            state, boxes = label_progress.state_of(record)
            assert state in {"labeled", "null"}, (record["name"], state)
            assert (state == "labeled") == (boxes > 0)

    # One page is a labelled slice of the project - count > 0 with the per-class breakdown ...
    assert all(record["annotations"]["count"] > 0 for record in page["results"])
    assert all(record["annotations"].get("classes") for record in page["results"])
    # ... and the other is the background batch: count 0, no classes, and never "unlabeled".
    assert all(record["annotations"]["count"] == 0 for record in last["results"])
    assert all(record["annotations"].get("classes") == {} for record in last["results"])
    assert all(record["name"].startswith("negative") for record in last["results"])
    assert {label_progress.state_of(record)[0] for record in last["results"]} == {"null"}
    # The pagination fields are the ones the sweep joins on: `offset` is what it increments and
    # `total` is what it converges to, so a page that stopped carrying either would loop forever.
    assert last["offset"] == last["total"] - len(last["results"])


def test_the_project_count_and_the_search_total_disagree():
    """Recorded evidence for why `project_readout` sweeps the search index instead of reading
    `images`.

    The project route reports its own image total and the index reports another one, and they are
    not the same number - the index counts the frames that carry a *decision* (including the nulls
    that enter a version as background), which is the number an operator has to see before spending
    a version number. If a later recording finds these equal, the readout's reason for paying for a
    paginated sweep has changed and that is worth re-reading rather than relaxing.
    """
    project_body = fixture(f"project-{V2_PROJECT}.json")
    total = fixture(f"search-{V2_PROJECT}-page.json")["total"]

    assert isinstance(project_body["project"]["images"], int)
    assert project_body["project"]["images"] != total, (
        "the project's own count and the search index now agree - the readout's reason for sweeping "
        "the index (only annotated frames enter a version, and a null is a decision) needs re-reading"
    )


# --------------------------------------------------------------------------
# The Trash route, and what the recordings must not carry
# --------------------------------------------------------------------------


def test_the_trash_route_gives_back_the_version_that_holds_its_number():
    """`_trashed_versions`' filter, run over the recording: `type` and `parentUrl` are what it keys
    on, and this project's trashed entry is one version of the project the next generation comes
    from. Recorded because the whole tool's version numbering rests on it: the item is why the next
    generation is 2, and its `id` is a bare `"1"` while the project body spells the same version
    `.../snc-grocery/1` - two spellings of one number, neither of which may be parsed as the other.
    """
    trash = fixture("trash-yusri-caloyloy.json")
    items = clean_v2._trashed_versions(_Client(trash), "k", label_classes.WORKSPACE, V2_PROJECT)

    assert len(items) == 1
    item = items[0]
    assert item["type"] == "version" and item["parentUrl"] == V2_PROJECT
    assert item["id"] == "1"
    # A project's trash is filtered out rather than reported as a version of it.
    assert clean_v2._trashed_versions(_Client(trash), "k", label_classes.WORKSPACE, V1_PROJECT) == []


def test_no_recording_carries_a_credential_or_an_address():
    """The recordings are committed, so what they may contain is part of the contract.

    Both leaked on the first recording: the export link is a *signed* URL and the Trash body names
    the account that deleted the version, as an address. The recorder redacts both and prints what
    it removed; this asserts the files, because a redactor is a promise and a file is evidence.
    """
    files = sorted(FIXTURES.glob("*.json"))
    assert files, f"no recordings under {FIXTURES}"

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert not CREDENTIAL_IN_TEXT.search(text), f"{path.name} carries a signed/credential URL"
        assert not EMAIL.search(text), f"{path.name} carries an address"

    trash = fixture("trash-yusri-caloyloy.json")
    assert trash["items"][0]["deletedByName"] == "<redacted>"
    assert set(trash) == {"items", "sections"}
