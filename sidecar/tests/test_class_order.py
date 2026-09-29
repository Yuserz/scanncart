"""One rule, five tools: whose class list a label index means.

`accept_v2` learned this the expensive way. It scored v1 against a set whose labels index another
generation's order and reported `mAP50` 0.142, with the permutation's one fixed point as the only
class it found anything of, against 0.966 on the same frames read in v1's own order (that module's
docstring). It answers the problem by measuring each weight in its own order, because a comparison
has two sides.

The tools that *audit* frames cannot remap a set, so they answer it the other way: they refuse a set
whose labels are not in the class order the numbers are reported in. A label row carries a bare index
and no name, so unlike a prediction - which arrives with the model's own names attached and is
re-keyed through them - it cannot be repaired after the fact. That asymmetry is the whole rule, and
it is why these tools refuse the *labels* rather than a weight whose head indexes somewhere else.

The tests are the rule and then its four call sites, because the rule on its own would be a function
nothing used: a tool that measures a weight against frames and skips the check is exactly the defect
this module exists to notice. Each call site is exercised through its own `main`, so a tool that
dropped the call fails here rather than at the next operator's terminal.

`class_gaps` has the mirror-image failure, and it is the one a *call log* cannot see: the relation
can be written out by hand as easily as it can be skipped, and a tool that re-derives it never calls
the owner and so never appears in the log above. So the last test here scans the whole tools tree
for that shape - a comprehension or set difference that walks one class list and keeps the names the
other lacks - and fails with the file and the line, the way this repo's other scans do. The subject
is a difference *between two class lists*, which is why the scan is armed with the class-list
expressions the tools themselves name (and the locals derived from them) instead of with any
membership test at all.
"""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import audit_recall
import build_dataset
import clamp_probe
import clean_v2
import generations
import label_classes
import spec_check
import train_model
import unsure_probe
from generations import V1, V2
from tests.dataset_tool_helpers import _export_with_names, _fake_export

# Rebuilt from the module that owns the tools tree, not copied by value.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _mismatched(tmp_path):
    """A set declaring `V2`'s names in the order `V1` indexes: the same seven products, moved.

    Not a contrived pairing - it is the shape `merged-v2` has (built in `V2.classes` order) and the
    one the acceptance was measured through before it read each weight's own record.
    """
    return _fake_export(tmp_path / "export-v2", names=list(V2.classes))


def _weights(tmp_path) -> str:
    """A weight that only has to exist, so the tools' own checks get out of the way of this one."""
    path = tmp_path / "v1.pt"
    path.write_bytes(b"weights")
    return str(path)


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def test_a_set_in_its_generations_order_needs_no_refusal(tmp_path):
    """The ordinary case, and the one that must stay cheap: a set whose labels index its
    generation's list is what every documented run measures against."""
    export = _fake_export(tmp_path / "export-v1", names=list(V1.classes))

    assert train_model.require_labels_order(V1, export) == V1.classes
    assert train_model.labels_class_order(V1, export) == list(V1.classes)


def test_a_set_whose_labels_are_in_another_order_is_refused(tmp_path):
    """Membership is not order, which is the point: both lists hold these seven names, so every
    cheaper check passes while each label row is read as a position in the wrong list."""
    export = _mismatched(tmp_path)

    with pytest.raises(SystemExit) as caught:
        train_model.require_labels_order(V1, export)

    message = str(caught.value)
    assert "another class order" in message
    # Both lists, because the difference *is* the finding and neither list alone shows it.
    assert V2.classes[0] in message and V1.classes[0] in message
    assert "regenerate the export" in message


def test_a_set_with_no_readable_class_list_is_refused_rather_than_assumed(tmp_path):
    """"Cannot tell" is not "agrees": with no names in the set there is nothing that says which
    product a label index means, and falling back to the generation's list is the assumption that
    produced the 0.142 rather than a reading of the set."""
    empty = tmp_path / "not-an-export"
    empty.mkdir()

    with pytest.raises(SystemExit) as caught:
        train_model.require_labels_order(V1, empty)

    assert "no class list could be read" in str(caught.value)


def test_the_refusal_names_the_tool_that_asked(tmp_path):
    """Four tools call this and three of them are reached from a shell, where an unattributed
    `SystemExit` would not say which one refused."""
    with pytest.raises(SystemExit) as caught:
        train_model.require_labels_order(V1, _mismatched(tmp_path), tool="audit_recall.py")

    assert str(caught.value).startswith("audit_recall.py: ")


def test_the_check_reads_the_generations_own_export_by_default(tmp_path, monkeypatch):
    """The default is `generation.export_dir` - the one the tools' `--dataset-dir` replaces - so a
    caller that points the tool at another set cannot end up checked against the old one."""
    export = _fake_export(tmp_path / "export-v1", names=list(V1.classes))
    monkeypatch.setitem(generations.GENERATIONS, "v1", replace(V1, export_dir=export))

    assert train_model.require_labels_order(generations.get("v1")) == V1.classes


def test_one_owner_decides_what_each_class_list_declares(tmp_path, monkeypatch):
    """The membership relation has one implementation, and all three judgement sites ask it.

    `train_model.check_export` judges an export against its generation, `clean_v2.class_list_rows`
    judges the live Roboflow project before a shoot, and `build_dataset.translation_problem` judges
    each side of a merge against the set's order: three readers of the same two-list relation, which
    is `generations.class_gaps`. Spying on it is what makes that a test rather than a habit - a tool
    that re-derived the difference by hand (`[n for n in names if n not in expected]`, which all
    three of them used to) would answer every behavioural question below correctly and never call
    the owner, and *that* is the drift worth guarding: one tool judging membership where another
    judges order, or forgetting a direction.
    """
    import yaml

    asked: list[tuple[list[str], list[str]]] = []
    real = generations.class_gaps

    def spy(names, expected):
        asked.append((list(names), list(expected)))
        return real(names, expected)

    monkeypatch.setattr(generations, "class_gaps", spy)

    # The relation itself, both directions - and position is deliberately not part of it, which is
    # `order_of`'s question: a reversed list is no member of it, in either direction.
    gaps = generations.class_gaps(["a", "b"], ["b", "c"])
    assert (gaps.source_only, gaps.target_only) == (("a",), ("c",))
    assert gaps.any is True
    assert generations.class_gaps(list(V2.classes), list(V1.classes)).any is False

    export = _export_with_names(
        tmp_path / "export-v2", {"train": ["a.jpg"], "valid": ["b.jpg"], "test": ["c.jpg"]}
    )
    train_model.check_export(export, V2)
    assert (list(V2.classes), list(V2.classes)) in asked

    build_dataset.translation_problem(["x"], ["y"], source="v1")
    assert (["x"], ["y"]) in asked

    # The third reader: the live project's class list, asked before a shoot rather than after a
    # version number is spent. Its source list is sorted on the way in, so that is what the owner
    # sees.
    clean_v2.class_list_rows(list(V2.classes))
    assert (sorted(V2.classes), list(clean_v2.V1_CLASSES)) in asked

    # `generations.added_over` is one direction of the same relation - "which classes does `later`
    # declare that `earlier` does not" - and was the last place in the tree holding its own copy of
    # it, inside the owner's own module. The scan at the bottom of this module is what keeps a
    # fourth one from appearing; this pins the one that had already.
    generations.added_over(V2, V1)
    assert (list(V2.classes), list(V1.classes)) in asked

    # And the answer is the same relation read from the two sides, one direction each: the merge
    # reports a source list declaring what the set has no position for, and a training run reports
    # the export's copy of the same fact.
    body = yaml.safe_load((export / "data.yaml").read_text(encoding="utf-8"))
    body["names"] = [*V2.classes, "Palmolive Naturals Bar Soap 85g"]
    (export / "data.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    _, problems = train_model.check_export(export, V2)
    assert "the export declares classes that are not v2 classes: Palmolive Naturals Bar Soap 85g" in problems
    assert "'Palmolive Naturals Bar Soap 85g'" in build_dataset.translation_problem(
        [*V2.classes, "Palmolive Naturals Bar Soap 85g"], source="v1"
    )

    body["names"] = list(V2.classes[:-1])
    (export / "data.yaml").write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    _, problems = train_model.check_export(export, V2)
    assert f"the export has no class for: {V2.classes[-1]}" in problems

    # ...and the project's own reading of the same pair, in the shape `sanity` prints it: a name
    # the roster declares is missed work, and one it does not know is a head output nothing can
    # ever match - the class's own sentence, from the owner's answer.
    rows = clean_v2.class_list_rows([*V2.classes[:-1], "Palmolive Naturals Bar Soap 85g"])
    detail = rows[0][2]
    assert rows[0][0] == "warn"  # a missing name is work not done yet
    assert f"missing 1 name(s):\n  {V2.classes[-1]}" in detail
    assert "unexpected: Palmolive Naturals Bar Soap 85g" in detail
    assert "(a Palmolive class means it was re-created after being dropped)" in detail


# --------------------------------------------------------------------------
# The owner, enforced across the tools tree
# --------------------------------------------------------------------------

#: The one file that is *allowed* the shape, and the one function in it that is: the owner's own
#: body. Every other difference in the tree asks `class_gaps`, including the rest of
#: `generations.py` - `added_over` was a second derivation inside this very module until the scan
#: found it, which is why the exemption is a function name rather than the file.
OWNER = ("sidecar/tools/generations.py", "class_gaps")

#: Every expression in the tools tree that names a class list. Each is somewhere a class list is
#: *read from* - `label_classes.SLUG_TO_CLASS`/`NEW_CLASS_SLUGS`, `clean_v2.CLASS_MAP`/`V1_CLASSES`,
#: `annotate.store.CLASS_NAMES`, `build_dataset.CANONICAL_NAMES`/`INDEX_BY_NAME`, a generation's own
#: `.classes`, the `"classes"` key a Roboflow project's list travels under, and the runtime's two
#: rosters - so a difference computed against one is a hand-rolled `class_gaps`. Spelled out rather
#: than discovered because a *name* is all the scan can see: by the time a tool has both lists in
#: hand they are values, and `class_gaps(names, expected)` itself names none of these.
CLASS_LIST_NAMES = frozenset(
    {
        "SLUG_TO_CLASS",
        "NEW_CLASS_SLUGS",
        "CLASS_MAP",
        "V1_CLASSES",
        "V2_CLASSES",
        "CLASS_NAMES",
        "CANONICAL_NAMES",
        "INDEX_BY_NAME",
        "CLASSES",
        "V1_ROSTER",
        "V2_ROSTER",
        "roster",
        "classes",
    }
)

#: `path`, the source the site still carries, and why that site is not two class lists being
#: compared - the same shape as `test_docs_claims.EXEMPT`, and checked the same way: a snippet that
#: is gone fails `test_the_class_list_scan_exemptions_are_still_live`, so an exemption cannot
#: outlive the site it was written for and start covering the next one.
EXEMPT: tuple[tuple[str, str, str], ...] = (
    (
        "sidecar/tools/label_classes.py",
        "if s not in V2_ONLY_SLUGS",
        "a subset of one mapping rather than two lists: `V2_ONLY_SLUGS` names the *slugs* v2 added, "
        "so what this keeps is the v1-inherited half of `SLUG_TO_CLASS` - the names never meet a "
        "second list",
    ),
    (
        "sidecar/tools/label_progress.py",
        "if t not in known",
        "an image's provenance rather than a class list: `known` is the class slugs *and* the "
        "distance axis rolled together on purpose, and what is left over is the session tag",
    ),
)


def _names_a_class_list(node: ast.AST, derived: set[str]) -> bool:
    """Does this expression reach a class list? A token, a generation's `.classes`, the `"classes"`
    key a project's list arrives under, or a local this scope derived from one.
    """
    for part in ast.walk(node):
        if isinstance(part, ast.Name) and (part.id in CLASS_LIST_NAMES or part.id in derived):
            return True
        if isinstance(part, ast.Attribute) and part.attr in CLASS_LIST_NAMES:
            return True
        if isinstance(part, ast.Constant) and part.value == "classes":
            return True
    return False


def _own_nodes(scope: ast.AST):
    """Every node of this scope and none of a nested one.

    A nested function is *yielded* - so the walk below can enter it as its own scope - but not
    descended into: its locals are its own, which is what keeps the alias tracking honest.
    """
    for child in ast.iter_child_nodes(scope):
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield from _own_nodes(child)


def _class_locals(scope: ast.AST) -> set[str]:
    """The names in this scope bound to a class list, to a fixed point.

    Every copy this scan is for names its lists before comparing them - `wanted =
    list(generation.classes)`, `names = project.get("classes")`, `have = set(earlier.classes)` - so
    an operand that is only a local would be invisible without this. Name-based and scope-local on
    purpose: the same name in two functions is two facts, and a name derived here is class-ish for
    this scope's difference shapes only.
    """
    derived: set[str] = set()
    assignments = [node for node in _own_nodes(scope) if isinstance(node, ast.Assign)]
    for _ in range(3):  # one alias per round, enough for a chain such as `want` -> `still`
        for node in assignments:
            if _names_a_class_list(node.value, derived):
                derived.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return derived


def _is_difference(node: ast.AST, derived: set[str]) -> bool:
    """Is this node two class lists' declarations, written out by hand?

    The two spellings, both of them `class_gaps`: a comprehension that walks one list and keeps the
    names the other lacks (`[n for n in names if n not in expected]`), and the set difference of
    two lists (`set(a) - set(b)`). A comprehension counts only when it *builds* the names it walks
    - a filter over ids, counts or shapes (`{DISTRACTORS[c] for c in names if c in ...}`) is a
    different question about the same list - and the test has to be a `not in`: `[... if name in
    present]` reads membership, it does not report a gap.
    """
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        if len(node.generators) != 1:
            return False
        target = node.generators[0].target
        bound = (
            {target.id}
            if isinstance(target, ast.Name)
            else {elt.id for elt in getattr(target, "elts", ()) if isinstance(elt, ast.Name)}
        )
        built = (node.key,) if isinstance(node, ast.DictComp) else (node.elt,)
        if not any(isinstance(part, ast.Name) and part.id in bound for part in built):
            return False
        for cond in node.generators[0].ifs:
            if (
                isinstance(cond, ast.Compare)
                and isinstance(cond.left, ast.Name)
                and cond.left.id in bound
                and any(isinstance(op, ast.NotIn) for op in cond.ops)
            ):
                sides = (node.generators[0].iter, *cond.comparators)
                return any(_names_a_class_list(side, derived) for side in sides)
        return False
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and any(_names_a_class_list(side, derived) for side in (node.left, node.right))
    )


def _scan(path: Path) -> list[str]:
    """Every hand-rolled class-list difference in one file, as `path:line: source`.

    Read as text rather than imported: the tools are scripts with workspace-lazy imports, and this
    is a scan of *source* anyway - what it is looking for is a shape, and the shape is only visible
    before it runs.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    where = path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)
    owner_function = OWNER[1] if where == OWNER[0] else None
    sites: list[str] = []

    def walk(scope: ast.AST, derived: set[str]) -> None:
        nodes = list(_own_nodes(scope))
        for node in nodes:
            if _is_difference(node, derived):
                sites.append(f"{where}:{node.lineno}: {ast.get_source_segment(source, node)}")
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if getattr(node, "name", None) == owner_function:
                    continue
                walk(node, _class_locals(node))

    walk(tree, _class_locals(tree))
    return sites


def _tree_sites() -> list[str]:
    """Every site in the tools tree, before the exemptions are applied."""
    tools = Path(label_classes.__file__).parent
    return [site for path in sorted(tools.glob("*.py")) for site in _scan(path)]


def _exempt(site: str) -> str | None:
    """The reason this site is not two class lists being compared, if any exemption covers it."""
    for path, snippet, reason in EXEMPT:
        if site.startswith(f"{path}:") and snippet in site:
            return reason
    return None


def test_no_hand_rolled_class_list_difference_in_the_tools_tree():
    """The owner can be bypassed as easily as skipped, and a bypass leaves no trace in a call log.

    `test_one_owner_decides_what_each_class_list_declares` proves the readers it knows about ask
    `class_gaps`, and it is blind to the failure this catches: a tool that writes the relation out
    by hand never calls the owner, so there is nothing in the spy's log to notice. That is not
    hypothetical - five copies of it were in this tree (`train_model`'s export check,
    `clean_v2`'s project rows, both sides of `build_dataset`'s merge, `accept_v2`'s order view,
    `generate_version`'s report, `label_classes`' three rows and `generations.added_over` itself),
    and every one of them answered *correctly*: what they cannot be is one fact, so a direction
    forgotten in one of them is a tool disagreeing with another about the same two lists - which is
    exactly what `class_gaps` was created for.

    The subject is a difference between two class lists, which is why the scan is armed with the
    expressions these tools read their class lists from (and the locals they derive from them)
    rather than with membership tests in general: `[n for n in staged if n not in index]` is a
    perfectly ordinary line, and a guard that flagged it would be turned off within a week.

    What it does not claim, and this is the hole it leaves on purpose: two class lists that arrive as
    bare parameters and are compared without ever being named are invisible here - the shape is the
    right shape and nothing about the code says which axis it is. That is the case a reviewer still
    has to notice, and it is why the copies that existed were *converted* rather than merely
    monitored: `generations.added_over`, `generate_version`'s report, `label_classes`' three rows,
    `train_model`'s other-generation note, `accept_v2.order_view` and the label-row counts in
    `build_dataset` all name at least one of the lists, so the scan would have caught every one of
    them.

    Scope: the tools tree. The runtime holds this relation by hand in `app/roster.py` - both
    directions of it - and is *supposed* to: the sidecar may not import `tools/`, which is why that
    copy is mirrored instead of asked (`tests/test_roster.py`), so it is a different fact held by a
    different guard rather than a fifth reader of this owner.
    """
    offenders = [site for site in _tree_sites() if _exempt(site) is None]
    assert offenders == [], (
        "a class-list difference written by hand: "
        + "; ".join(offenders)
        + " - ask `generations.class_gaps(names, expected)` instead, or add the site to EXEMPT with "
        "the reason it is a different relation"
    )


def test_the_class_list_scan_catches_every_spelling_it_was_written_for(tmp_path):
    """Non-vacuity, planted rather than asserted: a scan that found nothing would pass the test
    above, so each spelling the scan is armed for is written into a file and required back - with
    the file and the line, and with the source, because that is the whole of what an operator gets.

    The last case is the control. `clean_v2` and `import_labels` both difference image names
    against a project index, which is the same *shape* one axis over, and a scan that could not
    tell them apart would be a guard nobody keeps.
    """
    planted = tmp_path / "planted.py"
    planted.write_text(
        "SLUG_TO_CLASS = {'a': 'A'}\n"
        "CANONICAL_NAMES = ('A',)\n"
        "\n"
        "def by_token(names):\n"
        "    return [n for n in CANONICAL_NAMES if n not in names]\n"
        "\n"
        "def by_locals(project, generation):\n"
        "    names = project.get('classes')\n"
        "    wanted = list(generation.classes)\n"
        "    return [name for name in wanted if name not in set(names)]\n"
        "\n"
        "def by_set_difference(names):\n"
        "    have = set(SLUG_TO_CLASS.values())\n"
        "    return sorted(have - names)\n"
        "\n"
        "def another_axis(staged, index):\n"
        "    return [n for n in staged if n not in index]\n"
    )

    found = _scan(planted)

    for lineno, snippet in (
        (5, "if n not in names"),  # the export-check spelling, against a named list
        (10, "if name not in set(names)"),  # `generate_version`'s, against locals derived from two
        (14, "have - names"),  # the set difference `label_classes` had, one operand derived
    ):
        hits = [site for site in found if site.startswith(f"{planted}:{lineno}: ")]
        assert hits, f"line {lineno} ({snippet!r}) was not reported - the scan is vacuous: {found}"
        assert snippet in hits[0], hits[0]
    # The image-name difference is not this relation, and is not reported.
    assert not [site for site in found if site.startswith(f"{planted}:17: ")], found


def test_the_class_list_scan_exemptions_are_still_live():
    """An exemption is a claim too, and the one that could rot: a snippet edited away leaves an
    exemption covering text nobody read, and the next site with that shape under it is waved
    through. Both ends, like `test_docs_claims.test_the_docs_guard_exemptions_are_still_live`.
    """
    for path, snippet, reason in EXEMPT:
        assert any(
            site.startswith(f"{path}:") and snippet in site for site in _tree_sites()
        ), (
            f"{path} no longer holds {snippet!r}, so the exemption ({reason}) covers nothing - "
            "drop it, or point it at the site it was written for"
        )


# --------------------------------------------------------------------------
# The call sites
# --------------------------------------------------------------------------


def _refusal(tool_main, tmp_path, monkeypatch, *argv: str) -> str:
    """`tool_main` run against a set in another class order, and the sentence it refused with."""
    export = _mismatched(tmp_path)
    monkeypatch.setitem(generations.GENERATIONS, "v1", replace(V1, export_dir=export))
    with pytest.raises(SystemExit) as caught:
        tool_main(["--generation", "v1", "--weights", _weights(tmp_path), *argv])
    return str(caught.value)


def test_audit_recall_refuses_a_set_in_another_class_order(tmp_path, monkeypatch):
    """`truth` is a label row's index and the report is indexed by the generation's list, so this is
    the tool where the mismatch lands in the numbers: pointed at v1's set as v2 it printed `0.000`
    recall for every class and counted 97 instances of a product that set holds none of."""
    assert _refusal(audit_recall.main, tmp_path, monkeypatch).startswith("audit_recall.py: ")


def test_spec_check_refuses_a_set_in_another_class_order(tmp_path, monkeypatch):
    """Its accuracy half goes through `audit_recall.collect`, but the *timing* half runs first and
    prints a PASS/FAIL strip: a refusal that arrived after it would have checked the wrong set's
    accuracy against the PRD's targets."""
    assert _refusal(spec_check.main, tmp_path, monkeypatch).startswith("spec_check.py: ")


def test_clamp_probe_refuses_a_set_in_another_class_order(tmp_path, monkeypatch):
    """Two of its three populations - the control frames and `training_labels`, the boxes the weights
    were taught - are read out of the export, and the claims are statements about *that* population,
    so another set is another dataset being priced under this generation's name."""
    assert _refusal(clamp_probe.main, tmp_path, monkeypatch).startswith("clamp_probe.py: ")


def test_unsure_probe_refuses_a_set_in_another_class_order(tmp_path, monkeypatch):
    """Its census names every priced detection through the generation's class list and divides by the
    labelled frames, so a set in another order prices the rule against the wrong products."""
    assert _refusal(unsure_probe.main, tmp_path, monkeypatch).startswith("unsure_probe.py: ")
