"""Every CI job must test against the SAME siblings, pinned the SAME way.

``cogno-soma`` depends on ``cogno-anima`` and friends, none released to PyPI at a version
that matches git: the published ``cogno-anima`` is 0.1.0 and main is far ahead under that same
number, so ``pip install -e .`` alone silently resolves the stale wheel. The nightly job did
exactly that and spent four nights red on ``module 'cogno_anima.metakeys' has no attribute
'JUDGE_VERDICT'`` — one job testing main, the other a release from before the key existed.

Parsed with a real YAML parser, not regex, and PyYAML is a hard dev dependency rather than an
``importorskip`` — a guard that skips itself is the same "green means nothing" state it exists
to end.

The first version of this file read the workflow with hand-rolled patterns and had five holes,
each of which let through exactly the divergence it advertises catching:

* job names with ``_`` or uppercase were not treated as block boundaries, so a whole new
  nightly's steps were absorbed into the previous job and never checked;
* only ``pip install -e .`` counted as installing the package, so ``pip install .`` escaped;
* requirements had to be one per ``pip install`` command, so a correct-but-reformatted chain
  went RED;
* the git ref was thrown away, so two jobs pinned to different commits read as identical;
* ``_JOB.finditer(text, re.MULTILINE)`` passed the flag as the ``pos`` argument, so the named
  pattern never matched and only an inline fallback did any work.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"

# `cogno-x @ git+https://…[@ref]` inside a quoted requirement, anywhere in a run: block.
# Captures the name AND the url-with-ref: two jobs on different refs are not the same chain.
_GIT_DEP = re.compile(r"(cogno-[\w.-]+?)\s*@\s*git\+(\S+?)(?=[\"'\s]|$)")
# Installing THIS package: editable or not, with or without extras, either quote style.
_SELF = re.compile(r"pip\s+install\s+(?:-e\s+)?[\"']?\.(?:\[[^\]]*\])?[\"']?(?:\s|$)")


def _jobs() -> "dict[str, str]":
    """Job name → all of its ``run:`` text joined. YAML does the block splitting, so a job
    called ``bench_nightly`` or ``Build-Docs`` is a job like any other."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for name, job in (doc.get("jobs") or {}).items():
        steps = (job or {}).get("steps") or []
        out[str(name)] = "\n".join(s.get("run", "") for s in steps if isinstance(s, dict))
    return out


def _chain(block: str) -> "frozenset[tuple[str, str]]":
    """The (package, git-url-including-ref) pairs a job installs from git."""
    return frozenset(_GIT_DEP.findall(block))


# ── guards on the guard ───────────────────────────────────────────────────────────────
def test_the_workflow_parses_and_has_the_jobs_we_think():
    """A moved file, a renamed job or a YAML change would otherwise make every check below
    pass over an empty mapping."""
    assert WORKFLOW.is_file()
    jobs = _jobs()
    assert {"test", "ollama-integration"} <= set(jobs), sorted(jobs)


def test_a_job_name_with_an_underscore_is_still_a_job():
    """The hole that mattered most: under the old regex, `bench_nightly:` was not a boundary,
    so its steps merged into the previous job — whose chain was already correct — and a
    brand-new nightly resolving PyPI passed green."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    doc["jobs"]["bench_nightly"] = {"steps": [{"run": 'pip install -e ".[dev]"'}]}
    jobs = {n: "\n".join(s.get("run", "") for s in (j.get("steps") or []))
            for n, j in doc["jobs"].items()}
    assert "bench_nightly" in jobs
    assert _chain(jobs["bench_nightly"]) == frozenset()      # seen, and seen as empty


def test_the_detectors_match_what_they_claim_to():
    for spelling in ('pip install -e ".[dev]"', "pip install -e .", "pip install .",
                     'pip install ".[dev]"', "pip install -e '.[dev]'"):
        assert _SELF.search(spelling), spelling
    for other in ("pip install requests", "pip install -r requirements-dev.txt"):
        assert not _SELF.search(other), other


def test_reformatting_the_chain_is_free_but_changing_the_ref_is_not():
    one_per_line = ('pip install "cogno-homeo @ git+https://h.git"\n'
                    'pip install "cogno-synapse @ git+https://s.git"')
    one_command = ('pip install "cogno-homeo @ git+https://h.git" '
                   '"cogno-synapse @ git+https://s.git"')
    assert _chain(one_per_line) == _chain(one_command)
    assert _chain('pip install "cogno-tool-belt @ git+https://t.git"')   # hyphens count
    floating = _chain('pip install "cogno-anima @ git+https://a.git"')
    pinned = _chain('pip install "cogno-anima @ git+https://a.git@abc123"')
    assert floating and pinned and floating != pinned


# ── the invariant ─────────────────────────────────────────────────────────────────────
def test_every_job_that_installs_this_package_uses_the_same_git_chain():
    installing = {n: b for n, b in _jobs().items() if _SELF.search(b)}
    assert installing, "no job installs this package — the check would be vacuous"
    chains = {name: _chain(block) for name, block in installing.items()}
    # The UNION is the reference, not the longest: a job cannot become the standard by
    # installing least, and a length tie cannot pick the reference by dict order.
    expected = frozenset().union(*chains.values())
    assert expected, "no job installs the siblings from git — every one resolves PyPI"
    drift = {name: sorted(f"{p} @ {u}" for p, u in (expected - deps))
             for name, deps in chains.items() if deps != expected}
    assert not drift, (
        "these jobs install this package with a DIFFERENT sibling chain, so they test "
        "different code — a missing entry resolves the stale PyPI wheel (same version "
        f"number, older code), and a different ref is a different commit: {drift}")
