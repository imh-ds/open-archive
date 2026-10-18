"""Frozen-study checks run against the commit each study was frozen and run on.

These checks hash every package source file and compare the hashes with the
study's frozen manifest, so they can only pass on that study's own commit.
The commit is extracted with ``git archive`` and the study's tests run there.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).parents[1]

# study: (commit recorded as source_sha in its published results, tests at that commit)
FROZEN = {
    "tree-recursive-signal (2026-10-16)": (
        "e1dc2fa5cf25611ff2ab54499594f6121298bfd7",
        ("tests/test_tree_recursive_signal.py", "tests/test_tree_recursive_signal_publication.py"),
    ),
}


def _extract(commit, target):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is unavailable")
    if subprocess.run([git, "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT,
                      capture_output=True).returncode != 0:
        pytest.skip(f"frozen commit {commit[:12]} is not in this clone (shallow checkout?)")
    archive = target.parent / "frozen.tar"
    with archive.open("wb") as out:
        subprocess.run([git, "archive", "--format=tar", commit], cwd=ROOT, stdout=out, check=True)
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    archive.unlink()


@pytest.mark.parametrize("study", sorted(FROZEN))
def test_frozen_study_checks_pass_on_their_commit(tmp_path, study):
    commit, tests = FROZEN[study]
    tree = tmp_path / "tree"
    _extract(commit, tree)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=tree, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
