#!/usr/bin/env python3
"""Fail if a tracked file names private context or a local path.

This package is distributed to other researchers; everything in it
must make sense to an external reader. Private downstream analysis
repositories and local development machines mean nothing outside
this project family -- if a comment, docstring, test, doc or data
file needs to explain something a downstream pipeline or a machine
did, it should describe it generically instead of naming it.

Every file git tracks is scanned (code, tests, docs, JSON, config),
comments and docstrings included, case-insensitively; binary files
are skipped. This file is the one exception, since it has to list
the names it bans.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Repos private to this project family. Update if one is archived/
# deleted (drop it) or a new private downstream repo appears (add it)
# -- don't list a repo that no longer exists, it's just noise.
BANNED_REPOS = ["mutation_rates", "tcga_analysis", "dnds_comparison"]

# Local development infrastructure that has no meaning off-machine.
BANNED_HOSTS = ["gauss", "snark", "boojum", "jabberwocky"]

# A name counts when no letter or digit is glued to it: "<name>_dir"
# or "<name>/" match, "<name>ian" does not. (Underscores are
# deliberately not a boundary, unlike regex \b, so a name inside a
# snake_case identifier is still caught.)
BANNED_PATTERNS = [
    (
        name,
        re.compile(
            rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])",
            re.IGNORECASE,
        ),
    )
    for name in BANNED_REPOS + BANNED_HOSTS
]

PATH_PATTERN = re.compile(r"(?<![\w.])/home/\w+|(?<![\w.])/Users/\w+")

SELF = Path(__file__).resolve()


def scan_text(text):
    """Return ``(lineno, reason, line)`` for each offending line."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in BANNED_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, name, line.strip()))
        if PATH_PATTERN.search(line):
            hits.append((lineno, "local path", line.strip()))
    return hits


def tracked_files(root):
    """The files git tracks under ``root`` (paths relative to it)."""
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    return [Path(p) for p in out.decode().split("\0") if p]


def find_violations(root=REPO_ROOT):
    """Scan every tracked text file under ``root``."""
    root = Path(root)
    violations = []
    for rel in tracked_files(root):
        path = root / rel
        if path.resolve() == SELF or not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:  # binary
            continue
        text = data.decode("utf-8", errors="replace")
        for lineno, reason, line in scan_text(text):
            violations.append((rel, lineno, reason, line))
    return violations


def main():
    violations = find_violations()
    if not violations:
        return 0
    print("Found references that don't belong in a public package:\n")
    for rel, lineno, reason, line in violations:
        print(f"  {rel}:{lineno} ({reason}): {line[:200]}")
    print(
        "\nDescribe private repositories, machines and local paths "
        "generically instead of naming them."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
