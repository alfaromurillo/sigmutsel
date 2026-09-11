#!/usr/bin/env python3
"""Fail if a public-facing doc names a private downstream repo or a
local development path.

This package is distributed to other researchers; its docs must be
self-contained. Downstream analysis repos (mutation_rates,
tcga_analysis, dnds_comparison) are private and meaningless to an
external reader -- if a doc needs to explain something a downstream
pipeline does, describe it generically instead of naming the repo.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Repos private to this project family. Update if one is archived/
# deleted (drop it) or a new private downstream repo appears (add it)
# -- don't list a repo that no longer exists, it's just noise.
BANNED_REPOS = ["mutation_rates", "tcga_analysis", "dnds_comparison"]

# Local development infrastructure that has no meaning off-machine.
BANNED_HOSTS = ["gauss", "snark", "boojum"]

PATH_PATTERN = re.compile(r"(?<![\w.])/home/\w+|(?<![\w.])/Users/\w+")

DOC_GLOBS = ["*.md", "**/*.md"]


def find_violations():
    violations = []
    seen = set()
    for pattern in DOC_GLOBS:
        for path in REPO_ROOT.glob(pattern):
            if path in seen or ".venv" in path.parts:
                continue
            seen.add(path)
            rel = path.relative_to(REPO_ROOT)
            for lineno, line in enumerate(
                path.read_text(errors="replace").splitlines(), start=1
            ):
                for name in BANNED_REPOS + BANNED_HOSTS:
                    if re.search(rf"\b{re.escape(name)}\b", line, re.IGNORECASE):
                        violations.append((rel, lineno, name, line.strip()))
                if PATH_PATTERN.search(line):
                    violations.append((rel, lineno, "local path", line.strip()))
    return violations


def main():
    violations = find_violations()
    if not violations:
        return 0
    print("Found references that don't belong in public docs:\n")
    for rel, lineno, reason, line in violations:
        print(f"  {rel}:{lineno} ({reason}): {line}")
    print(
        "\nDescribe downstream repos/local infra generically instead of "
        "naming them -- see mutation_rates/CLAUDE.md's 'Working on the "
        "sigmutsel package itself' section for the full rule."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
