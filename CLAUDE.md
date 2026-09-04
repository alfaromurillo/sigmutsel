# sigmutsel

Signature-based mutation rate estimation and selection inference.
Public package — see `DEVELOPMENT.md` for internals (module map,
`Model` mechanics, the consequence-split and `r_g` machinery) and
`CONTRIBUTING.md`/`SETUP_GUIDE.md` for setup. This file covers only
the workflow traps that cost time and are invisible from the code.

## Checks before committing

`pre-commit` runs `black --check` and `ruff check` on every commit,
both pinned in `.pre-commit-config.yaml`.

**Lint with `ruff`, never `python3 -m ruff`.** They are different
versions on this machine — `ruff` on PATH matches the pinned
0.16.2, while `python3 -m ruff` is older (0.14.11) with a smaller
default rule set. Checking with the module form gives a false pass
and the commit hook then rejects the same files.

```sh
black src/ tests/ && ruff check src/ tests/ && pytest tests/ -q
```

`tests/test_split_maf_file.py::test_force_generation_overwrites`
compares `st_mtime` before and after a rewrite, so it can fail
under load when the timestamps land in the same tick. It passes in
isolation; re-run before assuming a real breakage.

## `__version__` is stale in an editable install

`setuptools-scm` writes `_version.py` at install time, so an
editable checkout keeps reporting whatever it said then — it can be
many commits behind `git describe`. Do not use `sigmutsel.__version__`
to check whether a checkout has some commit; use git.

The staleness matters when refreshing the non-editable install on
gauss, where `tcga_analysis`'s notes suggest verifying the `.postN`
suffix against the commits just pushed: compare it against
`git describe`, not against the local editable version.

```sh
# on gauss, after pushing to master
code/.venv/bin/python3 -m pip install --force-reinstall --no-deps \
  'git+https://github.com/alfaromurillo/sigmutsel.git@master'
```

## Public package, internal projects

Docstrings and `DEVELOPMENT.md` must not reference `mutation_rates`,
`tcga_analysis`, `sigmutselcovs` or `dnds_comparison` — keep
project-specific rationale in those repos' own notes.
