"""Provenance: package stamps and run history for saved objects.

A saved dataset or model directory records *what the object is* --
its covariates, its signature parameters, its fitted scalars. This
module adds the two things a manifest needs to also say *where it
came from*: which build of this package wrote it, and which calls
produced it.

Two functions carry the whole mechanism:

- :func:`package_provenance` builds the stamp written into a
  manifest, and :func:`check_provenance` compares a stamp read back
  from disk against the running build, warning on a difference.
- :func:`record_call` is a decorator for the state-mutating methods
  of ``MutationDataset`` and ``Model``. It appends the call, with
  the arguments the caller actually passed, to the object's
  ``_run_history``, which is saved and reloaded with the object.

**The version stamp is two fields, and the second is the one that
matters.** ``sigmutsel.__version__`` comes from ``_version.py``,
which ``setuptools-scm`` writes at *install* time; in an editable
install it keeps reporting the version of the commit that was
checked out when ``pip install -e`` ran, however many commits ago
that was. A stamp built from it alone would record the last
reinstall, not the code that ran. So the stamp also carries a
``git describe`` of the tree the package is imported from, resolved
at call time, which is accurate for exactly the editable-checkout
case the version field gets wrong. It is ``None`` when the package
is not running from a git work tree (an ordinary wheel install),
which is the case where the version field is trustworthy instead.
"""

import functools
import inspect
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

#: Upper bound on retained run-history entries. Fitting loops call
#: ``estimate_gamma`` once per gene or variant, so an unbounded
#: history would grow with the cohort rather than with the analysis.
#: Past the cap the *middle* is elided: the first entries (how the
#: object was built) and the most recent ones (what it was last
#: asked to do) are the informative ends.
MAX_RUN_HISTORY = 1000

#: Longest string echoed into a history entry before truncation.
MAX_ARG_CHARS = 60

_UNRESOLVED = object()
_git_commit_cache = _UNRESOLVED


def git_commit():
    """Return ``git describe`` of the tree this package runs from.

    Returns ``None`` when the package is not inside a git work tree,
    or when git is unavailable. Resolved once per process.
    """
    global _git_commit_cache

    if _git_commit_cache is not _UNRESOLVED:
        return _git_commit_cache

    package_dir = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            [
                "git",
                "describe",
                "--always",
                "--dirty",
                "--abbrev=12",
            ],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        _git_commit_cache = (
            result.stdout.strip() if result.returncode == 0 else None
        )
    # A missing git, a permission problem and a hung filesystem all
    # mean the same thing here -- no commit to record -- and none of
    # them should break a save.
    except Exception:  # noqa: BLE001
        _git_commit_cache = None

    return _git_commit_cache


def package_version():
    """Return ``sigmutsel.__version__``, or ``"unknown"``."""
    import sigmutsel

    return getattr(sigmutsel, "__version__", "unknown")


def package_provenance():
    """Return the provenance stamp to write into a manifest."""
    return {
        "sigmutsel_version": package_version(),
        "sigmutsel_commit": git_commit(),
        "saved_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def check_provenance(manifest, kind, directory=None):
    """Warn if `manifest` was written by a different build.

    Parameters
    ----------
    manifest : dict
        The manifest read back from disk.
    kind : str
        What is being loaded, for the message (``"model"``,
        ``"dataset"``).
    directory : str or pathlib.Path, optional
        Where it was loaded from, for the message.

    Notes
    -----
    The comparison prefers the commit over the version, since the
    version is frozen at install time in an editable install (see
    the module docstring). A manifest written before stamping
    existed carries neither field and is passed silently: it is an
    older file, not a mismatch.
    """
    saved_version = manifest.get("sigmutsel_version")
    saved_commit = manifest.get("sigmutsel_commit")

    if saved_version is None and saved_commit is None:
        return

    current_commit = git_commit()
    current_version = package_version()

    if saved_commit is not None and current_commit is not None:
        if saved_commit == current_commit:
            return
        saved_stamp, current_stamp = saved_commit, current_commit
    elif saved_version != current_version:
        saved_stamp, current_stamp = saved_version, current_version
    else:
        return

    where = f" from {directory}" if directory is not None else ""
    logger.warning(
        f"This {kind}{where} was written by sigmutsel "
        f"{saved_stamp}; the running build is {current_stamp}. "
        "Results loaded from it were produced by different code. "
        "See its manifest's run_history for the calls that made it."
    )


def _summarize(value):
    """Return a small JSON-safe stand-in for an argument value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_ARG_CHARS:
            return value[:MAX_ARG_CHARS] + "..."
        # A NaN or an infinity would serialize to invalid JSON.
        if isinstance(value, float) and not float(
            "-inf"
        ) < value < float("inf"):
            return repr(value)
        return value

    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) <= 5:
            return [_summarize(item) for item in value]
        return f"<{type(value).__name__} of {len(value)}>"

    if isinstance(value, dict):
        if len(value) <= 5:
            return {
                str(key): _summarize(item)
                for key, item in value.items()
            }
        return f"<dict of {len(value)}>"

    if isinstance(value, Path):
        return _summarize(str(value))

    shape = getattr(value, "shape", None)
    if shape is not None:
        return f"<{type(value).__name__} {tuple(shape)}>"

    return f"<{type(value).__name__}>"


def _history(obj):
    """Return `obj`'s run history, creating it if absent."""
    history = getattr(obj, "_run_history", None)
    if history is None:
        history = []
        obj._run_history = history
    return history


def _append(history, entry):
    """Append `entry`, eliding the middle past the cap."""
    history.append(entry)

    if len(history) > MAX_RUN_HISTORY:
        marker_index = MAX_RUN_HISTORY // 2
        marker = history[marker_index]
        if not (isinstance(marker, dict) and "elided" in marker):
            marker = {"elided": 0}
            history.insert(marker_index, marker)
        history.pop(marker_index + 1)
        marker["elided"] += 1

    return entry


def record_event(obj, call, **fields):
    """Append a history entry for something that is not a call."""
    entry = {
        "call": call,
        "at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }
    entry.update(fields)
    return _append(_history(obj), entry)


def record_call(func):
    """Record calls to a state-mutating method in ``_run_history``.

    Only the arguments the caller actually passed are recorded --
    defaults are left out, so the entry reads like the call that was
    written, the way cancereffectsizeR's ``match.call()`` history
    does. Large values (frames, arrays) are replaced by a shape
    stand-in.

    The entry is appended *before* the method runs, and gains a
    ``"failed"`` field if it raises, so a call that died still
    leaves a trace.
    """
    signature = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
            recorded = {
                name: _summarize(value)
                for name, value in list(bound.arguments.items())[1:]
            }
        # A bad call is about to raise from func itself with a far
        # better message; record what we can and let it through.
        except TypeError:
            recorded = {
                "args": _summarize(args),
                "kwargs": _summarize(kwargs),
            }

        entry = record_event(self, func.__qualname__, args=recorded)

        try:
            return func(self, *args, **kwargs)
        except BaseException as exc:
            entry["failed"] = type(exc).__name__
            raise

    return wrapper


def format_run_history(history):
    """Return `history` as lines, for printing or logging."""
    lines = []
    for entry in history or []:
        if "elided" in entry:
            lines.append(f"... {entry['elided']} entries elided ...")
            continue
        args = ", ".join(
            f"{name}={value!r}"
            for name, value in (entry.get("args") or {}).items()
        )
        failed = (
            f"  -- raised {entry['failed']}"
            if entry.get("failed")
            else ""
        )
        lines.append(
            f"{entry.get('at', '?')}  {entry['call']}({args}){failed}"
        )
    return lines
