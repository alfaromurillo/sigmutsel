"""Run independent per-item work across processes, in order.

Gene and variant gammas and the selection table's per-gene likelihoods
are independent of one another, and each is pure NumPy (and PyTensor,
compiled per call) once the model is fitted. :func:`map_ordered` runs
them in forked worker processes, which share the fitted model
copy-on-write, so nothing large is pickled on the way in; only each
item's result comes back. The results are the serial path's exactly
when the work is deterministic: the order of the output is the order
of the input, and anything random must be drawn from a generator
seeded per item (:func:`item_rng`), never from one stream shared by
the items, whose draws would then depend on the order they ran in.

``n_jobs`` defaults to :data:`.constants.n_jobs` (environment variable
``SIGMUTSEL_N_JOBS``), which is 1: the serial path.
"""

import multiprocessing as mp
import os
import warnings
import zlib

import numpy as np

from . import constants

# The function the workers apply. A fork child inherits the parent's
# memory, so the function (often a closure over a fitted model) is
# handed over by setting this module global before the pool forks,
# rather than pickled.
_WORK = None


def numpy_backed(posterior):
    """``posterior`` with every array a NumPy array.

    A posterior straight from NumPyro holds JAX arrays, and reading
    one (``az.extract``) dispatches to JAX, which hangs in a forked
    worker. Its values are unchanged; a posterior loaded from disk is
    NumPy already. Takes a DataTree (arviz 1.x), an InferenceData
    (arviz 0.x) or None.
    """
    if posterior is None:
        return None
    if hasattr(posterior, "map_over_datasets"):
        return posterior.map_over_datasets(lambda ds: ds.as_numpy())
    if hasattr(posterior, "map"):
        return posterior.map(lambda ds: ds.as_numpy())
    return posterior


def resolve_n_jobs(n_jobs=None):
    """``n_jobs``, or :data:`.constants.n_jobs`; -1 means every core."""
    n = constants.n_jobs if n_jobs is None else n_jobs
    n = int(n)
    if n < 0:
        n = len(os.sched_getaffinity(0))
    return max(1, n)


def item_rng(seed, key):
    """A generator for one item, the same whatever order items run in.

    Seeded by ``seed`` and a CRC-32 of ``key`` (a gene id, a variant
    name, a channel key), so an item's draws do not depend on which
    items ran before it, or in which process.
    """
    if seed is None:
        return None
    return np.random.default_rng(
        [int(seed), zlib.crc32(str(key).encode())]
    )


class _Raised:
    """An exception raised by one item, carried back to the parent."""

    def __init__(self, error):
        self.error = error


def _call(index):
    try:
        return _WORK(index)
    # Any error of one item is carried back, not swallowed.
    except Exception as error:  # noqa: BLE001
        return _Raised(error)


def _call_here(func, item):
    try:
        return func(item)
    # Any error of one item is carried back, not swallowed.
    except Exception as error:  # noqa: BLE001
        return _Raised(error)


def _one_blas_thread():
    """One BLAS thread while the pool forks: the parallelism is the
    processes. Set in the parent, never in a forked child, where
    resizing a pool whose threads did not survive the fork hangs."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        import contextlib

        return contextlib.nullcontext()
    return threadpool_limits(1)


def map_ordered(
    func,
    items,
    n_jobs=None,
    return_exceptions=False,
    warm_first=True,
    chunksize=1,
):
    """``[func(x) for x in items]``, across ``n_jobs`` forked processes.

    Parameters
    ----------
    func : callable
        Applied to each item. It runs in a fork of the calling
        process, so it may close over large objects (a fitted model);
        its result must be picklable. It must not call JAX: JAX's
        threads do not survive a fork (NumPy, SciPy and PyTensor are
        fine). JAX's warning about forking is silenced for that reason.
    items : sequence
    n_jobs : int, optional
        Processes; default :data:`.constants.n_jobs`. 1 (or a single
        item) runs ``func`` in this process, the serial path.
    return_exceptions : bool, default False
        An exception raised by ``func`` on an item is returned in that
        item's place instead of raised (the first one is raised
        otherwise, after every item has run).
    warm_first : bool, default True
        Run the first item in this process before forking, so any
        cache it fills on the way (a lazily computed table of the
        model or dataset) is filled once and inherited by every
        worker instead of rebuilt in each.
    chunksize : int, default 1
        Items handed to a worker at a time: 1 for items of seconds
        (gammas), more for many items of milliseconds (genes of the
        selection table).

    Returns
    -------
    list
        One result per item, in the order of ``items``.
    """
    global _WORK
    items = list(items)
    n = min(resolve_n_jobs(n_jobs), len(items))
    if n <= 1:
        out = []
        for x in items:
            try:
                out.append(func(x))
            except Exception as error:
                if not return_exceptions:
                    raise
                out.append(error)
        return out
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "n_jobs > 1 needs the 'fork' start method (Linux)."
        )
    raw = []
    start = 0
    if warm_first:
        raw.append(_call_here(func, items[0]))
        start = 1
    _WORK = lambda i: func(items[i])
    try:
        with warnings.catch_warnings():
            # JAX's and Python's warnings that this process has
            # threads: the workers never call JAX (see above).
            warnings.filterwarnings(
                "ignore",
                message=r"os\.fork\(\) was called",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded",
                category=DeprecationWarning,
            )
            with (
                _one_blas_thread(),
                mp.get_context("fork").Pool(n) as pool,
            ):
                raw += pool.map(
                    _call,
                    range(start, len(items)),
                    chunksize=chunksize,
                )
    finally:
        _WORK = None
    out = []
    for r in raw:
        if isinstance(r, _Raised):
            if not return_exceptions:
                raise r.error
            r = r.error
        out.append(r)
    return out
