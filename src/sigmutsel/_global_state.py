"""Use SigProfiler's packages without their global side effects.

Importing them changes the whole Python session, not just their own
modules:

- SigProfilerAssignment calls ``matplotlib.use("Agg")`` and
  ``plt.switch_backend("agg")`` at import, so in Jupyter no figure is
  shown afterwards;
- SigProfilerMatrixGenerator calls ``logging.basicConfig(level=INFO)``
  at import, which gives the *root* logger a handler at INFO, so every
  library's INFO messages (sigmutsel's, PyTensor's) start printing.

:func:`keep_global_state` restores both on exit, and sigmutsel imports
and calls those packages inside it. A program that wants sigmutsel's
INFO messages configures logging itself, as any program does
(``logging.basicConfig(level=logging.INFO)``).
"""

import contextlib
import logging


def _current_backend():
    """The backend chosen so far, or None if matplotlib has not chosen."""
    import matplotlib

    try:
        return matplotlib.get_backend(auto_select=False)
    except TypeError:  # matplotlib < 3.10
        return matplotlib.rcParams._get_backend_or_none()


def _restore_backend(backend):
    import sys

    import matplotlib

    if backend is not None:
        if _current_backend() != backend:
            matplotlib.use(backend, force=True)
        return
    # Nothing was chosen before: let matplotlib choose again, as if the
    # import had not happened.
    from matplotlib import rcsetup

    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].switch_backend(
            rcsetup._auto_backend_sentinel
        )
    else:
        matplotlib.rcParams["backend"] = (
            rcsetup._auto_backend_sentinel
        )


@contextlib.contextmanager
def keep_global_state():
    """Restore the matplotlib backend and the root logger on exit.

    Examples
    --------
    >>> from sigmutsel._global_state import keep_global_state
    >>> with keep_global_state():
    ...     from SigProfilerAssignment import Analyzer  # doctest: +SKIP
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    backend = _current_backend()
    try:
        yield
    finally:
        root.handlers[:] = handlers
        root.setLevel(level)
        _restore_backend(backend)
