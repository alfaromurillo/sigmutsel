"""Use SigProfiler's packages without their global side effects.

Importing them changes the whole Python session, not just their own
modules:

- SigProfilerAssignment calls ``matplotlib.use("Agg")`` and
  ``plt.switch_backend("agg")`` at import, so in Jupyter no figure is
  shown afterwards;
- SigProfilerMatrixGenerator calls ``logging.basicConfig(level=INFO)``
  at import, which gives the *root* logger a handler at INFO, so every
  library's INFO messages (sigmutsel's, PyTensor's) start printing.

:func:`keep_global_state` blocks the backend switch while it is active
(``matplotlib.use`` and ``pyplot.switch_backend`` do nothing; restoring
the backend afterwards is not enough in Jupyter, where switching to Agg
unregisters the figure display that switching back does not
re-register) and restores the root logger on exit; sigmutsel imports
and calls those packages inside it. sigmutsel never asks them for
plots. A program that wants sigmutsel's
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


def _no_switch(*args, **kwargs):
    """Stands in for a backend switch inside :func:`keep_global_state`."""


@contextlib.contextmanager
def keep_global_state():
    """Keep the matplotlib backend; restore the root logger on exit.

    Examples
    --------
    >>> from sigmutsel._global_state import keep_global_state
    >>> with keep_global_state():
    ...     from SigProfilerAssignment import Analyzer  # doctest: +SKIP
    """
    import matplotlib
    import matplotlib.pyplot as plt

    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    backend = _current_backend()
    use, switch = matplotlib.use, plt.switch_backend
    matplotlib.use = plt.switch_backend = _no_switch
    try:
        yield
    finally:
        matplotlib.use, plt.switch_backend = use, switch
        root.handlers[:] = handlers
        root.setLevel(level)
        # In case the backend was changed another way (rcParams).
        _restore_backend(backend)
