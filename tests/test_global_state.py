"""SigProfiler's packages must not change the session's matplotlib
backend or root logger (SigProfilerAssignment switches to Agg, and
SigProfilerMatrixGenerator calls logging.basicConfig, at import)."""

import logging
import os
import subprocess
import sys


def test_keep_global_state_restores_backend_and_root_logger():
    import matplotlib
    import matplotlib.pyplot as plt

    from sigmutsel._global_state import keep_global_state

    before = matplotlib.get_backend()
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        plt.switch_backend("svg")
        with keep_global_state():
            matplotlib.use("Agg", force=True)
            logging.basicConfig(level=logging.INFO, force=True)
        assert matplotlib.get_backend() == "svg"
        assert root.handlers == handlers
        assert root.level == level
    finally:
        plt.switch_backend(before)


def test_importing_the_sigprofiler_modules_changes_nothing():
    """In a fresh interpreter, as a notebook would import them."""
    code = (
        "import logging, matplotlib\n"
        "import matplotlib.pyplot\n"
        "from sigmutsel import signature_decomposition\n"
        "from sigmutsel import mutational_matrices_generator\n"
        "root = logging.getLogger()\n"
        "print(matplotlib.get_backend(), len(root.handlers), root.level)\n"
    )
    env = {**os.environ, "MPLBACKEND": "svg"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out == ["svg", "0", str(logging.WARNING)]
