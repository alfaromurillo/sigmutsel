"""docs/api.md is generated from the code, and the user-facing API
documents its parameters, its return value and an example."""

import importlib.util
import inspect
import re
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "build_api_reference.py"
)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _user_api(builder):
    return [name for _, names in builder.USER_API for name in names]


def _section(doc, title):
    return re.search(rf"^{title}\n-{{3,}}$", doc, re.MULTILINE)


def test_api_reference_is_current():
    """Regenerate with: python scripts/build_api_reference.py

    In a fresh interpreter: the test session's fixtures patch some of
    the package's functions (tests/conftest.py), which would change
    what the reference shows.
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_user_api_name_exists(builder):
    for name in _user_api(builder):
        builder.resolve(name)


@pytest.mark.parametrize(
    "section", ["Parameters", "Returns", "Examples"]
)
def test_user_api_documents_each_section(builder, section):
    missing = []
    for name in _user_api(builder):
        obj = builder.resolve(name)
        doc = inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            # A class documents its constructor's parameters and an
            # example; what it "returns" is itself.
            if section == "Returns":
                continue
            target = obj.__init__
            doc = doc + "\n" + (inspect.getdoc(target) or "")
        else:
            target = obj
        if section == "Parameters":
            params = [
                p
                for p in inspect.signature(target).parameters
                if p not in ("self", "cls")
            ]
            if not params:
                continue
            # A function documents its parameters in its own section;
            # a constructor may document them in the class docstring.
            if not _section(doc, "Parameters"):
                missing.append(name)
        elif not _section(doc, section):
            missing.append(name)
    assert not missing, f"no {section} section: {missing}"


def test_documented_parameters_exist(builder):
    """A Parameters section names only parameters the code has."""
    wrong = []
    for name in _user_api(builder):
        obj = builder.resolve(name)
        target = obj.__init__ if inspect.isclass(obj) else obj
        doc = inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            doc += "\n" + (inspect.getdoc(target) or "")
        match = _section(doc, "Parameters")
        if not match:
            continue
        body = doc[match.end() :]
        end = re.search(r"^\w[\w ]*\n-{3,}$", body, re.MULTILINE)
        body = body[: end.start()] if end else body
        params = set(inspect.signature(target).parameters)
        for line in body.splitlines():
            m = re.match(
                r"^(\*{0,2}[A-Za-z_][\w, *]*?)\s*(:|$)", line
            )
            if not m or line.startswith(" "):
                continue
            for word in m.group(1).split(","):
                word = word.strip().lstrip("*")
                if word and word not in params:
                    wrong.append(f"{name}: {word}")
    assert not wrong, wrong
