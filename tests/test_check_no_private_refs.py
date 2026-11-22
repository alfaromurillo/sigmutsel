"""The private-reference checker in scripts/ (no names restated
here: they come from the checker's own lists)."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "check_no_private_refs.py"
)


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names(checker):
    return checker.BANNED_REPOS + checker.BANNED_HOSTS


def test_lists_cover_repos_and_machines(checker):
    assert len(checker.BANNED_REPOS) >= 3
    assert len(checker.BANNED_HOSTS) >= 4


@pytest.mark.parametrize(
    "template",
    [
        "# see {name}'s notes",  # a comment
        '    """Built on {name}."""',  # a docstring
        '  "description": "after {name} ran out of disk"',  # JSON
        "path = '{name}/data'",
        "{name}_cache = 1",  # inside a snake_case identifier
        "x = 1  # {upper}",  # any case
        "{title} filled to 45G",
    ],
)
def test_every_name_is_caught(checker, template):
    for name in _names(checker):
        line = template.format(
            name=name, upper=name.upper(), title=name.title()
        )
        hits = checker.scan_text("ok\n" + line + "\n")
        assert [(n, reason) for n, reason, _ in hits] == [(2, name)]


def test_longer_words_are_not_caught(checker):
    for name in _names(checker):
        assert checker.scan_text(f"{name.title()}ian smoothing") == []
        assert checker.scan_text(f"pre{name}") == []


def test_local_paths_are_caught(checker):
    home = "/" + "home"  # spelled out, the line would flag itself
    hits = checker.scan_text(f"cache = '{home}/someone/data'")
    assert [reason for _, reason, _ in hits] == ["local path"]
    assert checker.scan_text("https://example.org/home/page") == []


def test_scans_tracked_text_files_only(checker, tmp_path):
    name = checker.BANNED_HOSTS[0]
    files = {
        "src/pkg/mod.py": f'"""Docstring naming {name}."""\n',
        "tests/test_x.py": f"# {name.upper()}\n",
        "data/rows.json": f'{{"description": "{name}"}}\n',
        "docs/a.md": "clean\n",
        "setup.cfg": f"[x]\nhost = {name}\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    (tmp_path / "blob.bin").write_bytes(b"\0\1" + name.encode())
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    # untracked: not part of the package
    (tmp_path / "notes.txt").write_text(name)

    found = {
        (str(rel), lineno)
        for rel, lineno, _, _ in checker.find_violations(tmp_path)
    }
    assert found == {
        ("src/pkg/mod.py", 1),
        ("tests/test_x.py", 1),
        ("data/rows.json", 1),
        ("setup.cfg", 2),
    }


def test_repository_is_clean(checker):
    assert checker.find_violations() == []
