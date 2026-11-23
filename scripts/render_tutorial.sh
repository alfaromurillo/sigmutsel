#!/bin/bash
# Run a tutorial and render it as Markdown with its outputs and
# figures: tutorials/<name>.md and tutorials/<name>_files/.
#
#   scripts/render_tutorial.sh tutorials/selection_in_hnsc.py WORKDIR
#
# WORKDIR is where the tutorial runs (it writes its data under
# WORKDIR/hnsc/); a rerun reuses what is there. Needs
# `pip install jupytext nbconvert ipykernel` in the environment that
# has sigmutsel. The notebook is executed in WORKDIR and only the
# rendered page is copied back; the .py stays the source.
set -euo pipefail
src=$(realpath "$1"); work=$(realpath "${2:-.}")
name=$(basename "$src" .py); out=$(dirname "$src")
mkdir -p "$work"
cd "$work"
jupytext --to ipynb --output "$name.ipynb" "$src"
# Progress bars (tqdm) print escape codes into the page: off.
TQDM_DISABLE=1 jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 "$name.ipynb"
jupyter nbconvert --to markdown --output-dir "$out" "$name.ipynb"
# pandas' HTML tables carry a <style scoped> block, which GitHub shows
# as text: drop it, keep the table.
python - "$out/$name.md" <<'PY'
import re, sys
from pathlib import Path
p = Path(sys.argv[1])
p.write_text(re.sub(r"<style scoped>.*?</style>\n?", "", p.read_text(), flags=re.S))
PY
echo "rendered $out/$name.md"
