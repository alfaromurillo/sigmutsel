#!/usr/bin/env python3
"""Write docs/api.md, the API reference, from the code itself.

Signatures (and so every default) are read with ``inspect.signature``
and docstrings with ``inspect.getdoc``, so the reference cannot state
a default the code does not have. ``USER_API`` is the part a user of
the tutorial touches, in the order of a workflow: each entry is
documented in full. Every other public name of the package is listed
after it with its one-line summary.

Usage::

    python scripts/build_api_reference.py          # rewrite docs/api.md
    python scripts/build_api_reference.py --check  # exit 1 if stale

tests/test_api_reference.py runs the check, and checks that every
``USER_API`` entry documents its parameters, its return value and an
example.
"""

import argparse
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "docs" / "api.md"

# (section title, [dotted names]) in workflow order.
USER_API = [
    (
        "Setup and data",
        [
            "sigmutsel.setup.download_all",
            "sigmutsel.setup.install_reference_genome",
            "sigmutsel.locations.print_data_status",
            "sigmutsel.download_tcga_data.download_project_maf_files",
            "sigmutsel.download_tcga_data.download_tcga_maf_files",
            "sigmutsel.tcga_sample_selection.select_tcga_maf_files",
        ],
    ),
    (
        "Building a dataset",
        [
            "sigmutsel.models.MutationDataset",
            "sigmutsel.models.MutationDataset.generate_mutation_db",
            "sigmutsel.models.MutationDataset."
            "run_two_pass_signature_decomposition",
            "sigmutsel.models.MutationDataset.build_full_dataset",
            "sigmutsel.models.MutationDataset.save_dataset",
            "sigmutsel.models.MutationDataset.load_dataset",
            "sigmutsel.qc.load_repeat_intervals",
            "sigmutsel.setup.download_repeatmasker_bed",
        ],
    ),
    (
        "Calibration steps",
        [
            "sigmutsel.setup.download_tcga_absolute_purity",
            "sigmutsel.setup.download_tcga_absolute_segments",
            "sigmutsel.sample_qc.load_copy_number_segments_from_file",
            "sigmutsel.sample_qc.flag_low_purity_samples",
            "sigmutsel.sample_qc.flag_vaf_shape_samples",
            "sigmutsel.sample_qc.combine_sample_flags",
            "sigmutsel.models.MutationDataset.set_exclude_repeats",
            "sigmutsel.models.MutationDataset.set_exclude_overlaps",
            "sigmutsel.spectrum_shrinkage.heldout_spectrum_scores",
            "sigmutsel.spectrum_shrinkage.select_kappas",
            "sigmutsel.models.MutationDataset.set_spectrum_shrinkage",
            "sigmutsel.models.Model.estimate_site_weights",
            "sigmutsel.sample_qc.homozygously_deleted_genes",
            "sigmutsel.channel_universe.load_or_build_transcript_models",
            "sigmutsel.models.MutationDataset.build_indel_db",
        ],
    ),
    (
        "The rate model",
        [
            "sigmutsel.models.Model",
            "sigmutsel.models.Model.assign_cov_matrix",
            "sigmutsel.models.Model.compute_channel_base_mus",
            "sigmutsel.models.Model.estimate_channel_rg_cov_effects",
            "sigmutsel.models.Model.compute_r_g_for_evaluation",
            "sigmutsel.models.Model.estimate_passenger_genes_r2",
            "sigmutsel.models.Model.save_model",
            "sigmutsel.models.Model.load_model",
        ],
    ),
    (
        "Rates",
        [
            "sigmutsel.site_rates.SiteRates.from_model",
            "sigmutsel.site_rates.SiteRates.variant_rates",
            "sigmutsel.site_rates.SiteRates.snv_rates",
        ],
    ),
    (
        "Selection",
        [
            "sigmutsel.models.Model.estimate_gene_tumor_dispersion",
            "sigmutsel.models.Model.estimate_gamma",
            "sigmutsel.models.Model.estimate_gammas",
            "sigmutsel.estimate_gammas.summarize_gamma",
            "sigmutsel.models.Model.gene_selection_table",
        ],
    ),
]


def resolve(dotted):
    """The object a dotted name points to (module, class, attribute)."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = inspect.getattr_static(obj, attr)
            if isinstance(obj, (classmethod, staticmethod)):
                obj = obj.__func__
        return obj
    raise ImportError(dotted)


class _Sorted:
    """A set default, shown sorted: a set's repr follows hash order,
    which changes from run to run."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        items = ", ".join(
            repr(v) for v in sorted(self.value, key=repr)
        )
        kind = type(self.value).__name__
        return f"{kind}({{{items}}})" if items else f"{kind}()"


def _stable(param):
    if isinstance(param.default, (set, frozenset)):
        return param.replace(default=_Sorted(param.default))
    return param


def signature_of(name, obj):
    """``name(...)`` with every default as the code has it."""
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return name.split(".")[-1]
    params = [
        _stable(p)
        for p in sig.parameters.values()
        if p.name not in ("self", "cls")
    ]
    sig = sig.replace(parameters=params, return_annotation=sig.empty)
    short = name.split(".")[-1]
    text = f"{short}{sig}"
    if len(text) <= 70:
        return text
    lines = [f"{short}("]
    for p in params:
        lines.append(f"    {p},")
    lines.append(")")
    # A bare "*" separator renders as "*," on its own line: fine.
    return "\n".join(lines)


def public_names():
    """Every public module-level callable and public method."""
    import sigmutsel

    found = []
    for info in pkgutil.iter_modules(sigmutsel.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"sigmutsel.{info.name}")
        for attr, obj in vars(module).items():
            if attr.startswith("_"):
                continue
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                found.append(f"{module.__name__}.{attr}")
            elif inspect.isclass(obj):
                found.append(f"{module.__name__}.{attr}")
                for meth, member in vars(obj).items():
                    if meth.startswith("_"):
                        continue
                    if isinstance(
                        member, (classmethod, staticmethod)
                    ):
                        member = member.__func__
                    if inspect.isfunction(member):
                        found.append(
                            f"{module.__name__}.{attr}.{meth}"
                        )
    return sorted(set(found))


def summary(obj):
    doc = inspect.getdoc(obj) or ""
    return doc.strip().split("\n\n")[0].replace("\n", " ")


def render():
    out = [
        "# sigmutsel API reference",
        "",
        "Generated from the code by `scripts/build_api_reference.py`; "
        "do not edit by hand. Signatures and defaults are the code's.",
        "",
        "**Defaults are not always the recommended settings.** Several "
        "defaults are kept for older analyses; the tutorial "
        "(`tutorials/selection_in_hnsc.md`) passes the recommended ones, "
        "and the README lists them.",
        "",
        "## Contents",
        "",
    ]
    for title, _ in USER_API:
        anchor = title.lower().replace(" ", "-")
        out.append(f"- [{title}](#{anchor})")
    out += [
        "- [Every other public name](#every-other-public-name)",
        "",
    ]
    documented = set()
    for title, names in USER_API:
        out += [f"## {title}", ""]
        for name in names:
            obj = resolve(name)
            documented.add(name)
            out += [
                f"### `{name.removeprefix('sigmutsel.')}`",
                "",
                "```python",
                signature_of(name, obj),
                "```",
                "",
                "```text",
                (inspect.getdoc(obj) or "(no docstring)").rstrip(),
                "```",
                "",
            ]
    out += [
        "## Every other public name",
        "",
        "One line each; `help(name)` or the source has the rest.",
        "",
        "| name | summary |",
        "|---|---|",
    ]
    for name in public_names():
        if name in documented:
            continue
        text = summary(resolve(name)).replace("|", "\\|")
        out.append(
            f"| `{name.removeprefix('sigmutsel.')}` | {text} |"
        )
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(
                f"{OUTPUT.relative_to(REPO_ROOT)} is stale: run "
                "python scripts/build_api_reference.py"
            )
            return 1
        return 0
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
