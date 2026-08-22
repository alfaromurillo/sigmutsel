"""Select which downloaded TCGA MAF files to use for an analysis.

`download_tcga_data.py` fetches one MAF file per sequenced *aliquot*
from GDC's Aliquot Ensemble Somatic Variant Merging and Masking
workflow, with no sample-type filtering or per-case
deduplication -- a GDC case (patient) with multiple sequenced
aliquots (e.g. a re-plated portion, or both a primary and a
metastatic tumor) becomes multiple independent files. This module
selects a configurable subset of an already-downloaded
``all_maf_files/`` directory: which sample types to keep (barcode
based), and what to do when a case has more than one file left after
that filter (keep all, or pick one by a policy).

Conventions largely follow cancereffectsizeR's
``get_TCGA_project_MAF()``
(https://github.com/Townsend-Lab-Yale/cancereffectsizeR,
R/get_TCGA_project_MAF.R): default to keeping only sample types 01
(Primary Solid Tumor) and 03 (Primary Blood Derived Cancer), and warn
when that would gut a mostly-metastatic cohort like TCGA-SKCM.
cancereffectsizeR does not offer an oldest/newest policy for
same-type duplicates -- it instead merges them at the mutation-call
level via a shared patient identifier, which assumes duplicate
aliquots are technical re-sequencing of one specimen. Because this
pipeline's `load_maf_files.py` treats each MAF file as one
independent sample rather than merging calls across files, this
module instead selects one file per case; "oldest"/"newest" use
GDC's per-sample ``days_to_collection`` field (fetched from the GDC
cases API) rather than assuming duplicate aliquots share a collection
date.

Barcode parsing and the GDC cases-API lookup live in
`gdcfetch.tcga_barcode` (moved there so any gdcfetch-based download,
not just this pipeline's, can use them from the start) -- this module
re-exports :data:`SAMPLE_TYPE_CODES`, :class:`TcgaBarcodeInfo`, and
:func:`parse_tcga_barcode` for convenience, so callers don't need a
separate gdcfetch import for those.
"""

import logging
import random
from collections.abc import Iterable
from pathlib import Path

from gdcfetch.tcga_barcode import (
    SAMPLE_TYPE_CODES,
    TcgaBarcodeInfo,
    fetch_case_metadata,
    parse_tcga_barcode,
)

__all__ = [
    "DEFAULT_KEEP_SAMPLE_TYPES",
    "DUPLICATE_POLICIES",
    "PROJECTS_WITH_MOSTLY_NONPRIMARY_SAMPLES",
    "SAMPLE_TYPE_CODES",
    "TcgaBarcodeInfo",
    "catalog_maf_files",
    "fetch_gdc_case_metadata",
    "filter_by_sample_type",
    "parse_tcga_barcode",
    "read_maf_tumor_sample_barcode",
    "select_one_per_case",
    "select_tcga_maf_files",
]

logger = logging.getLogger(__name__)

# Matches cancereffectsizeR's get_TCGA_project_MAF() default
# (exclude_TCGA_nonprimary = TRUE): keep only primary solid tumors
# and primary blood-derived cancers.
DEFAULT_KEEP_SAMPLE_TYPES = frozenset({"01", "03"})

# TCGA projects that are predominantly non-primary-tumor samples, so
# the default filter would remove most or all of the cohort.
# cancereffectsizeR special-cases SKCM with an explicit warning for
# the same reason (get_TCGA_project_MAF.R, project == 'TCGA-SKCM'
# check) -- confirmed here empirically: 365/472 TCGA-SKCM MAF files
# audited 2026-08-21 are sample type 06 (Metastatic), only 104 are
# type 01.
PROJECTS_WITH_MOSTLY_NONPRIMARY_SAMPLES = frozenset({"SKCM"})

DUPLICATE_POLICIES = frozenset(
    {"keep_all", "oldest", "newest", "random"}
)

# Thin wrapper kept under this pipeline's historical name; the
# implementation (and its docstring) lives in gdcfetch.
fetch_gdc_case_metadata = fetch_case_metadata


def read_maf_tumor_sample_barcode(maf_file: str | Path) -> str | None:
    """Read the ``Tumor_Sample_Barcode`` of a single-aliquot MAF file.

    Reads only up to the header row plus the first data row -- these
    MAF files (from the "Aliquot Ensemble Somatic Variant Merging and
    Masking" workflow, one aliquot pair per file) always carry a
    single, constant ``Tumor_Sample_Barcode`` value, so there is no
    need to scan the whole file.

    Parameters
    ----------
    maf_file : str or pathlib.Path
        Path to the MAF file.

    Returns
    -------
    str or None
        The barcode, or ``None`` if the file has no
        ``Tumor_Sample_Barcode`` column, or has a header but no data
        rows (a real, if uncommon, case: an aliquot with zero somatic
        mutations called).
    """
    with Path(maf_file).open("r", errors="replace") as fh:
        header = None
        idx = None
        for line in fh:
            if line.startswith("#"):
                continue
            if header is None:
                header = line.rstrip("\n").split("\t")
                if "Tumor_Sample_Barcode" not in header:
                    return None
                idx = header.index("Tumor_Sample_Barcode")
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= idx:
                continue
            return fields[idx]
    return None


def catalog_maf_files(
    maf_dir: str | Path,
) -> dict[Path, TcgaBarcodeInfo]:
    """Map every ``.maf`` file in a directory to its parsed barcode.

    Parameters
    ----------
    maf_dir : str or pathlib.Path
        Directory of single-aliquot ``.maf`` files, e.g.
        ``data/<CODE>/tcga/all_maf_files/``.

    Returns
    -------
    dict[pathlib.Path, TcgaBarcodeInfo]
        One entry per MAF file that has a readable barcode. Files
        with zero data rows or an unparseable barcode are skipped
        (logged at WARNING).
    """
    maf_dir = Path(maf_dir)
    all_files = sorted(maf_dir.glob("*.maf"))
    catalog: dict[Path, TcgaBarcodeInfo] = {}
    n_skipped = 0
    for maf_file in all_files:
        barcode = read_maf_tumor_sample_barcode(maf_file)
        if barcode is None:
            logger.warning(
                f"Skipping {maf_file.name}: no data rows or no "
                "Tumor_Sample_Barcode column (likely zero somatic "
                "mutations called for this aliquot)."
            )
            n_skipped += 1
            continue
        try:
            catalog[maf_file] = parse_tcga_barcode(barcode)
        except ValueError as e:
            logger.warning(f"Skipping {maf_file.name}: {e}")
            n_skipped += 1
    if n_skipped:
        logger.info(
            f"Cataloged {len(catalog)}/{len(all_files)} MAF files in "
            f"{maf_dir} ({n_skipped} skipped)."
        )
    else:
        logger.info(
            f"Cataloged {len(catalog)} MAF files in {maf_dir}."
        )
    return catalog


def filter_by_sample_type(
    catalog: dict[Path, TcgaBarcodeInfo],
    keep_sample_types: Iterable[str] = DEFAULT_KEEP_SAMPLE_TYPES,
    *,
    project: str | None = None,
) -> dict[Path, TcgaBarcodeInfo]:
    """Keep only MAF files whose sample-type code is in *keep_sample_types*.

    Parameters
    ----------
    catalog : dict[pathlib.Path, TcgaBarcodeInfo]
        As returned by :func:`catalog_maf_files`.
    keep_sample_types : iterable of str, default {"01", "03"}
        TCGA sample-type codes to keep (see :data:`SAMPLE_TYPE_CODES`
        for the full official table).
    project : str or None
        TCGA project code (e.g. ``"SKCM"``), used only to emit a
        warning when filtering would remove most of a
        predominantly-non-primary cohort (see
        :data:`PROJECTS_WITH_MOSTLY_NONPRIMARY_SAMPLES`).

    Returns
    -------
    dict[pathlib.Path, TcgaBarcodeInfo]
        The filtered subset of *catalog*.
    """
    keep_sample_types = frozenset(keep_sample_types)
    kept = {
        path: info
        for path, info in catalog.items()
        if info.sample_type_code in keep_sample_types
    }
    n_removed = len(catalog) - len(kept)
    if n_removed:
        logger.info(
            f"Sample-type filter ({sorted(keep_sample_types)}) kept "
            f"{len(kept)}/{len(catalog)} files, removed {n_removed}."
        )
    if (
        project in PROJECTS_WITH_MOSTLY_NONPRIMARY_SAMPLES
        and len(catalog) > 0
        and len(kept) / len(catalog) < 0.5
    ):
        logger.warning(
            f"TCGA-{project} is predominantly non-primary samples "
            f"(kept only {len(kept)}/{len(catalog)} files with "
            f"sample types {sorted(keep_sample_types)}). If you want "
            "those samples included, pass a broader "
            "keep_sample_types (e.g. add '06' for Metastatic)."
        )
    return kept


def select_one_per_case(
    catalog: dict[Path, TcgaBarcodeInfo],
    policy: str = "keep_all",
    *,
    case_metadata: dict[str, dict] | None = None,
    random_seed: int | None = None,
) -> dict[Path, TcgaBarcodeInfo]:
    """Resolve cases with more than one remaining MAF file.

    Parameters
    ----------
    catalog : dict[pathlib.Path, TcgaBarcodeInfo]
        Typically the output of :func:`filter_by_sample_type`.
    policy : {"keep_all", "oldest", "newest", "random"}, default "keep_all"
        - ``"keep_all"``: no-op; every file is kept (the implicit
          status quo before this module existed).
        - ``"oldest"`` / ``"newest"``: keep the file whose
          ``samples.days_to_collection`` (from *case_metadata*, see
          :func:`fetch_gdc_case_metadata`) is smallest/largest. If a
          case's files have no usable dates (all missing, or tied),
          falls back to the first file in sorted-path order and logs
          a warning, so results stay deterministic.
        - ``"random"``: keep one file per case chosen uniformly at
          random.
    case_metadata : dict or None
        Required for ``"oldest"``/``"newest"``; as returned by
        :func:`fetch_gdc_case_metadata`. Ignored otherwise.
    random_seed : int or None
        Seed for ``"random"``. Sourced independently rather than
        from :data:`sigmutsel.constants.random_seed`, since sample
        *selection* is a data-preparation step, not part of the
        Bayesian model fit that seed governs.

    Returns
    -------
    dict[pathlib.Path, TcgaBarcodeInfo]
        Subset of *catalog* with at most one file per case (unless
        ``policy == "keep_all"``).

    Raises
    ------
    ValueError
        If *policy* is not one of :data:`DUPLICATE_POLICIES`, or if
        *policy* needs dates and *case_metadata* is ``None``.
    """
    if policy not in DUPLICATE_POLICIES:
        raise ValueError(
            f"policy must be one of {sorted(DUPLICATE_POLICIES)}, "
            f"got {policy!r}."
        )
    if policy == "keep_all":
        return dict(catalog)

    if policy in ("oldest", "newest") and case_metadata is None:
        raise ValueError(
            f"policy={policy!r} requires case_metadata "
            "(see fetch_gdc_case_metadata)."
        )

    by_case: dict[str, list[Path]] = {}
    for path, info in catalog.items():
        by_case.setdefault(info.case_id, []).append(path)

    rng = random.Random(random_seed)
    selected: dict[Path, TcgaBarcodeInfo] = {}
    n_resolved = 0
    for case_id, paths in by_case.items():
        if len(paths) == 1:
            selected[paths[0]] = catalog[paths[0]]
            continue

        n_resolved += 1
        paths = sorted(paths)  # deterministic base ordering

        if policy == "random":
            chosen = rng.choice(paths)
        else:
            sample_days = (case_metadata.get(case_id) or {}).get(
                "sample_days_to_collection", {}
            )
            dated = [
                (path, sample_days.get(catalog[path].sample_id))
                for path in paths
            ]
            dated = [
                (path, days)
                for path, days in dated
                if days is not None
            ]
            if not dated:
                logger.warning(
                    f"No days_to_collection available for case "
                    f"{case_id}'s {len(paths)} files; keeping "
                    f"{paths[0].name} (arbitrary, sorted-path "
                    f"tie-break) for policy={policy!r}."
                )
                chosen = paths[0]
            else:
                reverse = policy == "newest"
                chosen = sorted(
                    dated, key=lambda pd: pd[1], reverse=reverse
                )[0][0]

        selected[chosen] = catalog[chosen]

    if n_resolved:
        logger.info(
            f"Duplicate policy {policy!r} resolved {n_resolved} "
            f"multi-file case(s) to one file each."
        )
    return selected


def select_tcga_maf_files(
    maf_dir: str | Path,
    *,
    keep_sample_types: Iterable[str] = DEFAULT_KEEP_SAMPLE_TYPES,
    duplicate_policy: str = "keep_all",
    exclude_prior_treatment: bool = False,
    project: str | None = None,
    random_seed: int | None = None,
) -> list[Path]:
    """Select which MAF files in an ``all_maf_files/`` directory to use.

    Combines sample-type filtering, per-case duplicate resolution,
    and (optionally) prior-treatment exclusion into one call. See the
    module docstring for the rationale, and
    :func:`fetch_gdc_case_metadata` for what GDC actually reports
    (dates and treatment flags are often missing).

    Parameters
    ----------
    maf_dir : str or pathlib.Path
        Directory of single-aliquot ``.maf`` files.
    keep_sample_types : iterable of str, default {"01", "03"}
        See :func:`filter_by_sample_type`.
    duplicate_policy : {"keep_all", "oldest", "newest", "random"}, default "keep_all"
        See :func:`select_one_per_case`. ``"oldest"``/``"newest"``
        trigger a GDC API call (needs *project* only for logging
        context, not for the query itself).
    exclude_prior_treatment : bool, default False
        If ``True``, drop every file belonging to a case whose GDC
        ``diagnoses.prior_treatment`` is ``"Yes"``. This is a
        case-level flag, not sample-level -- GDC does not reliably
        record which specific aliquot was collected relative to a
        treatment's start date (see :func:`fetch_gdc_case_metadata`),
        so this cannot distinguish a pre-treatment primary resection
        from a post-treatment sample within the same case. Cases with
        an unreported (``None``) prior_treatment status are kept.
    project : str or None
        TCGA project code (e.g. ``"SKCM"``), used only for the
        mostly-non-primary-cohort warning in
        :func:`filter_by_sample_type`.
    random_seed : int or None
        Seed for ``duplicate_policy="random"``.

    Returns
    -------
    list[pathlib.Path]
        Selected MAF file paths, sorted.
    """
    catalog = catalog_maf_files(maf_dir)
    catalog = filter_by_sample_type(
        catalog, keep_sample_types, project=project
    )

    needs_gdc = (
        duplicate_policy in ("oldest", "newest")
        or exclude_prior_treatment
    )
    case_metadata = None
    if needs_gdc:
        case_ids = sorted({info.case_id for info in catalog.values()})
        case_metadata = fetch_gdc_case_metadata(case_ids)

    if exclude_prior_treatment:
        treated_cases = {
            case_id
            for case_id, meta in (case_metadata or {}).items()
            if meta.get("prior_treatment") == "Yes"
        }
        if treated_cases:
            n_before = len(catalog)
            catalog = {
                path: info
                for path, info in catalog.items()
                if info.case_id not in treated_cases
            }
            logger.info(
                f"Excluded {n_before - len(catalog)} file(s) from "
                f"{len(treated_cases)} case(s) with reported prior "
                "treatment."
            )

    selected = select_one_per_case(
        catalog,
        duplicate_policy,
        case_metadata=case_metadata,
        random_seed=random_seed,
    )
    return sorted(selected.keys())
