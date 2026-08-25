"""Signature decomposition."""

import logging
import shutil
from pathlib import Path

import pandas as pd
from SigProfilerAssignment import Analyzer

from .constants import (
    ARTIFACT_SIGNATURES,
    TREATMENT_ASSOCIATED_SIGNATURES,
)
from .locations import (
    location_exclusion_signatures_matrix,
    location_inclusion_signatures_matrix,
)

logger = logging.getLogger(__name__)


def _normalize_signature_group_arg(arg, default_matrix):
    """Normalize signature subgroup arguments.

    Accepts:
    - None: returned as-is
    - tuple(path, cancer_type): returned unchanged
    - str: treated as a cancer type using the provided default matrix
    - other iterables (lists of signatures): returned unchanged
    """
    if arg is None:
        return None
    if isinstance(arg, tuple):
        return arg
    if isinstance(arg, str):
        return (default_matrix, arg)
    return arg


def _match_cancer_type_rows(table_df, cancer_type):
    """Return every row of a per-cancer-type table matching cancer_type.

    Matches on an exact "PCAWG" label, or membership in the
    comma-separated "Applicable_TCGA" column. Returns *all* matching
    rows rather than assuming a single match: most TCGA codes map to
    exactly one PCAWG row, but "ESCA" ambiguously matches both
    "Eso-AdenoCA" and "Eso-SCC" (esophageal adenocarcinoma and
    squamous-cell carcinoma are separate rows with different
    signature profiles). Silently taking the first match here used to
    mean ESCA-derived cohorts always got Eso-AdenoCA's profile, never
    Eso-SCC's -- callers that know which histology they mean (e.g.
    the ESCA-EAC / ESCA-ESCC TCGA subcohorts) should pass the exact
    PCAWG label ("Eso-AdenoCA" / "Eso-SCC") rather than the ambiguous
    TCGA code, which resolves unambiguously via the exact-match branch
    above. Callers stuck with an ambiguous code (e.g. bare "ESCA",
    covering both histologies) get every matching row back and must
    decide how to combine them (see `_signatures_from_rows`).

    Raises
    ------
    ValueError
        If no row matches cancer_type.
    """

    def _row_matches(row):
        if row["PCAWG"] == cancer_type:
            return True
        tcga_val = row["Applicable_TCGA"]
        if pd.notna(tcga_val):
            tcga_types = [t.strip() for t in str(tcga_val).split(",")]
            if cancer_type in tcga_types:
                return True
        return False

    mask = table_df.apply(_row_matches, axis=1)
    rows = table_df[mask]
    if rows.empty:
        raise ValueError(
            f"Cancer type '{cancer_type}' not found in table"
        )
    if len(rows) > 1:
        matched_labels = rows["PCAWG"].tolist()
        logger.warning(
            f"Cancer type '{cancer_type}' matches more than one PCAWG "
            f"row ({matched_labels}) -- combining across rows rather "
            "than arbitrarily picking one. Pass the exact PCAWG label "
            "instead of the ambiguous TCGA code if you mean only one "
            "of these."
        )
    return rows


def _signatures_from_rows(rows, sig_cols, table_semantics):
    """Resolve the set of signatures marked 1 across one or more rows.

    Parameters
    ----------
    rows : pd.DataFrame
        Matched rows from `_match_cancer_type_rows`.
    sig_cols : Iterable[str]
        Signature column names to check.
    table_semantics : {"inclusion", "exclusion"}
        How to combine multiple matched rows (only matters when more
        than one row matched, e.g. bare "ESCA"):
        - "inclusion" (value=1 means "keep this signature"): union
          across rows -- permissive, a signature is plausible for the
          combined cohort if *any* matched histology supports it.
        - "exclusion" (value=1 means "drop this signature"): a
          signature is dropped only if *every* matched row marks it
          for exclusion (intersection) -- conservative, so a signature
          that's real for one histology in a mixed cohort doesn't get
          excluded because it happens to be marked absent for another.
          This is the mathematical complement of the "inclusion" rule
          above (confirmed elsewhere the two tables are exact
          complements of each other), so the two rules agree exactly
          when applied to the same underlying information.
    """
    if table_semantics not in ("inclusion", "exclusion"):
        raise ValueError(
            "table_semantics must be 'inclusion' or 'exclusion', got "
            f"{table_semantics!r}"
        )
    per_row_sets = [
        {sig for sig in sig_cols if row[sig] == 1}
        for _, row in rows.iterrows()
    ]
    if table_semantics == "inclusion":
        combined = set().union(*per_row_sets)
    else:
        combined = set(sig_cols).intersection(*per_row_sets)
    # Preserve sig_cols order for readability/determinism.
    return [sig for sig in sig_cols if sig in combined]


def _expand_subvariants(base_sigs, available_sigs):
    """Expand base signature names to include lettered subvariants.

    e.g. "SBS10" -> ["SBS10", "SBS10a", "SBS10b", "SBS10c", "SBS10d"]
    if those subvariants are present in available_sigs. Handles COSMIC
    version-driven splits (SBS22 -> SBS22a/b/c, SBS40 -> SBS40a/b/c)
    transparently, since it only depends on what's actually in
    available_sigs, not a hardcoded split table.
    """
    expanded = []
    for base_sig in base_sigs:
        expanded.append(base_sig)
        for avail_sig in available_sigs:
            if (
                avail_sig.startswith(base_sig)
                and len(avail_sig) > len(base_sig)
                and avail_sig[len(base_sig)]
                in "abcdefghijklmnopqrstuvwxyz"
            ):
                expanded.append(avail_sig)
    return expanded


def resolve_exclusion_list(
    cancer_type,
    location=None,
    available_sigs=None,
    treatment_naive=True,
    exclude_artifacts=False,
):
    """Resolve the final list of signatures to exclude for a cancer type.

    Combines three independent sources, per the project's design
    (biological tissue-absence and treatment-association are
    signature-level exclusion decisions; technical artifacts are
    handled separately via mutation-level removal, not signature
    exclusion -- see sigmutsel.qc.flag_artifact_signature_mutations):

    1. The per-cancer-type table's excluded signatures (via
       `_match_cancer_type_rows` / `_signatures_from_rows`, so
       ambiguous TCGA-code matches like "ESCA" are combined
       conservatively rather than silently picking one row).
    2. `TREATMENT_ASSOCIATED_SIGNATURES`, unioned in only if
       `treatment_naive` is True.
    3. `ARTIFACT_SIGNATURES`, unioned in only if `exclude_artifacts`
       is True.

    Regardless of (2) and (3), any signature in `ARTIFACT_SIGNATURES`
    contributed by the per-cancer-type table itself (source 1) is
    always stripped back out before combining -- the table can mark
    an artifact signature "excluded" for a cancer type simply because
    Extended Data Fig. 5-style attribution wouldn't show real signal
    for a technical artifact in most tissues anyway, which would
    otherwise let an artifact signature end up excluded "by accident"
    via the table even when `exclude_artifacts=False` (e.g. during
    pass A of the two-pass artifact-detection procedure, where
    artifact signatures must stay in the fitting basis). Whether an
    artifact signature is excluded is always explicitly attributable
    to `exclude_artifacts`, never to the table.

    Parameters
    ----------
    cancer_type : str
        Cancer type / TCGA code / PCAWG label to look up.
    location : str or Path, optional
        Path to the exclusion matrix. Defaults to
        `location_exclusion_signatures_matrix`.
    available_sigs : list of str, optional
        Signatures to expand subvariants against (see
        `_expand_subvariants`). If None, subvariant expansion uses the
        table's own signature columns.
    treatment_naive : bool, default True
        Whether to also exclude `TREATMENT_ASSOCIATED_SIGNATURES`.
    exclude_artifacts : bool, default False
        Whether to also exclude `ARTIFACT_SIGNATURES`. Leave False for
        pass A of the two-pass artifact-detection procedure (artifacts
        must stay in the basis); set True for pass B (the final fit).

    Returns
    -------
    list of str
        Final exclusion list, with subvariants expanded.
    """
    if location is None:
        location = location_exclusion_signatures_matrix
    location = Path(location)
    exclusion_df = pd.read_csv(location, sep="\t")
    sig_cols = list(exclusion_df.columns[4:])

    rows = _match_cancer_type_rows(exclusion_df, cancer_type)
    table_excluded = set(
        _signatures_from_rows(rows, sig_cols, "exclusion")
    )
    # Never let the table itself exclude an artifact signature --
    # that's exclude_artifacts's job, explicitly.
    table_excluded -= set(ARTIFACT_SIGNATURES)

    combined = set(table_excluded)
    if treatment_naive:
        combined |= set(TREATMENT_ASSOCIATED_SIGNATURES)
    if exclude_artifacts:
        combined |= set(ARTIFACT_SIGNATURES)

    if available_sigs is None:
        available_sigs = sig_cols
    return _expand_subvariants(sorted(combined), available_sigs)


def run_signature_decomposition(
    samples,
    output,
    signatures=None,
    signature_database=None,
    nnls_add_penalty=0.05,
    nnls_remove_penalty=0.01,
    initial_remove_penalty=0.05,
    genome_build="GRCh38",
    cosmic_version=3.4,
    make_plots=False,
    collapse_to_SBS96=True,
    connected_sigs=True,
    verbose=False,
    devopts=None,
    exclude_signature_subgroups=None,
    include_signature_subgroups=None,
    treatment_naive=True,
    exclude_artifacts=False,
    exome=True,
    input_type="vcf",
    context_type="96",
    export_probabilities=True,
    export_probabilities_per_mutation=True,
    sample_reconstruction_plots=False,
    volume=None,
):
    """Fits COSMIC mutational signatures to input mutation data.

    This function assigns known mutational signatures (e.g., COSMIC
    signatures) to one or more tumor samples using different types of
    input mutation data.

    This function wraps :func:`Analyzer.cosmic_fit` with additional
    features for cancer-type-specific signature filtering. Key
    differences include:
    - Support for `exclude_signature_subgroups` as a tuple
      (matrix_file, cancer_type) to exclude signatures based on a
      cancer-type-specific matrix
    - Support for `include_signature_subgroups` as a tuple
      (matrix_file, cancer_type) to include only specified signatures
      based on a cancer-type-specific matrix
    - Automatic expansion of base signatures to include subvariants
      (e.g., SBS10 expands to SBS10a, SBS10b, SBS10c, SBS10d)
    - Changed default values for some parameters (e.g., genome_build,
      exome, make_plots, input_type)

    Parameters
    ----------
    samples : str
        Path to the input somatic mutations file (if using a
        segmentation file or mutational matrix) or folder (if using
        mutation calling files).

    output : str
        Path to the output folder.

    signatures : array-like, optional
        Set of known mutational signatures to use in the fit. If None,
        default COSMIC signatures will be used.

    signature_database : str, optional
        Path to a custom signature matrix file (tab-delimited), where
        rows are mutation types and columns are signature IDs. Only
        used if COSMIC reference signatures are not used.

    nnls_add_penalty : float, optional
        Penalty for adding new signatures during fitting. Default is 0.05.

    nnls_remove_penalty : float, optional
        Penalty for removing signatures during fitting. Default is 0.01.

    initial_remove_penalty : float, optional
        Initial penalty for signature removal. Default is 0.05.

    genome_build : str, optional
        Reference genome build used to align mutations and select
        COSMIC signatures.  Supported options: {'GRCh37', 'GRCh38',
        'mm9', 'mm10', 'rn6'}. Default is 'GRCh38' (different from
        Analyzer.cosmic_fit).

    cosmic_version : float, optional
        COSMIC signature version to use. Valid options include 1, 2,
        3, 3.1, 3.2, and 3.3.  Default is 3.4.

    make_plots : bool, optional
        Whether to generate and save plots. Default is False
        (different from Analyzer.cosmic_fit).

    collapse_to_SBS96 : bool, optional
        Whether to collapse input mutations to SBS96 format. Default
        is True. If `input_type` is 'ID' or 'DINUC', this setting has
        no effect, it is always set as False.

    connected_sigs : bool, optional
        Whether to use connected signature groups during
        fitting. Default is True.

    verbose : bool, optional
        Whether to print detailed output messages. Default is False.

    devopts : dict, optional
        Developer options (internal use).

    exclude_signature_subgroups : list of str, str, or tuple, optional
        List of COSMIC signature subgroups to exclude from
        fitting. Alternatively, provide a cancer type string
        (e.g., "COAD") to automatically look it up in the default
        exclusion matrix located at
        :data:`locations.location_exclusion_signatures_matrix`, or a
        tuple (location, cancer_type) where location is a Path or str
        to a custom matrix file and cancer_type is the cancer type to
        look up.
        When a tuple is provided, signatures marked as 1 in the
        matrix row for the specified cancer_type will be excluded.
        Base signatures are automatically expanded to include
        subvariants (e.g., SBS10 excludes SBS10a, SBS10b, etc.).
        Only applies when using COSMIC reference signatures.
        Default is None.

    include_signature_subgroups : list of str, str, or tuple, optional
        List of COSMIC signature subgroups to include in
        fitting (all others will be excluded). Alternatively,
        provide a cancer type string (e.g., "COAD") to automatically
        look it up in the default inclusion matrix located at
        :data:`locations.location_inclusion_signatures_matrix`, or a
        tuple (location, cancer_type) where location is a
        Path or str to a custom inclusion matrix file and
        cancer_type is the cancer type to look up. When a
        tuple is provided, only signatures marked as 1 in the
        matrix row for the specified cancer_type will be
        fitted. Base signatures are automatically expanded to
        include subvariants (e.g., SBS10 includes SBS10a,
        SBS10b, SBS10c, SBS10d). Cannot be used together with
        exclude_signature_subgroups; providing both will raise
        a ValueError. Only applies when using COSMIC reference
        signatures. Default is None.

    treatment_naive : bool, optional
        Only applies when `exclude_signature_subgroups` resolves from
        a cancer-type table (str or tuple form). If True (default),
        also excludes `constants.TREATMENT_ASSOCIATED_SIGNATURES`
        (signatures COSMIC's own aetiology attributes to prior
        chemotherapy/treatment exposure) -- appropriate when samples
        are known treatment-naive. Set False for data that may include
        treated samples.

    exclude_artifacts : bool, optional
        Only applies when `exclude_signature_subgroups` resolves from
        a cancer-type table. If True, also excludes
        `constants.ARTIFACT_SIGNATURES` (signatures COSMIC's own
        aetiology flags as likely sequencing artifacts). Default is
        False: artifact signatures are ordinarily handled by
        mutation-level removal (see `sigmutsel.qc`), not signature
        exclusion, so they should stay in the fitting basis unless a
        caller has already run the mutation-level artifact-detection
        pass and wants a final refit with artifacts excluded.

    exome : bool, optional
        Whether to use exome-normalized COSMIC signatures. Default is
        True, since most of our data will be WES (different from
        Analyzer.cosmic_fit).

    input_type : str, optional
        Type of input data. Options include:
            - 'vcf': mutation calling files (VCF, MAF, etc.)
            - 'seg:TYPE': segmentation file, where TYPE is one of
              {'ASCAT', 'ASCAT_NGS', 'SEQUENZA', 'ABSOLUTE',
              'BATTENBERG', 'FACETS', 'PURPLE', 'TCGA'}
            - 'matrix': pre-computed mutational matrix

        Default is 'vcf' (different from Analyzer.cosmic_fit).

    context_type : str, optional
        Required if `input_type` is 'vcf'. Contextual resolution of
        mutation types.  Valid options: {'96', '288', '1536', 'DINUC',
        'ID'}. Default is '96' (alias 'SNP').

    export_probabilities : bool, optional
        Whether to export the probability matrix per context for all
        samples. Default is True.

    export_probabilities_per_mutation : bool, optional
        Whether to export the probability matrix per individual
        mutation.  Only available for `input_type='vcf'`. Default is
        True (different from Analyzer.cosmic_fit).

    sample_reconstruction_plots : {'pdf', 'png', 'both', None}, optional
        Format for exporting reconstruction plots per sample. Default
        is None.

    volume : str or None, optional
        Volume label or path for storing outputs in a specific volume
        (cloud or cluster setting).

    Returns
    -------
    None
        All results are saved to the specified output directory.

    """
    context_type = context_type.upper()

    if context_type == "SNP":
        context_type = "96"
    if context_type == "INDEL":  # alias
        context_type = "ID"
    if (context_type == "ID") or (context_type == "DINUC"):
        collapse_to_SBS96 = False

    # Normalize shorthand cancer-type arguments
    exclude_signature_subgroups = _normalize_signature_group_arg(
        exclude_signature_subgroups,
        location_exclusion_signatures_matrix,
    )
    include_signature_subgroups = _normalize_signature_group_arg(
        include_signature_subgroups,
        location_inclusion_signatures_matrix,
    )

    # Check that both exclude and include are not provided
    if (
        exclude_signature_subgroups is not None
        and include_signature_subgroups is not None
    ):
        raise ValueError(
            "Cannot provide both exclude_signature_subgroups and "
            "include_signature_subgroups. Choose one: "
            "- Use exclude_signature_subgroups to exclude specific "
            "signatures and keep all others "
            "- Use include_signature_subgroups to keep only specific "
            "signatures and exclude all others"
        )

    # Process exclude_signature_subgroups if it's a tuple
    if (
        exclude_signature_subgroups is not None
        and isinstance(exclude_signature_subgroups, tuple)
        and len(exclude_signature_subgroups) == 2
    ):

        location, cancer_type = exclude_signature_subgroups

        if signatures is not None:
            available_sigs = list(signatures)
        elif signature_database is not None:
            sig_db = pd.read_csv(signature_database, sep="\t")
            available_sigs = sig_db.columns[1:].tolist()
        else:
            available_sigs = None  # resolve_exclusion_list uses the
            # exclusion table's own columns in this case

        exclude_signature_subgroups = resolve_exclusion_list(
            cancer_type,
            location=location,
            available_sigs=available_sigs,
            treatment_naive=treatment_naive,
            exclude_artifacts=exclude_artifacts,
        )

    # Process include_signature_subgroups if it's a tuple
    # (sets the signatures parameter to the included list)
    if (
        include_signature_subgroups is not None
        and isinstance(include_signature_subgroups, tuple)
        and len(include_signature_subgroups) == 2
    ):

        location, cancer_type = include_signature_subgroups
        location = Path(location)

        inclusion_df = pd.read_csv(location, sep="\t")
        sig_cols = list(inclusion_df.columns[4:])

        rows = _match_cancer_type_rows(inclusion_df, cancer_type)
        included_sigs = _signatures_from_rows(
            rows, sig_cols, "inclusion"
        )

        if signatures is not None:
            available_sigs = list(signatures)
        elif signature_database is not None:
            sig_db = pd.read_csv(signature_database, sep="\t")
            available_sigs = sig_db.columns[1:].tolist()
        else:
            available_sigs = sig_cols

        expanded_included = _expand_subvariants(
            included_sigs, available_sigs
        )

        # Set signatures to the included list
        # (this is more reliable than using exclude)
        signatures = expanded_included
        logger.info(
            f"include_signature_subgroups for cancer type "
            f"'{cancer_type}': restricting to "
            f"{len(expanded_included)} signatures"
        )
        logger.debug(f"Signatures to include: {expanded_included}")

    if exclude_signature_subgroups is not None:
        how_many_sigs = (
            f"{len(exclude_signature_subgroups)}"
            if isinstance(exclude_signature_subgroups, list)
            else "unknown"
        )
        logger.info(
            "Passing exclude_signature_subgroups with "
            f"{how_many_sigs} signatures"
        )

    Analyzer.cosmic_fit(
        samples=samples,
        output=output,
        signatures=signatures,
        signature_database=signature_database,
        nnls_add_penalty=nnls_add_penalty,
        nnls_remove_penalty=nnls_remove_penalty,
        initial_remove_penalty=initial_remove_penalty,
        genome_build=genome_build,
        cosmic_version=cosmic_version,
        make_plots=make_plots,
        collapse_to_SBS96=collapse_to_SBS96,
        connected_sigs=connected_sigs,
        verbose=verbose,
        devopts=devopts,
        exclude_signature_subgroups=exclude_signature_subgroups,
        exome=exome,
        input_type=input_type,
        context_type=context_type,
        export_probabilities=export_probabilities,
        export_probabilities_per_mutation=export_probabilities_per_mutation,
        sample_reconstruction_plots=sample_reconstruction_plots,
        volume=volume,
    )


def signature_decomposition(
    results_dir: str,
    input_data: str,
    force_generation: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load or generate signature decomposition results.

    Loads existing signature assignment results if present, otherwise
    runs signature decomposition and then load the results.

    Parameters
    ----------
    results_dir : str
        Path to the base directory where results are saved or will be
        saved.

    input_data : str
        Path to the input data (e.g., mutation matrix or VCF/MAF
        directory).

    force_generation : bool
        If True, deletes existing Assignment_Solution results and
        re-runs decomposition.

    **kwargs : dict
        Additional keyword arguments to be passed to
        `run_signature_decomposition`.

    Returns
    -------
    results : pd.DataFrame
        The loaded or newly computed signature decomposition results.

    """
    if "exclude_signature_subgroups" in kwargs:
        kwargs["exclude_signature_subgroups"] = (
            _normalize_signature_group_arg(
                kwargs["exclude_signature_subgroups"],
                location_exclusion_signatures_matrix,
            )
        )
    if "include_signature_subgroups" in kwargs:
        kwargs["include_signature_subgroups"] = (
            _normalize_signature_group_arg(
                kwargs["include_signature_subgroups"],
                location_inclusion_signatures_matrix,
            )
        )

    results_path = Path(results_dir)
    solution_dir = results_path / "Assignment_Solution"

    results_file = (
        solution_dir
        / "Activities"
        / "Assignment_Solution_Activities.txt"
    )

    if force_generation and solution_dir.exists():
        logger.info(
            f"Deleting previous signature decomposition from {solution_dir}"
        )
        shutil.rmtree(solution_dir)

    if not solution_dir.exists():
        logger.info(
            "Running signature decomposition for all tumors..."
        )
        # This will create the solution_dir if it is not there
        # Convert input_data to string in case it's a Path object
        run_signature_decomposition(
            str(input_data), str(results_path), **kwargs
        )
    else:
        logger.info(
            "Loading signature decomposition for all tumors..."
        )

    assignments = pd.read_csv(results_file, sep="\t")
    assignments = assignments.set_index("Samples")

    # Filter assignments based on include/exclude parameters
    # (post-processing since SigProfilerAssignment ignores
    # these parameters)
    if "include_signature_subgroups" in kwargs:
        include_param = kwargs["include_signature_subgroups"]
        if (
            isinstance(include_param, tuple)
            and len(include_param) == 2
        ):
            location, cancer_type = include_param
            location = Path(location)

            inclusion_df = pd.read_csv(location, sep="\t")
            sig_cols = list(inclusion_df.columns[4:])
            available_sigs = [
                col
                for col in assignments.columns
                if col.startswith("SBS")
            ]

            rows = _match_cancer_type_rows(inclusion_df, cancer_type)
            included_sigs = _signatures_from_rows(
                rows, sig_cols, "inclusion"
            )
            expanded_included = _expand_subvariants(
                included_sigs, available_sigs
            )

            sigs_to_keep = [
                col
                for col in assignments.columns
                if not col.startswith("SBS")
                or col in expanded_included
            ]
            n_excluded = len(available_sigs) - len(
                [c for c in sigs_to_keep if c.startswith("SBS")]
            )
            assignments = assignments[sigs_to_keep]
            logger.info(
                f"Filtered assignments to "
                f"{len(sigs_to_keep)} included "
                f"signatures (excluded "
                f"{n_excluded} signatures) "
                f"for cancer type '{cancer_type}'"
            )

    elif "exclude_signature_subgroups" in kwargs:
        exclude_param = kwargs["exclude_signature_subgroups"]
        if (
            isinstance(exclude_param, tuple)
            and len(exclude_param) == 2
        ):
            location, cancer_type = exclude_param
            available_sigs = [
                col
                for col in assignments.columns
                if col.startswith("SBS")
            ]

            expanded_excluded = resolve_exclusion_list(
                cancer_type,
                location=location,
                available_sigs=available_sigs,
                treatment_naive=kwargs.get("treatment_naive", True),
                exclude_artifacts=kwargs.get(
                    "exclude_artifacts", False
                ),
            )

            sigs_to_keep = [
                col
                for col in assignments.columns
                if col not in expanded_excluded
            ]
            assignments = assignments[sigs_to_keep]
            logger.info(
                f"Filtered assignments to remove "
                f"{len(expanded_excluded)} excluded "
                f"signatures for cancer type "
                f"'{cancer_type}'"
            )

    logger.info("... done.")
    print()

    return assignments
