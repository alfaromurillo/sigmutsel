"""Signature decomposition."""

import logging
import shutil
from pathlib import Path

import pandas as pd

from SigProfilerAssignment import Analyzer

from .locations import location_exclusion_signatures_matrix
from .locations import location_inclusion_signatures_matrix

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


def run_signature_decomposition(samples,
                                output,
                                signatures=None,
                                signature_database=None,
                                nnls_add_penalty=0.05,
                                nnls_remove_penalty=0.01,
                                initial_remove_penalty=0.05,
                                genome_build='GRCh38',
                                cosmic_version=3.4,
                                make_plots=False,
                                collapse_to_SBS96=True,
                                connected_sigs=True,
                                verbose=False,
                                devopts=None,
                                exclude_signature_subgroups=None,
                                include_signature_subgroups=None,
                                exome=True,
                                input_type='vcf',
                                context_type='96',
                                export_probabilities=True,
                                export_probabilities_per_mutation=True,
                                sample_reconstruction_plots=False,
                                volume=None):
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
        context_type = 'ID'
    if (context_type == 'ID') or (context_type == 'DINUC'):
        collapse_to_SBS96 = False

    # Normalize shorthand cancer-type arguments
    exclude_signature_subgroups = _normalize_signature_group_arg(
        exclude_signature_subgroups,
        location_exclusion_signatures_matrix)
    include_signature_subgroups = _normalize_signature_group_arg(
        include_signature_subgroups,
        location_inclusion_signatures_matrix)

    # Check that both exclude and include are not provided
    if (exclude_signature_subgroups is not None and
            include_signature_subgroups is not None):
        raise ValueError(
            "Cannot provide both exclude_signature_subgroups and "
            "include_signature_subgroups. Choose one: "
            "- Use exclude_signature_subgroups to exclude specific "
            "signatures and keep all others "
            "- Use include_signature_subgroups to keep only specific "
            "signatures and exclude all others")

    # Process exclude_signature_subgroups if it's a tuple
    if (exclude_signature_subgroups is not None and
            isinstance(exclude_signature_subgroups, tuple) and
            len(exclude_signature_subgroups) == 2):

        location, cancer_type = exclude_signature_subgroups
        location = Path(location)

        # Read the exclusion matrix
        exclusion_df = pd.read_csv(location, sep='\t')

        # Find the cancer type row
        # Check for exact match in PCAWG or as part of
        # comma-separated list in Applicable_TCGA
        def match_cancer_type(row):
            if row['PCAWG'] == cancer_type:
                return True
            tcga_val = row['Applicable_TCGA']
            if pd.notna(tcga_val):
                # Split by comma and strip whitespace
                tcga_types = [
                    t.strip() for t in str(tcga_val).split(',')]
                if cancer_type in tcga_types:
                    return True
            return False

        cancer_mask = exclusion_df.apply(
            match_cancer_type, axis=1)
        cancer_row = exclusion_df[cancer_mask]

        if cancer_row.empty:
            raise ValueError(
                f"Cancer type '{cancer_type}' not found in "
                f"exclusion matrix at {location}")

        # Get signature columns (all columns after the first 4)
        sig_cols = exclusion_df.columns[4:]

        # Get signatures to exclude (where value is 1)
        cancer_row = cancer_row.iloc[0]
        excluded_sigs = [
            sig for sig in sig_cols
            if cancer_row[sig] == 1]

        # Expand signatures like SBS10 to include SBS10a,
        # SBS10b, etc.
        # Get all available signatures from the matrix or
        # default set
        if signatures is not None:
            available_sigs = list(signatures)
        elif signature_database is not None:
            # Read signature database to get available
            # signatures
            sig_db = pd.read_csv(signature_database, sep='\t')
            available_sigs = sig_db.columns[1:].tolist()
        else:
            # Use default COSMIC signatures
            # Will be expanded by SigProfilerAssignment
            available_sigs = sig_cols.tolist()

        # Expand base signatures to include subvariants
        expanded_excluded = []
        for excl_sig in excluded_sigs:
            expanded_excluded.append(excl_sig)
            # Check for subvariants (e.g., SBS10 -> SBS10a,
            # SBS10b, etc.)
            for avail_sig in available_sigs:
                if (avail_sig.startswith(excl_sig) and
                        len(avail_sig) > len(excl_sig) and
                        avail_sig[len(excl_sig)] in
                        'abcdefghijklmnopqrstuvwxyz'):
                    expanded_excluded.append(avail_sig)

        exclude_signature_subgroups = expanded_excluded

    # Process include_signature_subgroups if it's a tuple
    # (sets the signatures parameter to the included list)
    if (include_signature_subgroups is not None and
            isinstance(include_signature_subgroups, tuple) and
            len(include_signature_subgroups) == 2):
        import pandas as pd
        from pathlib import Path

        location, cancer_type = include_signature_subgroups
        location = Path(location)

        # Read the inclusion matrix
        inclusion_df = pd.read_csv(location, sep='\t')

        # Find the cancer type row using same logic as exclude
        def match_cancer_type(row):
            if row['PCAWG'] == cancer_type:
                return True
            tcga_val = row['Applicable_TCGA']
            if pd.notna(tcga_val):
                tcga_types = [
                    t.strip() for t in str(tcga_val).split(',')]
                if cancer_type in tcga_types:
                    return True
            return False

        cancer_mask = inclusion_df.apply(
            match_cancer_type, axis=1)
        cancer_row = inclusion_df[cancer_mask]

        if cancer_row.empty:
            raise ValueError(
                f"Cancer type '{cancer_type}' not found in "
                f"inclusion matrix at {location}")

        # Get signature columns (all columns after the first 4)
        sig_cols = inclusion_df.columns[4:]

        # Get signatures to INCLUDE (where value is 1)
        cancer_row = cancer_row.iloc[0]
        included_sigs = [
            sig for sig in sig_cols
            if cancer_row[sig] == 1]

        # Get all available signatures
        if signatures is not None:
            available_sigs = list(signatures)
        elif signature_database is not None:
            sig_db = pd.read_csv(signature_database, sep='\t')
            available_sigs = sig_db.columns[1:].tolist()
        else:
            # Use default COSMIC signatures
            available_sigs = sig_cols.tolist()

        # Expand base signatures to include subvariants
        expanded_included = []
        for incl_sig in included_sigs:
            expanded_included.append(incl_sig)
            # Check for subvariants (e.g., SBS10 -> SBS10a,
            # SBS10b, etc.)
            for avail_sig in available_sigs:
                if (avail_sig.startswith(incl_sig) and
                        len(avail_sig) > len(incl_sig) and
                        avail_sig[len(incl_sig)] in
                        'abcdefghijklmnopqrstuvwxyz'):
                    expanded_included.append(avail_sig)

        # Set signatures to the included list
        # (this is more reliable than using exclude)
        signatures = expanded_included
        logger.info(
            f"include_signature_subgroups for cancer type "
            f"'{cancer_type}': restricting to "
            f"{len(expanded_included)} signatures")
        logger.debug(
            f"Signatures to include: {expanded_included}")

    if exclude_signature_subgroups is not None:
        logger.info(
            f"Passing exclude_signature_subgroups with "
            f"{(len(exclude_signature_subgroups)
                if isinstance(exclude_signature_subgroups, list)
                else 'unknown')} "
            f"signatures")

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
        volume=volume)


def signature_decomposition(
        results_dir: str,
        input_data: str,
        force_generation: bool = False,
        **kwargs) -> pd.DataFrame:
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
    if 'exclude_signature_subgroups' in kwargs:
        kwargs['exclude_signature_subgroups'] = (
            _normalize_signature_group_arg(
                kwargs['exclude_signature_subgroups'],
                location_exclusion_signatures_matrix))
    if 'include_signature_subgroups' in kwargs:
        kwargs['include_signature_subgroups'] = (
            _normalize_signature_group_arg(
                kwargs['include_signature_subgroups'],
                location_inclusion_signatures_matrix))

    results_path = Path(results_dir)
    solution_dir = results_path / "Assignment_Solution"

    results_file = (
        solution_dir
        / "Activities"
        / "Assignment_Solution_Activities.txt")

    if force_generation and solution_dir.exists():
        logger.info(
            f"Deleting previous signature decomposition from {solution_dir}")
        shutil.rmtree(solution_dir)

    if not solution_dir.exists():
        logger.info(
            "Running signature decomposition for all tumors...")
        # This will create the solution_dir if it is not there
        # Convert input_data to string in case it's a Path object
        run_signature_decomposition(
            str(input_data), str(results_path), **kwargs)
    else:
        logger.info(
            "Loading signature decomposition for all tumors...")

    assignments = pd.read_csv(results_file, sep="\t")
    assignments = assignments.set_index("Samples")

    # Filter assignments based on include/exclude parameters
    # (post-processing since SigProfilerAssignment ignores
    # these parameters)
    if 'include_signature_subgroups' in kwargs:
        include_param = kwargs['include_signature_subgroups']
        if (isinstance(include_param, tuple) and
                len(include_param) == 2):
            location, cancer_type = include_param
            location = Path(location)

            inclusion_df = pd.read_csv(location, sep='\t')

            def match_cancer_type(row):
                if row['PCAWG'] == cancer_type:
                    return True
                tcga_val = row['Applicable_TCGA']
                if pd.notna(tcga_val):
                    tcga_types = [
                        t.strip()
                        for t in str(tcga_val).split(',')]
                    if cancer_type in tcga_types:
                        return True
                return False

            cancer_mask = inclusion_df.apply(
                match_cancer_type, axis=1)
            cancer_row = inclusion_df[cancer_mask]

            if not cancer_row.empty:
                sig_cols = inclusion_df.columns[4:]
                cancer_row = cancer_row.iloc[0]
                included_sigs = [
                    sig for sig in sig_cols
                    if cancer_row[sig] == 1]

                # Expand to include subvariants
                available_sigs = [
                    col for col in assignments.columns
                    if col.startswith('SBS')]
                expanded_included = []
                for incl_sig in included_sigs:
                    expanded_included.append(incl_sig)
                    for avail_sig in available_sigs:
                        if (avail_sig.startswith(incl_sig) and
                                len(avail_sig) >
                                len(incl_sig) and
                                avail_sig[len(incl_sig)] in
                                'abcdefghijklmnopqrstuvwxyz'):
                            expanded_included.append(
                                avail_sig)

                # Keep only included signatures
                # (also exclude signatures that are not
                # included and not subvariants of included
                # signatures)
                available_sigs = [
                    col for col in assignments.columns
                    if col.startswith('SBS')]

                # Build exclusion list from signatures not
                # in expanded_included
                sigs_to_exclude = []
                for avail_sig in available_sigs:
                    if avail_sig not in expanded_included:
                        # Check if it's a subvariant of an
                        # included signature
                        is_subvariant = False
                        for incl_sig in included_sigs:
                            if (avail_sig.startswith(incl_sig) and
                                    len(avail_sig) >
                                    len(incl_sig) and
                                    avail_sig[len(incl_sig)] in
                                    'abcdefghijklmnopqrstuvwxyz'):
                                is_subvariant = True
                                break
                        if not is_subvariant:
                            sigs_to_exclude.append(avail_sig)

                # Keep only signatures not in exclusion list
                sigs_to_keep = [
                    col for col in assignments.columns
                    if col not in sigs_to_exclude]
                assignments = assignments[sigs_to_keep]
                logger.info(
                    f"Filtered assignments to "
                    f"{len(sigs_to_keep)} included "
                    f"signatures (excluded "
                    f"{len(sigs_to_exclude)} signatures) "
                    f"for cancer type '{cancer_type}'")

    elif 'exclude_signature_subgroups' in kwargs:
        exclude_param = kwargs['exclude_signature_subgroups']
        if (isinstance(exclude_param, tuple) and
                len(exclude_param) == 2):
            location, cancer_type = exclude_param
            location = Path(location)

            exclusion_df = pd.read_csv(location, sep='\t')

            def match_cancer_type(row):
                if row['PCAWG'] == cancer_type:
                    return True
                tcga_val = row['Applicable_TCGA']
                if pd.notna(tcga_val):
                    tcga_types = [
                        t.strip()
                        for t in str(tcga_val).split(',')]
                    if cancer_type in tcga_types:
                        return True
                return False

            cancer_mask = exclusion_df.apply(
                match_cancer_type, axis=1)
            cancer_row = exclusion_df[cancer_mask]

            if not cancer_row.empty:
                sig_cols = exclusion_df.columns[4:]
                cancer_row = cancer_row.iloc[0]
                excluded_sigs = [
                    sig for sig in sig_cols
                    if cancer_row[sig] == 1]

                # Expand to include subvariants
                available_sigs = [
                    col for col in assignments.columns
                    if col.startswith('SBS')]
                expanded_excluded = []
                for excl_sig in excluded_sigs:
                    expanded_excluded.append(excl_sig)
                    for avail_sig in available_sigs:
                        if (avail_sig.startswith(excl_sig) and
                                len(avail_sig) >
                                len(excl_sig) and
                                avail_sig[len(excl_sig)] in
                                'abcdefghijklmnopqrstuvwxyz'):
                            expanded_excluded.append(
                                avail_sig)

                # Remove excluded signatures
                sigs_to_keep = [
                    col for col in assignments.columns
                    if col not in expanded_excluded]
                assignments = assignments[sigs_to_keep]
                logger.info(
                    f"Filtered assignments to remove "
                    f"{len(expanded_excluded)} excluded "
                    f"signatures for cancer type "
                    f"'{cancer_type}'")

    logger.info("... done.")

    return assignments
