"""Mutational matrices generation."""

import logging
import shutil
from pathlib import Path

from SigProfilerMatrixGenerator.scripts import (
    SigProfilerMatrixGeneratorFunc,
)
from SigProfilerMatrixGenerator import install as genInstall


logger = logging.getLogger(__name__)


def run_mutational_matrices_generator(
    path_to_input_files,
    reference_genome,
    *,
    exome=True,
    bed_file=None,
    chrom_based=False,
    plot=False,
    tsb_stat=False,
    seqInfo=True,
    cushion=100,
    gs=False,
    volume=None,
):
    """Generate mutational matrices.

    This function generates context-dependent mutational matrices for
    SBS, DBS, and ID signature classes from input mutation data (MAF,
    VCF, simple text files). It wraps
    :func:`SigProfilerMatrixGeneratorFunc` from
    SigProfilerMatrixGenerator with customized defaults.

    This function generates context-dependent mutational matrices from
    input mutation data (MAF, VCF, simple text files). It wraps
    :func:`SigProfilerMatrixGeneratorFunc` from
    SigProfilerMatrixGenerator with customized defaults for this
    project.

    This wrapper provides the same interface as the original function
    with improved documentation and modified default values:
    - exome=True (original: False) - since most TCGA data is WES
    - All other defaults match SigProfilerMatrixGeneratorFunc

    **IMPORTANT**: For ID signature class analysis, seqInfo=True is
    **required** to generate the `*_seqinfo.txt` files that annotate
    each indel with its COSMIC ID-83 classification. Without these
    files, ID mutations cannot be properly typed for signature
    decomposition.

    **NOTE**: The project name is hardcoded to "mutational_matrix" so
    that the output filenames are of the form
    mutational_matrix.SBS96.exome.

    Parameters
    ----------
    path_to_input_files : str or Path
        Path to input mutation files or directory containing them.
        Supported formats:
        - VCF files (.vcf)
        - MAF files (.maf)
        - Simple text format (sample_name, chr, pos, ref, alt)

    reference_genome : str
        Reference genome assembly. Supported options:
        - 'GRCh37' (hg19)
        - 'GRCh38' (hg38, default)
        - 'mm9', 'mm10', 'rn6' (mouse and rat assemblies)

    exome : bool, optional
        Whether to restrict analysis to exome regions only. Set to
        True for whole-exome sequencing (WES) data, False for
        whole-genome sequencing (WGS). Default is True (different
        from SigProfilerMatrixGeneratorFunc which defaults to False),
        since most TCGA data is WES.

    bed_file : str or Path, optional
        Path to a BED file defining custom genomic regions to
        analyze. If provided, only mutations within these regions
        will be included. Overrides the exome parameter. Default is
        None.

    chrom_based : bool, optional
        Whether to generate chromosome-based matrices (one matrix per
        chromosome). Useful for very large datasets to reduce memory
        usage. Default is False.

    plot : bool, optional
        Whether to generate visualization plots of the mutational
        matrices and spectra. Plots are saved to
        `path_to_input_files/output/plots/`. Default is False.

    tsb_stat : bool, optional
        Whether to perform transcriptional strand bias (TSB) test for
        the 24, 384, and 6144 trinucleotide contexts. Tests whether
        mutations occur preferentially on the transcribed vs
        non-transcribed strand. Results are saved to
        `path_to_input_files/output/TSB/`. Requires gene annotation
        information from the reference genome. Default is False.

    seqInfo : bool, optional
        Whether to generate sequence info files that map each
        mutation to its extended context information. **REQUIRED for
        ID signature class**: generates `*_seqinfo.txt` files that
        contain COSMIC ID-83 classification for each indel (e.g.,
        '1:Del:C:4' for deletion of 1 C in repeat length 4). These
        files are used by
        :func:`load_maf_files.load_or_generate_compact_db` when
        signature_class='ID'. Default is True.

    cushion : int, optional
        Number of base pairs to extend around each mutation when
        extracting sequence context. Higher values provide more
        context but increase memory usage. Default is 100.

    gs : bool, optional
        Whether to perform gene strand bias test. Tests whether
        mutations occur preferentially on the genic vs non-genic
        strand. Default is False (somatic mutations only).

    volume : str or Path, optional
        Custom directory path for storing reference genomes, plotting
        templates, and COSMIC signature files. If None, uses default
        SigProfiler installation directories. Can be overridden by
        environment variables: SIGPROFILERMATRIXGENERATOR_VOLUME,
        SIGPROFILERPLOTTING_VOLUME, SIGPROFILERASSIGNMENT_VOLUME.
        Useful for shared computing environments or cloud storage.
        Default is None.

    Returns
    -------
    None
        All outputs are saved to the directory
        `path_to_input_files/output/`.

    Notes
    -----
    Output structure:
        {path_to_input_files}/output/
        ├── vcf_files/
        │   ├── SNV/              # VCF files for SNVs
        │   ├── DBS/              # VCF files for DBS
        │   ├── ID/               # VCF files for indels
        │   │   └── {sample}_seqinfo.txt  # ID-83 annotations
        │   └── MNS/              # VCF files for MNS
        ├── SBS/                  # SBS mutational matrices
        │   └── mutational_matrix.SBS{96,288,1536,...}.exome
        ├── DBS/                  # DBS mutational matrices
        │   └── mutational_matrix.DBS{78,...}.exome
        ├── ID/                   # ID mutational matrices
        │   └── mutational_matrix.ID{83,96,415,...}.exome
        └── plots/ (if plot=True)

    The generated matrices are tab-separated files where rows are
    mutation types (e.g., 96 trinucleotide contexts for SBS96) and
    columns are samples. Multiple resolutions are generated for each
    signature class (e.g., SBS6, SBS24, SBS96, SBS384, SBS1536).

    For ID signature class, the `*_seqinfo.txt` files in the
    `vcf_files/ID/` directory map each indel mutation to its COSMIC
    ID-83 type. The format is tab-separated with columns:
    - MutationType: COSMIC ID-83 category (e.g., '1:Del:C:4')
    - Sample: Sample identifier
    - Chr, Start, End: Genomic coordinates
    - Ref, Alt: Reference and alternate alleles
    - Additional context columns

    """
    # Ensure path_to_input_files is a string
    path_to_input_files = str(path_to_input_files)

    logger.info("Generating mutational matrices...")
    logger.info(
        f"Reference genome: {reference_genome}, exome: {exome}, "
        f"seqInfo: {seqInfo}"
    )

    try:
        SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc(
            project="mutational_matrix",  # hardcoded
            reference_genome=reference_genome,
            path_to_input_files=path_to_input_files,
            exome=exome,
            bed_file=bed_file,
            chrom_based=chrom_based,
            plot=plot,
            tsb_stat=tsb_stat,
            seqInfo=seqInfo,
            cushion=cushion,
            gs=gs,
            volume=volume,
        )
    except Exception as e:
        # Check if error is related to missing reference genome
        error_msg = str(e).lower()
        if (
            "reference" in error_msg
            or "install" in error_msg
            or "genome" in error_msg
            or "not found" in error_msg
        ):
            logger.warning(
                f"Reference genome {reference_genome} data not "
                f"available. Installing now..."
            )
            logger.info(
                f"Installing {reference_genome} reference genome "
                f"(this may take several minutes)..."
            )

            # Install the reference genome
            genInstall.install(
                reference_genome, rsync=False, bash=True
            )

            logger.info(
                f"{reference_genome} installation complete. "
                f"Retrying matrix generation..."
            )

            # Retry the matrix generation
            SigProfilerMatrixGeneratorFunc.SigProfilerMatrixGeneratorFunc(
                project="mutational_matrix",
                reference_genome=reference_genome,
                path_to_input_files=path_to_input_files,
                exome=exome,
                bed_file=bed_file,
                chrom_based=chrom_based,
                plot=plot,
                tsb_stat=tsb_stat,
                seqInfo=seqInfo,
                cushion=cushion,
                gs=gs,
                volume=volume,
            )
        else:
            # Re-raise if it's a different error
            raise

    logger.info("Mutational matrices generation complete")


def mutational_matrices_generation(
    path_to_input_files,
    reference_genome="GRCh38",
    force_generation=False,
    **kwargs,
):
    """Load or generate mutational matrices from MAF/VCF files.

    Checks if mutational matrices already exist in the output
    directory. If they exist and force_generation=False, skips
    regeneration. Otherwise, runs
    :func:`run_mutational_matrices_generator` to create the matrices.

    Parameters
    ----------
    path_to_input_files : str or Path
        Path to directory containing input mutation files (MAF, VCF,
        or simple text format).

    reference_genome : str, optional
        Reference genome assembly. Default is 'GRCh38'.

    force_generation : bool, optional
        If True, deletes existing output directory and regenerates
        all matrices. Default is False.

    **kwargs : dict
        Additional keyword arguments passed to
        :func:`run_mutational_matrices_generator`. Common options:
        - exome : bool (default True)
        - bed_file : str or Path (default None)
        - chrom_based : bool (default False)
        - plot : bool (default False)
        - tsb_stat : bool (default False)
        - seqInfo : bool (default True)
        - cushion : int (default 100)
        - gs : bool (default False)
        - volume : str or Path (default None)

    Returns
    -------
    output_path : Path
        Path to the output directory containing generated matrices.

    Notes
    -----
    The function checks for the existence of the output directory
    `{path_to_input_files}/output/` to determine if matrices have
    already been generated. If you want to regenerate with different
    parameters, set force_generation=True.

    """
    path_to_input_files = Path(path_to_input_files)
    output_path = path_to_input_files / "output"

    if force_generation and output_path.exists():
        logger.info(f"Deleting previous matrices from {output_path}")
        shutil.rmtree(output_path)

    if not output_path.exists():
        run_mutational_matrices_generator(
            path_to_input_files=str(path_to_input_files),
            reference_genome=reference_genome,
            **kwargs,
        )
    else:
        logger.info(
            f"Mutational matrices already exist at {output_path}"
        )

    return output_path
