"""Basic usage example for sigmutsel.

This script demonstrates how to use sigmutsel to analyze mutation
data and estimate selection coefficients.
"""

from pathlib import Path
import pandas as pd
from sigmutsel import MutationDataset, Model
from sigmutsel.download_tcga_data import download_tcga_maf_files


def download_example_data():
    """Download example TCGA MAF files using GDC manifest.

    This function demonstrates how to download TCGA mutation data
    using the included example manifest file. The example manifest
    contains 5 COAD (colon adenocarcinoma) samples.

    Requirements
    ------------
    You must have gdc-client installed and on your PATH.
    Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

    Returns
    -------
    Path
        Path to directory containing downloaded MAF files
    """
    # Path to example GDC manifest
    manifest_path = Path(__file__).parent / "example_gdc_manifest.txt"

    # Output directory for MAF files
    maf_directory = Path(__file__).parent / "example_maf_files"

    print(f"Downloading TCGA data from manifest: {manifest_path}")
    print(f"Output directory: {maf_directory}")

    # Download and unpack MAF files
    success = download_tcga_maf_files(
        gdc_manifest=manifest_path,
        destination=maf_directory
    )

    if success:
        print(f"Successfully downloaded MAF files to {maf_directory}")
    else:
        print("Warning: Some files may not have downloaded correctly")

    return maf_directory


def main():
    # Option 1: Download example data (requires gdc-client)
    # Uncomment the following line to download example TCGA data
    # maf_directory = download_example_data()

    # Option 2: Use your own MAF files directory
    maf_directory = Path("/path/to/your/maf/files")

    # Create and load dataset
    print("Creating MutationDataset...")
    dataset = MutationDataset(
        maf_directory,
        signature_class="SBS"
    )

    # Run signature decomposition
    print("Running signature decomposition...")
    dataset.run_signature_decomposition(
        genome_build='GRCh38',
        exome=True,
        cosmic_version=3.4,
        force_generation=False,
        include_signature_subgroups='SKCM'
    )

    # Build full dataset
    print("Building full dataset...")
    dataset.build_full_dataset()

    # Save dataset for future use
    dataset.save_dataset("./my_dataset")

    # Load dataset (in future runs, you can skip the above steps)
    # dataset = MutationDataset.load_dataset("./my_dataset")

    # Create model without covariates
    print("\nCreating model without covariates...")
    model_no_cov = Model(
        dataset,
        cov_matrix=None,
        cov_effects_per_sigma=False,
        prob_g_tau_tau_independent=False
    )

    # Compute baseline mutation rates
    print("Computing baseline mutation rates...")
    model_no_cov.compute_mu_gs()
    model_no_cov.compute_mu_ms()

    # Estimate selection for top variants
    print("\nEstimating selection for top 5 variants...")
    for variant in dataset.variant_counts.head(5).index:
        print(f"  Variant: {variant}")
        model_no_cov.estimate_gamma(variant, level='variant')

    # Estimate selection for top genes
    print("\nEstimating selection for top 5 genes...")
    for gene in dataset.gene_counts.head(5).index:
        print(f"  Gene: {gene}")
        model_no_cov.estimate_gamma(gene, level='gene')

    # Plot results
    print("\nPlotting gene-level results...")
    model_no_cov.plot_gamma_results(
        keys=[x for x in model_no_cov.gammas.keys()
              if x.startswith("ENSG")],
        level='gene',
        show=True,
        save='gene_selection_results.png'
    )

    # Save model
    model_no_cov.save_model("./my_model", overwrite=True)


if __name__ == "__main__":
    main()


# Alternative: Download data from command line
# =============================================
# You can also download TCGA data directly from the command line:
#
#   python -m sigmutsel.download_tcga_data \
#       --manifest examples/example_gdc_manifest.txt \
#       --destination examples/example_maf_files
#
# This requires gdc-client to be installed and on your PATH.
# Download from: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
