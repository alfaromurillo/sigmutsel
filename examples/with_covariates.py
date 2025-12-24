"""Example using sigmutsel with covariates.

This script demonstrates how to use sigmutsel with gene-level
covariate data to estimate covariate effects on mutation rates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sigmutsel import MutationDataset, Model


def load_and_prepare_covariates():
    """Load example covariate data and prepare for analysis.

    This example uses GTEx gene expression data from two skin tissues
    as covariates. In a real analysis, you would use covariates
    relevant to your cancer type (e.g., for COAD: colon tissue
    expression, chromatin marks, replication timing).

    Returns
    -------
    pd.DataFrame
        Covariate matrix with ensembl_gene_id as index and
        transformed covariate columns
    """
    # Load example covariate data (GTEx gene expression)
    covariates_path = (
        Path(__file__).parent / "example_gtex_skin_covariates.parquet")
    cov_matrix = pd.read_parquet(covariates_path)

    print(f"Loaded covariates for {len(cov_matrix)} genes")
    print(f"Covariate columns: {list(cov_matrix.columns)}")

    # Apply log1p transformation to gene expression values
    # This is standard practice for gene expression data
    cov_matrix_transformed = pd.DataFrame({
        "log1p_skin_sun_exposed": np.log1p(
            cov_matrix['gtex_skin_sun_exposed_lower_leg']),
        "log1p_skin_not_sun_exposed": np.log1p(
            cov_matrix['gtex_skin_not_sun_exposed_suprapubic'])
    }, index=cov_matrix.index)

    return cov_matrix_transformed


def main():
    # Load a previously created dataset
    print("Loading dataset...")
    dataset = MutationDataset.load_dataset("./my_dataset")

    # Load and prepare covariate matrix
    print("\nLoading covariate matrix...")
    cov_matrix = load_and_prepare_covariates()

    # Create model with covariates
    print("\nCreating model with covariates...")
    model = Model(
        dataset=dataset,
        cov_matrix=cov_matrix
    )

    # Compute baseline mutation rates (no covariate effects yet)
    print("\nComputing mutation rates...")
    model.compute_mu_taus()
    model.compute_base_mus()

    # Estimate covariate effects using MAP (Maximum A Posteriori)
    # This is fast and gives point estimates
    print("\nEstimating covariate effects (MAP)...")
    model.estimate_cov_effects(sample="MAP")

    # Print estimated effects
    print(f"\nCovariate effects (intercept, {', '.join(cov_matrix.columns)}):")
    print(f"  {model.cov_effects}")
    print(f"Passenger genes R²: {model.passenger_genes_r2:.4f}")

    # Optional: Get full posterior distribution (slower, MCMC sampling)
    # Uncomment to run full Bayesian inference:
    # print("\nGenerating posterior distributions (MCMC)...")
    # posterior = model.estimate_cov_effects(sample="full")
    # # Or subsample 1000 genes for faster exploration:
    # posterior = model.estimate_cov_effects(sample=1000)

    # Estimate selection for top variants
    print("\nEstimating selection for top 5 variants...")
    for variant in dataset.variant_counts.head(5).index:
        print(f"  Variant: {variant}")
        model.estimate_gamma(variant, level='variant')

    # Plot results
    print("\nPlotting variant-level results...")
    model.plot_gamma_results(
        level='variant',
        show=True,
        save='variant_selection_with_covariates.png'
    )

    # Save model
    model.save_model("./my_model_with_covs", overwrite=True)


if __name__ == "__main__":
    main()


# Creating Your Own Covariate Matrices
# ======================================
# This example uses pre-computed GTEx gene expression data. In your
# own analysis, you can create covariate matrices from various sources:
#
# 1. Gene Expression (GTEx, TCGA):
#    - Load tissue-specific expression data
#    - Apply log1p transformation: np.log1p(expression_values)
#
# 2. Chromatin Marks (Roadmap Epigenomics, ENCODE):
#    - H3K4me3 (active promoters), H3K9me3 (heterochromatin)
#    - Average signal over gene body or promoter regions
#
# 3. Replication Timing:
#    - Mean replication time per gene
#    - Convert to standardized values
#
# 4. Other Genomic Features:
#    - GC content, gene length, mutation context
#    - Evolutionary constraint (e.g., dN/dS)
#
# Example workflow for creating a covariate matrix:
#
#   import pandas as pd
#   import numpy as np
#
#   # Load your data sources
#   gene_expr = pd.read_csv("gene_expression.csv", index_col=0)
#   chrom_marks = pd.read_csv("chromatin_marks.csv", index_col=0)
#   rep_timing = pd.read_csv("replication_timing.csv", index_col=0)
#
#   # Combine into single DataFrame (aligned by ensembl_gene_id)
#   cov_matrix = pd.DataFrame({
#       'log1p_expression': np.log1p(gene_expr['tpm']),
#       'h3k4me3_promoter': chrom_marks['H3K4me3_prom'],
#       'mean_rep_time': rep_timing['mrt']
#   }, index=gene_expr.index)
#
#   # Use with Model
#   model = Model(dataset, cov_matrix=cov_matrix)
#   model.compute_mu_taus()
#   model.compute_base_mus()
#   model.estimate_cov_effects()
