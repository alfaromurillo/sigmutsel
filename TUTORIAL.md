# sigmutsel Tutorial

This tutorial provides a comprehensive walkthrough of using sigmutsel for mutation rate estimation and selection inference in cancer genomics.

## Table of Contents

1. [Setup and Data Download](#setup-and-data-download)
2. [Creating a Dataset](#creating-a-dataset)
3. [Model Without Covariates](#model-without-covariates)
4. [Model With Covariates](#model-with-covariates)
5. [Model With PCA](#model-with-pca)
6. [Signature-Specific Covariate Effects](#signature-specific-covariate-effects)
7. [Selection Inference](#selection-inference)

## Setup and Data Download

First, ensure you have sigmutsel installed and reference data downloaded:

```bash
pip install git+https://github.com/alfaromurillo/sigmutsel.git
python -m sigmutsel setup
```

### Downloading TCGA MAF Files

Download example MAF files using a GDC manifest:

```python
from pathlib import Path
from sigmutsel.download_tcga_data import download_tcga_maf_files

if not Path("maf_files").is_dir():
    download_tcga_maf_files(
        gdc_manifest="example_gdc_manifest.txt",
        destination="maf_files/")
```

You can obtain manifests from the [GDC Data Portal](https://portal.gdc.cancer.gov/). The example manifest contains 50 SKCM (skin melanoma) samples.

## Creating a Dataset

### Initialize Dataset

```python
from sigmutsel import MutationDataset

dataset = MutationDataset(
    location_maf_files="maf_files",
    signature_class="SBS")  # Single Base Substitution signatures
```

### Run Signature Decomposition

Decompose mutations into COSMIC mutational signatures:

```python
dataset.run_signature_decomposition(
    genome_build='GRCh38',
    exome=True,
    cosmic_version=3.4,
    force_generation=False,
    include_signature_subgroups="SKCM")  # Use cancer-specific signatures
```

**Parameters:**
- `genome_build`: Reference genome version
- `exome`: Set `True` for whole-exome sequencing data
- `cosmic_version`: COSMIC signature database version
- `force_generation`: Set `True` to regenerate even if cached
- `include_signature_subgroups`: Cancer type for tissue-specific signatures

### Build Full Dataset

Process mutations and compute genomic contexts:

```python
dataset.build_full_dataset()
```

This computes:
- Mutation presence matrices
- Genomic context counts per gene
- Variant annotations

### Save and Load Dataset

To save time on subsequent runs:

```python
# Save processed dataset
dataset.save_dataset("example_dataset")

# Load previously saved dataset
dataset = MutationDataset.load_dataset("example_dataset")
```

## Model Without Covariates

Create a baseline model without gene-level covariates:

```python
from sigmutsel import Model

model_no_cov = Model(
    dataset,
    cov_matrix=None,
    cov_effects_per_sigma=False,
    prob_g_tau_tau_independent=True)
```

**Key parameter:**
- `prob_g_tau_tau_independent=True`: Assumes mutation probability depends only on gene length, not mutation type. This setting often fits better for baseline models.

### Compute Mutation Rates

```python
model_no_cov.compute_mu_gs()
```

This computes baseline mutation rates for each gene in each sample.

### Save Model

```python
model_no_cov.save_model("model_no_cov", overwrite=True)
```

Saved files include mutation rates in parquet format, which can be read in R:

```r
library(arrow)
df <- read_parquet("model_no_cov/mu_gs.parquet")
```

## Model With Covariates

Incorporate gene-level covariates (e.g., expression, chromatin state):

```python
import pandas as pd

# Load covariate matrix (genes × covariates)
cov_matrix = pd.read_parquet("example_gtex_skin_covariates.parquet")

model_gtex = Model(
    dataset,
    cov_matrix,
    cov_effects_per_sigma=False,
    prob_g_tau_tau_independent=False)

# Estimate covariate effects (Maximum Likelihood Estimation)
model_gtex.estimate_cov_effects()
```

**Key differences:**
- `cov_matrix`: DataFrame with `ensembl_gene_id` as index
- `prob_g_tau_tau_independent=False`: Allows mutation rates to vary by type

After estimation, mutation rates incorporating covariate effects are available at `model_gtex.mu_gs`.

### Visualize Signature Correlations

Before deciding which covariates or signatures to focus on:

```python
model_gtex.plot_signature_correlations(
    save_path="signature_mutation_correlations.png")
```

This plots correlations between signatures and covariates, helping identify important signatures.

## Model With PCA

Use Principal Component Analysis to reduce covariate dimensionality:

```python
model_pca = Model(
    dataset,
    None,  # Start without covariates
    cov_effects_per_sigma=False,
    prob_g_tau_tau_independent=False)

# Assign covariates with PCA preprocessing
model_pca.assign_cov_matrix(
    cov_matrix,
    run_pca=True,
    pca_kwargs={'n_components': 2})  # Adjust based on your data

model_pca.estimate_cov_effects()
```

**When to use PCA:**
- Many correlated covariates
- Want to reduce model complexity
- Exploring major axes of variation

## Signature-Specific Covariate Effects

Model how covariates affect specific mutational signatures:

```python
# Based on correlation analysis, focus on key signatures
signature_selection = ['SBS7a', 'SBS7b']

model_sig = Model(
    dataset,
    cov_matrix[['gtex_skin_not_sun_exposed_suprapubic']],  # Select covariates
    cov_effects_per_sigma=True,  # KEY: Enable signature-specific effects
    signature_selection=signature_selection,
    prob_g_tau_tau_independent=True)

model_sig.estimate_cov_effects()
```

**Key parameter:**
- `cov_effects_per_sigma=True`: Estimates separate covariate effects for each selected signature

## Selection Inference

### Estimate Selection Coefficients

Infer positive or negative selection on genes and variants:

```python
# Variant-level selection
model_sig.estimate_gamma('BRAF p.V640E', level='variant')
model_sig.estimate_gamma('NRAS p.Q61R', level='variant')

# Gene-level selection
model_sig.estimate_gamma('LRP1B', level='gene')
model_sig.estimate_gamma('BRAF', level='gene')

# Auto-detect level from identifier
model_sig.estimate_gamma('ENSG00000155657')  # Ensembl ID → gene
model_sig.estimate_gamma('TP53 p.R175H')     # Protein change → variant
```

**Level detection:**
- Ensembl IDs (`ENSG...`) → gene level
- Protein changes (`p.X123Y`) → variant level
- Gene symbols → gene level

### Visualize Selection Results

Plot selection coefficients with mutation counts:

```python
# Plot specific genes
model_sig.plot_gamma_results(
    ['ENSG00000168702', 'ENSG00000155657', 'ENSG00000157764'],
    level='gene',
    save="selection_genes.png")

# Plot specific variants
model_sig.plot_gamma_results(
    ['BRAF p.V640E', 'NRAS p.Q61R'],
    level='variant',
    save="selection_variants.png")

# Plot all results
model_sig.plot_gamma_results(save="all_selection.png")
```

## Complete Example

```python
import pandas as pd
from pathlib import Path
from sigmutsel import MutationDataset, Model

# 1. Load or create dataset
if Path("example_dataset").exists():
    dataset = MutationDataset.load_dataset("example_dataset")
else:
    dataset = MutationDataset(
        location_maf_files="maf_files",
        signature_class="SBS")
    dataset.run_signature_decomposition(
        genome_build='GRCh38',
        exome=True,
        cosmic_version=3.4,
        include_signature_subgroups="SKCM")
    dataset.build_full_dataset()
    dataset.save_dataset("example_dataset")

# 2. Load covariates
cov_matrix = pd.read_parquet("example_gtex_skin_covariates.parquet")

# 3. Create model with signature-specific effects
model = Model(
    dataset,
    cov_matrix,
    cov_effects_per_sigma=True,
    signature_selection=['SBS7a', 'SBS7b'],
    prob_g_tau_tau_independent=False)

# 4. Estimate effects
model.estimate_cov_effects()

# 5. Infer selection
top_genes = dataset.gene_counts.head(10).index
for gene in top_genes:
    model.estimate_gamma(gene, level='gene')

# 6. Plot results
model.plot_gamma_results(
    save='top_genes_selection.png',
    show=True)
```

## Tips and Best Practices

### Choosing Model Parameters

- **prob_g_tau_tau_independent**:
  - `True`: Simpler model, faster, good for baseline models
  - `False`: More flexible, better with covariates

- **cov_effects_per_sigma**:
  - `False`: One effect per covariate (default)
  - `True`: Separate effects per signature (more parameters)

### Covariate Matrix Format

Your covariate matrix should:
- Have `ensembl_gene_id` as the index
- Contain numeric covariate columns
- Be aligned with dataset genes

Example covariate sources:
- GTEx: Tissue-specific gene expression
- ENCODE: Chromatin marks (H3K4me3, etc.)
- Replication timing
- GC content, gene length

### Performance Tips

- Save datasets after `build_full_dataset()` to avoid reprocessing
- Use `force_generation=False` to reuse signature decomposition
- For large datasets, consider signature selection to reduce parameters

## Troubleshooting

### "No valid MAF files"
- Ensure MAF files have correct format (tab-separated with headers)
- Check that `Variant_Type` column exists

### "Gene not found"
- Verify gene symbols/IDs are in `dataset.mutation_db`
- Use `dataset.gene_counts` to see available genes

### Memory issues
- Process MAF files in batches
- Use `signature_selection` to limit signatures
- Reduce covariate matrix size

## Further Reading

- [README.md](README.md): Installation and quick start
- [SETUP_GUIDE.md](SETUP_GUIDE.md): Detailed setup instructions
- [examples/](examples/): Example scripts and data
- [COSMIC Signatures](https://cancer.sanger.ac.uk/signatures/): Mutational signature database

## Example Data

Example files referenced in this tutorial:
- `example_gdc_manifest.txt`: 5 SKCM samples from TCGA
- `example_gtex_skin_covariates.parquet`: GTEx skin tissue expression

Download from:
```bash
wget https://raw.githubusercontent.com/alfaromurillo/sigmutsel/master/examples/example_gdc_manifest.txt
wget https://raw.githubusercontent.com/alfaromurillo/sigmutsel/master/examples/example_gtex_skin_covariates.parquet
```
