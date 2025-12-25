# sigmutsel

**Signature based mutation rate estimation and selection inference in cancer**

## Overview

`sigmutsel` is a Python package for estimating mutation rates and inferring selection coefficients from tumor sequencing data. It provides tools to:

- Build mutation datasets from MAF (Mutation Annotation Format) files
- Perform mutational signature decomposition using COSMIC signatures
- Estimate baseline mutation rates across genomic contexts
- Model covariate effects on mutation rates
- Infer selection coefficients at gene and variant levels using Bayesian inference

## Installation

### Recommended: Use a Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create a virtual environment
python -m venv sigmutsel_env

# Activate it
source sigmutsel_env/bin/activate  # On Linux/Mac
# OR
sigmutsel_env\Scripts\activate     # On Windows
```

### Install from GitHub

```bash
pip install git+https://github.com/alfaromurillo/sigmutsel.git

# Download reference data files
python -m sigmutsel setup
```

### Install from Source

```bash
git clone https://github.com/alfaromurillo/sigmutsel.git
cd sigmutsel
pip install -e .

# Download reference data files (required)
python -m sigmutsel setup
# Or use the installed command:
sigmutsel-setup
```

### For development

```bash
pip install -e ".[dev]"
python -m sigmutsel setup
```

### Reference Data

Sigmutsel requires several reference data files. Small files are
included in the package, while large files (FASTA, GTF, HGNC) are
downloaded separately:

```bash
# Download all reference files (~140 MB compressed, ~3.3 GB uncompressed)
sigmutsel-setup

# Check which files are available
sigmutsel-check-data

# Keep files compressed to save space
sigmutsel-setup --keep-fasta-compressed  # Saves 90 MB

# Use custom data directory
export SIGMUTSEL_DATA_DIR=/path/to/data
sigmutsel-setup --data-dir $SIGMUTSEL_DATA_DIR
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions.

## Quick Start

```python
from sigmutsel import MutationDataset, Model

# 1. Create dataset from MAF files
dataset = MutationDataset(
    location_maf_files="maf_files",
    signature_class="SBS")

# 2. Run signature decomposition
dataset.run_signature_decomposition(
    genome_build='GRCh38',
    exome=True,
    cosmic_version=3.4)

# 3. Build full dataset
dataset.build_full_dataset()

# 4. Create model and estimate mutation rates
model = Model(dataset, cov_matrix=None)
model.compute_mu_gs()

# 5. Estimate selection
model.estimate_gamma('TP53', level='gene')
model.estimate_gamma('BRAF p.V600E', level='variant')

# 6. Plot results
model.plot_gamma_results(save='selection.png')
```

See [TUTORIAL.md](TUTORIAL.md) for a detailed walkthrough including covariates, PCA, and signature-specific effects.

## Downloading TCGA Data

sigmutsel includes utilities for downloading TCGA MAF files from the Genomic Data Commons (GDC).

### Getting a GDC Manifest

1. Visit the [GDC Data Portal](https://portal.gdc.cancer.gov/)
2. Use the filters to select your cohort (e.g., TCGA-COAD)
3. For a SBS analysis, in the Files tab, filter for:
   - Data Category: Simple Nucleotide Variation
   - Data Format: MAF
   - Workflow Type: Aliquot Ensemble Somatic Variant Merging and Masking
4. Add files to cart and download the manifest file

### Installing GDC Data Transfer Tool

The download utility requires the `gdc-client` command-line tool:

```bash
# Download from GDC
# Visit: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

# On Linux/Mac, make executable and add to PATH
chmod +x gdc-client
sudo mv gdc-client /usr/local/bin/
```

### Download MAF Files Using Manifest

**Command-line usage:**
```bash
python -m sigmutsel.download_tcga_data \
    --manifest path/to/gdc_manifest.txt \
    --destination path/to/output/directory
```

**Programmatic usage:**
```python
from sigmutsel.download_tcga_data import download_tcga_maf_files

# Download and decompress MAF files
download_tcga_maf_files(
    gdc_manifest="path/to/gdc_manifest.txt",
    destination="path/to/output/directory"
)
```

**Example:** See `examples/example_gdc_manifest.txt` for a sample manifest file with 5 SKCM samples.

The utility will:
- Download files using `gdc-client`
- Automatically decompress `.maf.gz` files
- Place uncompressed MAF files in the destination directory
- Clean up temporary files

## Package Structure

The core of sigmutsel is in a single `models.py` file containing:
- **MutationDataset**: Data loading and processing
- **Model**: Signature based mutation rate estimation and selection inference in cancer

This keeps related functionality together and simplifies imports.

## Features

### MutationDataset

- Load and validate MAF files
- Perform mutational signature decomposition
- Build mutation presence matrices
- Count genomic contexts per gene
- Extract variant-level information

### Model

- Estimate baseline mutation rates per genomic type
- Model covariate effects on mutation rates
- Support for multi-signature analysis
- Bayesian inference of selection coefficients
- Comprehensive plotting and visualization

## Dependencies

- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- pymc >= 5.0.0
- arviz >= 0.15.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Related Packages

- [sigmutselcovs](https://github.com/alfaromurillo/sigmutselcovs): Tools for creating covariate matrices for use with sigmutsel

## License

MIT License - see LICENSE file for details

## Citation

If you use this software in your research, please cite:

```
[Citation to be added]
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please use the [GitHub issue tracker](https://github.com/alfaromurillo/sigmutsel/issues).
