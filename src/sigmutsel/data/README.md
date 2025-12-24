# Sigmutsel Reference Data

This directory contains reference data files used by sigmutsel.

## File Categories

### Small Files (Included in Package)

These files are included in the package distribution:

- `exclusion_signatures_matrix_by_cancer_type.txt` - Signatures to exclude by cancer type
- `inclusion_signatures_matrix_by_cancer_type.txt` - Signatures to include by cancer type
- `Census_allFri_May_30_16_47_00_2025.tsv` - Cancer Gene Census
- `Cosmic_CancerGeneCensus_v101_GRCh38.tsv` - COSMIC Cancer Gene Census


### Large Files (Downloaded Separately)

These files are downloaded on first use or via `python -m sigmutsel setup`:

- `hgnc_complete_set.txt` - HGNC gene name mappings (~20 MB)
- `Homo_sapiens.GRCh38.cds.all.fa` - Ensembl CDS sequences (~100 MB)
- `gencode.v38.annotation.gtf.gz` - GENCODE v38 annotations (~50 MB compressed)
- `gencode.v19.annotation.gtf.gz` - GENCODE v19 annotations (~40 MB compressed)

## Downloading Reference Files

### Automatic Download

Run the setup script:

```bash
python -m sigmutsel setup
```

Options:
```bash
# Re-download even if files exist
python -m sigmutsel setup --force

# Keep FASTA compressed (saves disk space)
python -m sigmutsel setup --keep-fasta-compressed

# Decompress GTF files (for faster parsing)
python -m sigmutsel setup --decompress-gtf

# Use custom data directory
python -m sigmutsel setup --data-dir /path/to/data
```

### Programmatic Download

```python
from sigmutsel.setup import download_all

# Download all files
download_all()

# Download specific file
from sigmutsel.setup import download_hgnc, download_cds_fasta

download_hgnc()
download_cds_fasta(decompress=True)
```

### Manual Download

Alternatively, download files manually and place them here:

1. **HGNC Complete Set**
   - URL: https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt
   - Save as: `hgnc_complete_set.txt`

2. **Ensembl CDS FASTA**
   - URL: https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz
   - Save as: `Homo_sapiens.GRCh38.cds.all.fa` (decompress) or keep as `.fa.gz`

3. **GENCODE GTF v38**
   - URL: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz
   - Save as: `gencode.v38.annotation.gtf.gz`

4. **GENCODE GTF v19**
   - URL: https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz
   - Save as: `gencode.v19.annotation.gtf.gz`

## Custom Data Directory

To use a different data directory, set the environment variable:

```bash
export SIGMUTSEL_DATA_DIR=/path/to/your/data
```

Then run the setup script or use sigmutsel normally.

## Checking File Status

Check which files are available:

```python
from sigmutsel.locations import print_data_status

print_data_status()
```

Output:
```
Data directory: /path/to/sigmutsel/data

File status:
------------------------------------------------------------
✓ exclusion_signatures_matrix  Found
✓ inclusion_signatures_matrix  Found
✓ hgnc_complete_set            Found
✗ cancer_gene_census           Missing
✗ cosmic_cancer_gene_census    Missing
✓ cds_fasta                    Found
✓ gencode38_annotation         Found
✓ gencode19_annotation         Found

To download missing files, run:
    python -m sigmutsel setup
```

## Disk Space Requirements

Approximate sizes:

| File              | Compressed  | Uncompressed |
|-------------------|-------------|--------------|
| HGNC complete set | 8 MB        | 20 MB        |
| CDS FASTA         | 40 MB       | 130 MB       |
| GENCODE v38 GTF   | 50 MB       | 1.7 GB       |
| GENCODE v19 GTF   | 40 MB       | 1.4 GB       |
| **Total**         | **~140 MB** | **~3.3 GB**  |

**Recommendation**: Keep GTF files compressed to save ~3 GB of disk space. Most tools can read `.gtf.gz` files directly.

## License & Attribution

Reference data files are subject to their original licenses:

- **HGNC**: CC0 1.0 (public domain)
- **Ensembl**: Apache 2.0
- **GENCODE**: Multiple licenses, see https://www.gencodegenes.org/pages/data_access.html
- **COSMIC**: Academic use only, requires license for commercial use

Please cite the original sources when using these data.
