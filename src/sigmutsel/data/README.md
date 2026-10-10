# Sigmutsel Reference Data

This directory contains reference data files used by sigmutsel.

## File Categories

### Small Files (Included in Package)

These files are included in the package distribution:

- `exclusion_signatures_matrix_by_cancer_type.txt` - Signatures to exclude by cancer type
- `inclusion_signatures_matrix_by_cancer_type.txt` - Signatures to include by cancer type
- `Census_allFri_May_30_16_47_00_2025.tsv` - Cancer Gene Census
- `Cosmic_CancerGeneCensus_v101_GRCh38.tsv` - COSMIC Cancer Gene Census
- `intogen_driver_genes_20240920.tsv` - IntOGen's compendium of mutational driver genes (release 2024-09-20, CC0; Martínez-Jiménez et al. 2020, Nat Rev Cancer), one row per gene with its Ensembl id resolved through HGNC; unioned with the Census as the known cancer genes that passenger-based fits exclude
- `aid_target_genes.tsv` - genes targeted by activation-induced cytidine deaminase, which passenger-based fits also exclude: every GENCODE v38 gene of an `IG_*` biotype (the immunoglobulin loci, 414), IGLL5 (in the lambda locus; Kasar et al. 2015, Nat Commun), and the aberrant somatic hypermutation targets of Hadj Khodabakhshi et al. 2012 (Oncotarget), Table 1 (42 of 44; two GenBank transcript ids do not resolve), symbols updated to GENCODE v38 (HIST1H2AC -> H2AC6, LLT1 -> CLEC2D, LRMP -> IRAG2); columns `symbol`, `ensembl_gene_id`, `gene_type`, `source`
- `published_driver_genes.tsv` - driver genes of three published exome-wide screens, unioned with the Census and IntOGen as known cancer genes: TCGA's PanCancer consensus (Bailey et al. 2018, Cell, Table S1, 299 genes), diffuse large B-cell lymphoma drivers of Reddy et al. 2017 (Cell, Table S1, 150 genes) and Chapuy et al. 2018 (Nat Med, Table S3a, MutSig2CV q < 0.1, 98 genes); 419 genes, Ensembl ids resolved through HGNC (current symbol, else a unique previous symbol); columns `symbol`, `ensembl_gene_id`, `symbol_in_source`, `ensembl_from`, `source`
- `histone_genes.tsv` - every gene of HGNC's H1, H2A, H2B, H3 and H4 histone gene groups (ids 1935-1939; 117 genes), which passenger-based fits exclude: mutated histones can be drivers (oncohistones), and the clustered paralogs mis-map; columns `symbol`, `ensembl_gene_id`, `locus_type`, `gene_group`


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
