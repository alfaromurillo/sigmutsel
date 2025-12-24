# Setup Guide for Sigmutsel

This guide explains how to set up sigmutsel with its reference data files.

## Quick Start

```bash
# 1. Install the package
cd /home/jorge/documents/projects/sigmutsel
pip install -e ".[dev]"

# 2. Download reference data
python -m sigmutsel setup

# Or use the installed command:
sigmutsel-setup

# 3. Check data status
python -m sigmutsel check-data
# Or:
sigmutsel-check-data
```

## Package Data Structure

sigmutsel uses a data directory to store reference files:

```
sigmutsel/src/sigmutsel/data/
├── README.md                              # Documentation
├── exclusion_signatures_matrix_*.txt      # Included in package
├── inclusion_signatures_matrix_*.txt      # Included in package
├── hgnc_complete_set.txt                  # Downloaded (20 MB)
├── Homo_sapiens.GRCh38.cds.all.fa         # Downloaded (130 MB)
├── gencode.v38.annotation.gtf.gz          # Downloaded (50 MB)
└── gencode.v19.annotation.gtf.gz          # Downloaded (40 MB)
```

### Small Files (Bundled)

These are included in the package distribution:
- Exclusion/inclusion signature matrices (~small)

### Large Files (Downloaded)

These are downloaded on first use:
- HGNC complete set
- Ensembl CDS FASTA
- GENCODE GTF annotations

## Installation Options

### Option 1: Standard Installation with Download

```bash
# Install package
pip install -e .

# Download all reference data
python -m sigmutsel setup
```

### Option 2: Custom Data Directory

Use a custom location for data files:

```bash
# Set environment variable
export SIGMUTSEL_DATA_DIR=/path/to/your/data

# Download to custom location
python -m sigmutsel setup --data-dir $SIGMUTSEL_DATA_DIR

# Package will use this location automatically
```

### Option 3: Manual Download

Download files manually and place them in the data directory:

```bash
cd /path/to/sigmutsel/src/sigmutsel/data

# Download HGNC
wget https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt

# Download CDS FASTA
wget https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz
gunzip Homo_sapiens.GRCh38.cds.all.fa.gz

# Download GENCODE GTF
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gtf.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz
```

## Download Options

### Command-Line Options

```bash
# Basic download
sigmutsel-setup

# Re-download even if files exist
sigmutsel-setup --force

# Keep FASTA compressed (saves 90 MB)
sigmutsel-setup --keep-fasta-compressed

# Decompress GTF files (uses 3 GB more space)
sigmutsel-setup --decompress-gtf

# Use custom directory
sigmutsel-setup --data-dir /custom/path
```

### Programmatic Download

```python
from sigmutsel.setup import download_all, download_hgnc

# Download all files
download_all()

# Download specific files
download_hgnc(force=False)
download_cds_fasta(decompress=True)
download_gencode_gtf(version="38", keep_compressed=True)
```

## Using the Package

### Accessing Reference Data Locations

```python
from sigmutsel import locations

# Access file paths
hgnc_file = locations.location_hgnc_complete_set
fasta_file = locations.location_cds_fasta

# Check if file exists (raises helpful error if not)
locations.check_data_file(hgnc_file, "HGNC complete set")

# List all files and their status
status = locations.list_data_files()
print(status)

# Print formatted status
locations.print_data_status()
```

### Example Usage

```python
from sigmutsel import MutationDataset, locations

# Locations are automatically available
dataset = MutationDataset(
    maf_directory="/path/to/maf/files",
    signature_class="SBS"
)

# The package knows where to find reference files
dataset.build_full_dataset()  # Uses locations internally
```

## Disk Space Requirements

| Files                       | Compressed | Uncompressed |
|-----------------------------|------------|--------------|
| Package (code + small data) | 5 MB       | 10 MB        |
| HGNC                        | 8 MB       | 20 MB        |
| CDS FASTA                   | 40 MB      | 130 MB       |
| GTF v38 + v19               | 90 MB      | 3.1 GB       |
| **Total Minimal**           | **145 MB** | **3.3 GB**   |

**Recommendations:**
- Keep GTF files compressed (saves ~3 GB)
- Most tools can read `.gtf.gz` directly
- Only decompress if you need faster random access

## Checking Installation

### Verify Package Installation

```python
import sigmutsel
print(sigmutsel.__version__)

from sigmutsel import MutationDataset, Model
print("✓ Core classes imported successfully")
```

### Verify Data Files

```bash
# Check data status
sigmutsel-check-data

# Or in Python:
python -c "from sigmutsel.locations import print_data_status; print_data_status()"
```

Expected output:
```
Data directory: /path/to/sigmutsel/src/sigmutsel/data

File status:
------------------------------------------------------------
✓ exclusion_signatures_matrix  Found
✓ inclusion_signatures_matrix  Found
✓ hgnc_complete_set            Found
✓ cds_fasta                    Found
✓ gencode38_annotation         Found
✓ gencode19_annotation         Found
```

## Troubleshooting

### Files Not Found Error

If you get `FileNotFoundError` when using the package:

```python
FileNotFoundError: hgnc_complete_set.txt not found at ...
Download it using:
    python -m sigmutsel setup
```

**Solution**: Run the download command:
```bash
python -m sigmutsel setup
```

### Download Fails

If downloads fail due to network issues:

1. **Manual download**: Download files from URLs in `data/README.md`
2. **Partial retry**: The download script can be re-run, it skips existing files
3. **Custom mirror**: Modify URLs in `sigmutsel/setup.py` if needed

### Permission Errors

If you can't write to the package data directory:

```bash
# Use custom directory
export SIGMUTSEL_DATA_DIR=~/sigmutsel_data
mkdir -p $SIGMUTSEL_DATA_DIR
python -m sigmutsel setup
```

### Large Files

If disk space is limited:

```bash
# Keep FASTA compressed (saves 90 MB)
sigmutsel-setup --keep-fasta-compressed

# GTF files stay compressed by default (saves 3 GB)
```

## For Developers

### Adding New Reference Files

1. Add download URL to `sigmutsel/setup.py`:
   ```python
   DOWNLOAD_URLS["new_file.txt"] = "https://..."
   ```

2. Add location to `sigmutsel/locations.py`:
   ```python
   location_new_file = DATA_DIR / "new_file.txt"
   ```

3. Update `download_all()` function

4. Update `list_data_files()` function

5. Document in `data/README.md`

### Package Distribution

Small files are included in the package wheel:
- Signature matrices
- Small reference tables

Large files are excluded and downloaded separately:
- FASTA files
- GTF annotations
- HGNC complete set

This keeps the package distribution small (~10 MB) while providing
easy access to required reference data.

## See Also

- `data/README.md` - Detailed data file documentation
- `sigmutsel/locations.py` - File path definitions
- `sigmutsel/setup.py` - Download implementation
