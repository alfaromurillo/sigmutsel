# Setup guide

sigmutsel needs reference data that is too large to ship inside the
package. This page says what is downloaded, where it goes, and how to
check it.

## In short

```bash
pip install git+https://github.com/alfaromurillo/sigmutsel.git
sigmutsel-setup        # or: python -m sigmutsel setup
sigmutsel-check-data   # or: python -m sigmutsel check-data
```

`sigmutsel-check-data` should report every file as `Found`. Setup is
idempotent: a rerun downloads only what is missing (`--force`
downloads the data files again).

## What is downloaded

| step | file | size on disk | used for |
|---|---|---|---|
| 1 | `hgnc_complete_set.txt` (HGNC) | 20 MB | gene symbols and ids |
| 2 | `Homo_sapiens.GRCh38.cds.all.fa` (Ensembl 114) | 130 MB | coding sequences, opportunities |
| 3 | `gencode.v38.annotation.gtf.gz` | 47 MB | transcripts (GRCh38) |
| 4 | `gencode.v19.annotation.gtf.gz` | 38 MB | the exome target's genes (hg19) |
| 5 | `gaf_20111020Plusbroad_wex_1.1_hg19.bed` (MC3 exome target, from the GDC) | 5 MB | the gene universe |
| 6 | `hg38ToHg19.over.chain.gz` (UCSC) | 1 MB | lifting positions to hg19 |
| 7 | `germline_mask/` (gnomAD v2.1.1 exomes, AF > 5e-5) | 15 MB | the germline mask |
| 8 | `rmsk_hg38_merged.bed.xz` (RepeatMasker) | 17 MB | call QC and the territory |
| 9 | `wes_target_gene_ids_gencode19.txt` | < 1 MB | derived from 4 and 5 |
| 10 | SigProfilerMatrixGenerator's GRCh38 genome | 2.9 GB | signatures, genome contexts |

Steps 1-9 go to the **data directory**; step 10 goes inside the
SigProfilerMatrixGenerator package of the same environment (that is
where it reads it). The genome took 18 minutes on a 100 Mb/s
connection; the rest under a minute.

**Do not skip the genome** (`--skip-genome`) on a machine that builds
datasets. Without it nothing fails: every base read from the genome
is "unknown", genome-dependent contexts fall back, and the rates
differ from those of an environment that has it. A warning,
`RuntimeWarning: No installed GRCh38 genome`, is the only sign. To
install it later:

```bash
python -c "from sigmutsel.setup import install_reference_genome; install_reference_genome()"
```

Two more files are downloaded on first use, not by setup: TCGA's
pan-cancer ABSOLUTE purity table and copy-number segments (for the
TCGA sample QC, `sigmutsel.setup.download_tcga_absolute_purity` and
`download_tcga_absolute_segments`, about 250 MB together). The
GENCODE files, the exome target, the RepeatMasker regions and the
ABSOLUTE files are also looked for in a shared location,
`$XDG_DATA_HOME` (default `~/.local/share`), and reused from there if
found, so packages that use the same files download them once.

The small reference tables shipped with the package (signature
inclusion/exclusion by cancer type, COSMIC signature aetiologies, the
Cancer Gene Census, IntOGen and published driver lists, AID targets,
histone genes, artifact-prone genes) are described in
`src/sigmutsel/data/README.md`.

## Where the data directory is

By default, the `data/` directory inside the installed package. To
use another one (a shared disk, a directory you can write to):

```bash
sigmutsel-setup --data-dir ~/sigmutsel_data
export SIGMUTSEL_DATA_DIR=~/sigmutsel_data
```

The environment variable must be set in **every** session that uses
the package (put the `export` in your shell's startup file):
sigmutsel reads it when it is imported. The files shipped with the
package are found in the package itself whatever the data directory
is.

## Options

```bash
sigmutsel-setup --force            # download the data files again
sigmutsel-setup --skip-genome      # leave out the 3 GB genome
sigmutsel-setup --decompress-gtf   # GTFs uncompressed (3 GB more)
sigmutsel-setup --repeatmasker-full    # UCSC's full rmsk table (149 MB; same QC)
sigmutsel-setup --germline-mask-full   # every gnomAD allele (streams ~92 GB;
                                       # only for a mask below AF 5e-5)
```

`--keep-fasta-compressed` keeps the CDS FASTA as `.fa.gz`; building a
dataset reads the uncompressed file, so use it only to decompress
later.

## From Python

```python
from sigmutsel.setup import download_all, install_reference_genome

paths = download_all()            # steps 1-10
paths = download_all(genome=False)
install_reference_genome("GRCh38")

from sigmutsel import locations

locations.print_data_status()     # what sigmutsel-check-data prints
locations.list_data_files()       # {name: found}
locations.location_cds_fasta      # every file's path
```

## Troubleshooting

- **`FileNotFoundError: ... not found at ... python -m sigmutsel
  setup`**: a file is missing from the data directory. Run
  `sigmutsel-check-data`; if you set `SIGMUTSEL_DATA_DIR`, check it is
  set in this session too.
- **A download fails**: rerun; finished files are skipped. Behind a
  proxy, the URLs are in `sigmutsel.setup.DOWNLOAD_URLS`, and a file
  downloaded by hand can be placed in the data directory under the
  name in the table above.
- **No permission to write the package directory**: use
  `--data-dir` (above).
- **The genome download stalls**: it comes from the Sanger Institute's
  FTP server through SigProfilerMatrixGenerator; rerun
  `install_reference_genome()`.
