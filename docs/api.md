# sigmutsel API reference

Generated from the code by `scripts/build_api_reference.py`; do not edit by hand. Signatures and defaults are the code's.

**Defaults are not always the recommended settings.** Several defaults are kept for older analyses; the tutorial (`tutorials/selection_in_hnsc.md`) passes the recommended ones, and the README lists them.

## Contents

- [Setup and data](#setup-and-data)
- [Building a dataset](#building-a-dataset)
- [Calibration steps](#calibration-steps)
- [The rate model](#the-rate-model)
- [Rates](#rates)
- [Selection](#selection)
- [Every other public name](#every-other-public-name)

## Setup and data

### `setup.download_all`

```python
download_all(
    force: bool = False,
    decompress_fasta: bool = True,
    keep_gtf_compressed: bool = True,
    genome: bool = True,
)
```

```text
Download all required reference files.

Parameters
----------
force : bool, default False
    If True, re-download even if files exist
decompress_fasta : bool, default True
    If True, decompress FASTA file
keep_gtf_compressed : bool, default True
    If True, keep GTF files compressed
genome : bool, default True
    Also install SigProfilerMatrixGenerator's GRCh38 genome
    (:func:`install_reference_genome`, about 3 GB), which it keeps
    inside its own package directory, not in the data directory.

Returns
-------
dict[str, Path]
    Mapping of file type to downloaded path

Examples
--------
From the command line, ``sigmutsel-setup`` does the same (and
``sigmutsel-setup --data-dir DIR`` into another directory):

>>> from sigmutsel.setup import download_all
>>> paths = download_all()  # about 300 MB plus the 3 GB genome
>>> sorted(paths)[:3]
['cds_fasta', 'gencode19', 'gencode38']
```

### `setup.install_reference_genome`

```python
install_reference_genome(build: str = 'GRCh38', force: bool = False)
```

```text
Install SigProfilerMatrixGenerator's reference genome (~3 GB).

The signature decomposition and every step that reads the genome
(pentanucleotide contexts, splice-flank compositions, the
opportunity across exon junctions) use SigProfilerMatrixGenerator's
installed reference. Without it those steps do not fail: they
warn once and fall back, and the rates differ from an environment
that has it. Install it before building a dataset.

Parameters
----------
build : str, default "GRCh38"
    Genome build, as SigProfilerMatrixGenerator names it.
force : bool, default False
    Reinstall even if the genome is already installed.

Returns
-------
Path
    The installed genome's directory
    (``references/chromosomes/tsb/<build>`` inside the
    SigProfilerMatrixGenerator package).

Examples
--------
>>> from sigmutsel.setup import install_reference_genome
>>> install_reference_genome("GRCh38")  # doctest: +SKIP
```

### `locations.print_data_status`

```python
print_data_status()
```

```text
Print status of all reference data files.

Returns
-------
None
    Prints one line per reference file, then the command that
    downloads the missing ones, if any.

Examples
--------
Also the console command ``sigmutsel-check-data``:

>>> from sigmutsel.locations import print_data_status
>>> print_data_status()
Data directory: .../sigmutsel/data
<BLANKLINE>
File status:
------------------------------------------------------------
✓ exclusion_signatures_matrix    Found
...
```

### `download_tcga_data.download_project_maf_files`

```python
download_project_maf_files(
    project: str,
    destination: pathlib.Path | str,
    download_dir: pathlib.Path | str | None = None,
    workers: int = 6,
)
```

```text
Download a TCGA project's open MAFs from the GDC, no ``gdc-client``.

Searches the GDC for the project's per-aliquot somatic-mutation
MAFs (gdcfetch's ``somatic-mutations`` preset: open-access
"Masked Somatic Mutation" files of the "Aliquot Ensemble Somatic
Variant Merging and Masking" workflow), downloads them with
gdcfetch over HTTPS, and decompresses each into ``destination``,
the flat directory of ``*.maf`` files that
:class:`~sigmutsel.MutationDataset` reads.

The compressed downloads are kept in ``download_dir``, so a rerun
after an interrupted download fetches only what is missing (and
checks every file's size and MD5). The GDC holds one MAF per
sequenced aliquot, including recurrent and metastatic samples and
several aliquots of one patient: select the files with
:func:`sigmutsel.tcga_sample_selection.select_tcga_maf_files`
before building a dataset.

Parameters
----------
project : str
    TCGA study code, with or without the ``TCGA-`` prefix
    (``"HNSC"`` or ``"TCGA-HNSC"``).
destination : Path or str
    Directory for the decompressed ``*.maf`` files; created if
    missing.
download_dir : Path or str, optional
    Where the compressed downloads go, laid out as
    ``<download_dir>/<file_id>/<file_name>``. Default: a sibling
    of ``destination`` named ``<destination>_gdc``.
workers : int, default 6
    Parallel downloads.

Returns
-------
list of Path
    The decompressed MAF files in ``destination``, sorted.

Raises
------
ValueError
    If the GDC lists no MAF for ``project``.
OSError
    If some downloads still fail after gdcfetch's retries; rerun
    to fetch only those.

Examples
--------
>>> from sigmutsel.download_tcga_data import (
...     download_project_maf_files)
>>> mafs = download_project_maf_files(  # doctest: +SKIP
...     "HNSC", "hnsc/all_maf_files_raw")
>>> len(mafs)  # doctest: +SKIP
511
```

### `download_tcga_data.download_tcga_maf_files`

```python
download_tcga_maf_files(
    gdc_manifest: pathlib.Path | str,
    destination: pathlib.Path | str,
)
```

```text
Download the MAFs of a GDC manifest via ``gdc-client``, unpacked.

Needs NCI's ``gdc-client`` executable on ``PATH``. To download a
whole TCGA project without it, use
:func:`download_project_maf_files`.

Parameters
----------
gdc_manifest : Path or str
    Path to the manifest file used by ``gdc-client`` (from the GDC
    Data Portal's cart, or ``gdcfetch manifest``).
destination : Path or str
    Directory where all decompressed ``*.maf`` files will be saved.

Returns
-------
bool
    ``True`` if every downloaded bundle contained exactly one MAF,
    ``False`` otherwise.

Examples
--------
>>> from sigmutsel.download_tcga_data import download_tcga_maf_files
>>> download_tcga_maf_files(  # doctest: +SKIP
...     "gdc_manifest.txt", "maf_files")
True
```

### `tcga_sample_selection.select_tcga_maf_files`

```python
select_tcga_maf_files(
    maf_dir: str | pathlib.Path,
    keep_sample_types: collections.abc.Iterable[str] = frozenset({'01', '03'}),
    duplicate_policy: str = 'keep_all',
    exclude_prior_treatment: bool = False,
    project: str | None = None,
    random_seed: int | None = None,
)
```

```text
Select which MAF files in an ``all_maf_files/`` directory to use.

Combines sample-type filtering, per-case duplicate resolution,
and (optionally) prior-treatment exclusion into one call. See the
module docstring for the rationale, and
:func:`fetch_gdc_case_metadata` for what GDC actually reports
(dates and treatment flags are often missing).

Parameters
----------
maf_dir : str or pathlib.Path
    Directory of single-aliquot ``.maf`` files.
keep_sample_types : iterable of str, default {"01", "03"}
    See :func:`filter_by_sample_type`.
duplicate_policy : {"keep_all", "oldest", "newest", "random"}, default "keep_all"
    See :func:`select_one_per_case`. ``"oldest"``/``"newest"``
    trigger a GDC API call (needs *project* only for logging
    context, not for the query itself).
exclude_prior_treatment : bool, default False
    If ``True``, drop every file belonging to a case whose
    primary diagnosis has GDC ``prior_treatment`` ``"Yes"``
    (gdcfetch reads the primary diagnosis since c24ae4f; before,
    it read whichever diagnosis GDC listed first, often a
    recurrence or metastasis whose treatment followed the
    primary's collection). This is a
    case-level flag, not sample-level -- GDC does not reliably
    record which specific aliquot was collected relative to a
    treatment's start date (see :func:`fetch_gdc_case_metadata`),
    so this cannot distinguish a pre-treatment primary resection
    from a post-treatment sample within the same case. Cases with
    an unreported (``None``) prior_treatment status are kept.
project : str or None
    TCGA project code (e.g. ``"SKCM"``), used only for the
    mostly-non-primary-cohort warning in
    :func:`filter_by_sample_type`.
random_seed : int or None
    Seed for ``duplicate_policy="random"``.

Returns
-------
list[pathlib.Path]
    Selected MAF file paths, sorted.

Examples
--------
The defaults keep every aliquot; for one treatment-naive primary
tumor per patient, pass:

>>> from sigmutsel.tcga_sample_selection import select_tcga_maf_files
>>> chosen = select_tcga_maf_files(
...     "hnsc/all_maf_files_raw",
...     duplicate_policy="newest",
...     exclude_prior_treatment=True,
...     project="HNSC")
>>> len(chosen)  # of 511 HNSC files in 2026
410
```

## Building a dataset

### `models.MutationDataset`

```python
MutationDataset(
    location_maf_files: str,
    signature_class: str = 'SBS',
    source_maf: str | None = None,
    _mutation_db: pandas.DataFrame = None,
    _genes_present: pandas.DataFrame = None,
    _genes_present_non_silent: pandas.DataFrame = None,
    _genes_present_silent: pandas.DataFrame = None,
    _genes_counts_silent: pandas.DataFrame = None,
    _genes_counts_non_silent: pandas.DataFrame = None,
    _variant_db: pandas.DataFrame = None,
    _variants_present: pandas.DataFrame = None,
    _sig_assignments: pandas.DataFrame = None,
    _signature_matrix: pandas.DataFrame = None,
    _signature_reference_genome: str | None = None,
    _signature_exome: bool | None = None,
    _signature_cosmic_version: float | None = None,
    _signature_genome_build: str | None = None,
    _contexts_by_gene: pandas.DataFrame = None,
    _contexts_by_gene_gene_universe: str | None = None,
    _contexts_by_gene_syn: pandas.DataFrame = None,
    _contexts_by_gene_nonsyn: pandas.DataFrame = None,
    _contexts_by_gene_mis: pandas.DataFrame = None,
    _contexts_by_gene_non: pandas.DataFrame = None,
    _contexts_by_gene_spl: pandas.DataFrame = None,
    _contexts_by_gene_esyn: pandas.DataFrame = None,
    _gene_transcripts: pandas.DataFrame = None,
    _channel_universe: dict | None = None,
    _spectrum_shrinkage: dict | None = None,
    _sample_qc_flags: pandas.DataFrame = None,
    _multi_base_db: pandas.DataFrame = None,
    _homdel_db: pandas.DataFrame = None,
    _indel_db: pandas.DataFrame = None,
    _qc_ledger: pandas.DataFrame = None,
    _assay_table: pandas.DataFrame = None,
)
```

```text
Container for mutation and variant data.

This class groups all the raw data that is shared across
different models. Most data is lazy-loaded since not all
analyses require it.

Attributes
----------
location_maf_files : str or Path
    Path to directory containing raw MAF files.
signature_class : str, default "SBS"
    COSMIC signature class. Must be one of:
    - "SBS": Single base substitution signatures
    - "DBS": Doublet base substitution signatures
    - "ID": Insertion/deletion signatures
    - "CN": Copy number signatures
    - "SV": Structural variant signatures
    - "RNA-SBS": RNA single base substitution signatures
source_maf : str or pathlib.Path, optional
    Path to a single multi-sample MAF file. If provided and
    ``location_maf_files`` does not yet contain per-sample
    ``.maf`` files, the file is split automatically on
    dataset creation. Skipped if per-sample files already
    exist.
mutation_db : pd.DataFrame or None
    Mutation database with individual mutation records.
    Columns include: gene, tumor_sample_barcode,
    ensembl_gene_id, chromosome, position, variant, etc.
    Lazy-loaded via generate_mutation_db() or load_dataset().
genes_present : pd.DataFrame or None
    Binary matrix (genes × samples) indicating which genes
    are mutated in which samples. Lazy-loaded.
genes_present_non_silent : pd.DataFrame or None
    Same as genes_present but for non-silent mutations
    only. Lazy-loaded.
genes_present_silent : pd.DataFrame or None
    Same as genes_present but for silent (synonymous)
    mutations only -- the other half of the consequence split,
    and the one that stays selection-free in driver genes too.
    Lazy-loaded.
genes_counts_silent, genes_counts_non_silent : pd.DataFrame or None
    The same two matrices *before* they are censored to 0/1:
    mutation counts per gene per sample. Presence is a censored
    count, and the censoring throws away most of the information
    precisely in the hypermutated samples that carry most of a
    cohort's mutation mass. Lazy-loaded.
variant_db : pd.DataFrame or None
    Table of unique variants annotated with genomic context
    and mutation types. Lazy-loaded via generate_variant_db()
    or load_dataset().
variants_present : pd.DataFrame or None
    Binary matrix (variants × samples) indicating which variants
    are present in which samples. Lazy-loaded.
sig_assignments : pd.DataFrame or None
    Signature assignment matrix (samples × signatures) from
    COSMIC signature decomposition. Each value represents the
    number of mutations attributed to each signature in each
    sample. Lazy-loaded via run_signature_decomposition().
signature_matrix : pd.DataFrame or None
    Normalized signature matrix (mutation types × signatures)
    from COSMIC signature decomposition. Each column represents
    a signature and sums to 1, showing the probability
    distribution over mutation types for that signature.
    Automatically loaded after run_signature_decomposition().
contexts_by_gene : pd.DataFrame or None
    Trinucleotide context counts by gene (genes × contexts). Each
    value represents the count of a specific trinucleotide context
    in a gene's coding sequence. Lazy-loaded via
    generate_contexts_by_gene(), whose `gene_universe` argument
    controls which genes are included ("wes_target", the
    default: the genes in the mutation database and MC3's TCGA
    WES-target gene set, the right denominator for a gene's share
    of a tumor's mutations; "own_cohort": the genes in the
    mutation database only, which inflates every rate). These values are needed to compute later the
    probability that a mutation of a certain type lands on a
    gene. Thus, gene mutation rates can only be obtained for
    genes with this information, and so the index of
    contexts_by_gene is the maximal scope of the analysis.
contexts_by_gene_syn, contexts_by_gene_nonsyn : pd.DataFrame or None
    Consequence-split opportunity counts (genes × 96 canonical SBS
    types): the synonymous and non-synonymous share of the same
    opportunities `contexts_by_gene` counts, so that
    ``syn[τ] + nonsyn[τ] == contexts_by_gene[context(τ)]`` for
    every type τ -- unless a germline mask is set, when the left
    side is smaller and is :attr:`type_opportunity`. SBS-only (see
    generate_consequence_contexts_by_gene()) and lazy-loaded --
    nothing in the default pipeline populates or consumes them
    yet.
signature_reference_genome : str or None
    Reference genome (same as genome_build) used when generating
    mutational matrices for signature decomposition. Populated
    when run_signature_decomposition() succeeds.
signature_exome : bool or None
    Whether exome-normalized signatures were used during
    decomposition. Populated when run_signature_decomposition()
    succeeds.
signature_cosmic_version : float or None
    COSMIC signature release used during decomposition. Populated
    when run_signature_decomposition() succeeds.
signature_genome_build : str or None
    Genome build passed to signature decomposition. Populated when
    run_signature_decomposition() succeeds.

Parameters
----------
location_maf_files : str or Path
    Directory of per-tumor ``.maf`` files (one per tumor;
    select TCGA's with
    :func:`~sigmutsel.tcga_sample_selection.select_tcga_maf_files`).
signature_class : str, default "SBS"
    COSMIC signature class (see Attributes). The rate model and
    selection estimates are built on SBS.
source_maf : str or Path, optional
    One multi-tumor MAF to split into ``location_maf_files``.

Examples
--------
The recommended build (each step explained in the tutorial,
``tutorials/selection_in_hnsc.md``):

>>> from sigmutsel import MutationDataset
>>> from sigmutsel.qc import load_repeat_intervals
>>> from sigmutsel.setup import download_repeatmasker_bed
>>> dataset = MutationDataset("hnsc/all_maf_files", signature_class="SBS")
>>> dataset.generate_mutation_db(
...     qc_mode=True,
...     qc_kwargs={"repeat_intervals": load_repeat_intervals(
...         download_repeatmasker_bed("hg38"))},
...     germline_mask_af=1.5e-4)
>>> dataset.run_two_pass_signature_decomposition(
...     exclude_signature_subgroups="HNSC", treatment_naive=True,
...     genome_build="GRCh38", exome=True, cosmic_version=3.6,
...     treatment_load_threshold=0.2,
...     prevalence_override_min_fraction=0.15)
>>> dataset.build_full_dataset(gene_universe="wes_target",
...                            regenerate_mutation_db=False)
>>> dataset.save_dataset("hnsc/HNSC_dataset")
>>> dataset.n_samples, len(dataset.model_db)
(392, 59453)
```

### `models.MutationDataset.generate_mutation_db`

```python
generate_mutation_db(
    location_gene_set=None,
    channel_universe=True,
    territory='mc3',
    splice_padding=0,
    germline_mask_af=None,
    **kwargs,
)
```

```text
Generate mutation database from MAF files.

This method wraps :func:`load_maf_files.generate_compact_db`
and stores the result in the dataset's _mutation_db attribute.
For SBS, it then places every call in the channel universe
(:meth:`classify_mutation_db`) unless ``channel_universe`` is
False.

For ID signature class, automatically sets seqinfo_dir to
`{location_maf_files}/output/vcf_files/ID/` if not
explicitly provided.

Parameters
----------
location_gene_set : str, Path, or None, default None
    Path to a gene set file for gene name standardization
    (e.g. ``locations.location_hgnc_complete_set``). If None
    (default), gene names are kept as in the MAF files: unlike
    :func:`load_maf_files.generate_compact_db`, whose default
    is the HGNC complete set, this method passes None through,
    which skips the update.
channel_universe : bool, default True
    Classify the calls into the channel universe (SBS only).
    False keeps the pre-universe behaviour, where every call
    is modelled and consequence comes from the MAF.
territory, splice_padding, germline_mask_af
    Forwarded to :meth:`classify_mutation_db`.
**kwargs : dict
    Additional arguments passed to
    :func:`load_maf_files.generate_compact_db`.

    **For ID signature class:**
    - seqinfo_dir : str or Path, optional
        Directory containing *_seqinfo.txt files from
        SigProfilerMatrixGenerator. These files provide
        COSMIC ID-83 mutation type annotations for indels.
        If not provided, automatically uses
        `{location_maf_files}/output/vcf_files/ID/`.

Notes
-----
The signature class used is determined by the dataset's
signature_class attribute, which must be one of the
COSMIC signature classes: SBS, DBS, ID, CN, SV, RNA-SBS.

For ID (indel) signature class, seqinfo files are
automatically located at the standard output location
from SigProfilerMatrixGenerator. You only need to provide
seqinfo_dir explicitly if the files are in a custom
location.

Returns
-------
None
    The calls are in :attr:`mutation_db` (and, with ``qc_mode``,
    what the QC removed in :attr:`qc_ledger`); save with
    :meth:`save_dataset`.

Examples
--------
>>> # SBS calls with the call QC and the germline mask
>>> from sigmutsel.qc import load_repeat_intervals
>>> from sigmutsel.setup import download_repeatmasker_bed
>>> dataset = MutationDataset(
...     location_maf_files="data/maf_files",
...     signature_class="SBS")
>>> dataset.generate_mutation_db(
...     qc_mode=True,
...     qc_kwargs={"repeat_intervals": load_repeat_intervals(
...         download_repeatmasker_bed("hg38"))},
...     germline_mask_af=1.5e-4)
>>>
>>> # Generate ID database (seqinfo_dir auto-detected)
>>> dataset_id = MutationDataset(
...     location_maf_files="data/maf_files",
...     signature_class="ID")
>>> dataset_id.generate_mutation_db()
>>> # seqinfo_dir automatically set to:
>>> # data/maf_files/output/vcf_files/ID/

See Also
--------
load_maf_files.generate_compact_db : Core generation function
```

### `models.MutationDataset.run_two_pass_signature_decomposition`

```python
run_two_pass_signature_decomposition(
    artifact_threshold=0.5,
    force_generation=False,
    exome=None,
    cosmic_version=None,
    genome_build='GRCh38',
    treatment_load_threshold=None,
    prevalence_override_min_fraction=None,
    min_burden_for_diagnostics=100,
    **kwargs,
)
```

```text
Two-pass, diagnostic-aware signature decomposition.

Fits signature decomposition twice rather than once. Pass A is
a fully unrestricted diagnostic fit -- no per-cancer-type
table, no `treatment_naive`, artifact signatures included --
used to compute up to three independent signals before pass B:

1. **Artifact-mutation detection** (always on): each
   mutation's probability mass on
   `constants.ARTIFACT_SIGNATURES`
   (`signature_attribution.compute_signature_probability_mass`),
   used to drop (:func:`qc.flag_artifact_signature_mutations`)
   mutations dominantly attributed to a technical artifact --
   without simply excluding artifact signatures from the fit,
   which would just force their mutations onto whichever real
   signature fits next-best (see this project's plan for the
   full rationale).
2. **Treatment-signature-load sample QC** (opt-in via
   `treatment_load_threshold`): TCGA's "treatment-naive"
   clinical annotation is retrospective self-report and can
   miss real cases. A sample whose *unrestricted* fit
   attributes an implausible fraction of its burden to
   `constants.TREATMENT_ASSOCIATED_SIGNATURES` is dropped
   entirely (:func:`qc.flag_treatment_signature_samples`) --
   sample-level, not mutation-level, because real treatment
   mutagenesis reshapes a tumor's whole subsequent
   evolutionary history, unlike a sequencing artifact. Gated
   by `min_burden_for_diagnostics`: a naive threshold with no
   burden gate is dominated by low-count NNLS noise, not real
   contamination (see this project's plan and TODO.md's
   2026-08-25 groundwork).
3. **Prevalence-based per-cancer-type table override** (opt-in
   via `prevalence_override_min_fraction`): a signature the
   table excludes for this cancer type but that recurs across
   enough of the cohort's own adequately-powered tumors is
   kept in pass B's basis instead
   (`signature_decomposition.compute_prevalence_overrides`) --
   a prior override, not a table replacement, and it never
   touches the treatment or artifact lists (see that
   function's docstring for the full rationale). Computed
   *after* step 2 removes treatment-flagged samples, so a
   contaminated tumor about to be dropped can't inflate the
   apparent prevalence of whatever signature it leaks into.

Then:

- Samples flagged by step 2 and mutations flagged by step 1
  are dropped from `self.mutation_db` -- guaranteeing they're
  excluded from burden, gene presence, and every other
  downstream calculation that reads `mutation_db`, not just
  from pass B's fit.
- **Pass B**: rebuild the SBS96 matrix directly from the
  cleaned `mutation_db` (not the original MAF files -- single
  source of truth with what `mutation_db` now holds) and refit
  with the final exclusion set (per-cancer-type table, net of
  any step-3 overrides, unioned with the treatment list if
  `treatment_naive` and always with `ARTIFACT_SIGNATURES`).
  This becomes the "official" result: ``self.sig_assignments``
  / ``self.signature_matrix`` are overwritten with pass B's,
  and ``self.mutation_db`` is overwritten with the cleaned
  version, once this method returns.

Requires ``self.mutation_db`` to already be built (e.g.
``self.generate_mutation_db(qc_mode=True, ...)``) *before*
calling this -- pass A's diagnostics need it, and it can't be
built afterward without regenerating over the cleaned data.
When wiring this into :meth:`build_full_dataset`, pass
``regenerate_mutation_db=False`` there, since this method
already updated ``mutation_db`` and a default rebuild would
silently overwrite the cleaned version with the original.

**Caching pitfall, found the hard way (2026-08-26): pass A and
pass B cache in *two separate directories*, and
`force_generation=False`'s cache check is existence-only, with
no awareness of `cosmic_version`/exclusion lists/thresholds
changing between calls.** Pass A caches under the standard
``{location_maf_files}/signature_decomposition/{signature_class}/``
directory (shared with any plain :meth:`run_signature_decomposition`
call); pass B caches separately, under a
``..._{signature_class}_artifact_cleaned`` directory. Clearing
or backing up only one of the two before a rerun with
different parameters (e.g. a COSMIC version bump) lets the
*other* one silently serve stale results with no error and no
warning -- caught only because pass B's stale cache reflected
a different set of cleaned tumors than pass A's fresh one,
producing a one-tumor mismatch between ``mutation_db`` and
``sig_assignments`` that surfaced as an all-NaN row deep in
downstream covariate-effect estimation, far from the actual
cause. Any rerun with different fit parameters must clear (or
back up) **both** directories, not just one. Since 2026-09-30
each cache also stores a fingerprint of its input matrix and
settings (:func:`signature_decomposition.signature_decomposition`)
and is refitted when they differ, so a rebuilt matrix -- e.g.
without the calls on germline-masked alleles -- can no longer
load old exposures.

Only supports ``signature_class="SBS"`` (the matrix rebuild
uses `constants.canonical_types_order`, which is SBS96-specific).

Parameters
----------
artifact_threshold : float, default 0.5
    Forwarded to `qc.flag_artifact_signature_mutations`. Not
    derived from theory -- tune against the real distribution
    observed on a few pilot cohorts (see this project's plan).
treatment_load_threshold : float, optional
    Forwarded to `qc.flag_treatment_signature_samples` as
    `threshold`. `None` (default) disables step 2 entirely --
    no sample is dropped for treatment-signature load.
prevalence_override_min_fraction : float, optional
    Forwarded to `signature_decomposition.compute_prevalence_overrides`
    as `min_prevalence`. `None` (default) disables step 3
    entirely -- the per-cancer-type table is used as-is.
min_burden_for_diagnostics : int, default 100
    Minimum fitted mutation count (pass A's per-sample
    assignment total) for a sample to be eligible for any of
    steps 1-3 -- all three read pass A's per-sample fit, which
    is too noisy to trust below this count (see this
    project's plan). Below it no mutation is flagged by step
    1, no sample by step 2, and none contributes to step 3's
    prevalence counts. Step 1 took no gate until 2026-10-04:
    low-burden tumors lost 4-9% of their calls, driver
    hotspots among them.
force_generation, exome, cosmic_version, genome_build :
    Forwarded to both passes (see :meth:`run_signature_decomposition`
    for defaults).
**kwargs : dict
    Forwarded to pass A unchanged, *except*
    `exclude_signature_subgroups` and `treatment_naive`, which
    this method consumes itself (pass A must not receive
    either -- it has to stay fully unrestricted to serve as
    steps 2 and 3's diagnostic fit) and re-applies to pass B
    after resolving step 3's overrides.
    `exclude_signature_subgroups` may be a cancer-type
    shorthand, a `(location, cancer_type)` tuple, or an
    explicit signature list (e.g. main.py's missing-PCAWG-row
    fallback); step 3 only has a table to override against for
    the first two forms. `exclude_artifacts` is set explicitly
    by this method for each pass and must not be passed here.

Returns
-------
pd.DataFrame
    Pass B's signature assignments (also stored in
    ``self.sig_assignments``).

Examples
--------
After :meth:`generate_mutation_db` (QC on), with the settings the
tutorial uses:

>>> dataset.run_two_pass_signature_decomposition(
...     exclude_signature_subgroups="HNSC", treatment_naive=True,
...     genome_build="GRCh38", exome=True, cosmic_version=3.6,
...     artifact_threshold=0.5, treatment_load_threshold=0.2,
...     prevalence_override_min_fraction=0.15,
...     min_burden_for_diagnostics=100)
>>> dataset.sig_assignments.shape  # tumors x signatures
```

### `models.MutationDataset.build_full_dataset`

```python
build_full_dataset(
    fastas=None,
    gene_universe='wes_target',
    regenerate_mutation_db=True,
    **kwargs,
)
```

```text
Run the full data-generation pipeline for this dataset.

Parameters
----------
fastas : str, Path, list, or None, default None
    Forwarded to :meth:`generate_contexts_by_gene`.
gene_universe : {"wes_target", "own_cohort"}, default "wes_target"
    Forwarded to :meth:`generate_contexts_by_gene`.
regenerate_mutation_db : bool, default True
    If False, skip calling :meth:`generate_mutation_db` and
    use whatever is already in ``self.mutation_db`` (raises if
    it hasn't been loaded). For callers that already built
    (and possibly modified) the mutation database themselves
    -- e.g. :meth:`run_two_pass_signature_decomposition`,
    which needs ``mutation_db`` built *before* it can compute
    per-mutation artifact probabilities, and updates it
    in-place with the artifact-mutation-cleaned version
    afterward. Calling this with the default `True` right
    after that would silently regenerate and overwrite the
    cleaned database with the original, uncleaned one.
**kwargs : dict
    Forwarded to :meth:`generate_mutation_db` -- e.g.
    `qc_mode=True` (optionally with `qc_kwargs`) to enable
    the structured QC pipeline in :mod:`qc` instead of the
    default silent validation. Ignored if
    `regenerate_mutation_db` is False.

Returns
-------
None
    The tables are set on the dataset (gene presence, contexts by
    gene, variants); save it with :meth:`save_dataset`.

Examples
--------
After :meth:`run_two_pass_signature_decomposition`:

>>> dataset.build_full_dataset(gene_universe="wes_target",
...                            regenerate_mutation_db=False)
```

### `models.MutationDataset.save_dataset`

```python
save_dataset(directory, overwrite=False)
```

```text
Persist loaded dataset artifacts to a directory.

Parameters
----------
directory : str or Path
    Where to write the dataset.
overwrite : bool, default False
    Replace an existing dataset without asking. The default
    prompts interactively, which is right at a console but
    hangs an unattended pipeline -- pass True from scripts,
    mirroring :meth:`Model.save_model`.

Returns
-------
None

Examples
--------
>>> dataset.save_dataset("hnsc/HNSC_dataset")
>>> dataset.save_dataset("hnsc/HNSC_dataset", overwrite=True)
```

### `models.MutationDataset.load_dataset`

```python
load_dataset(directory)
```

```text
Load dataset artifacts from a directory created by save_dataset.

Parameters
----------
directory : str or Path
    A directory written by :meth:`save_dataset`.

Returns
-------
MutationDataset
    With every table that was saved; the MAF files are not
    needed again. A dataset saved by an older version warns
    when its provenance stamp differs.

Examples
--------
>>> from sigmutsel import MutationDataset
>>> dataset = MutationDataset.load_dataset("hnsc/HNSC_dataset")
>>> dataset.n_samples
392
```

### `qc.load_repeat_intervals`

```python
load_repeat_intervals(rmsk_path)
```

```text
Load RepeatMasker regions into merged per-chromosome intervals.

Parameters
----------
rmsk_path : str or pathlib.Path
    Either UCSC's raw rmsk table dump (17 columns, no header) or
    a three-column BED of intervals (the distributed table of
    :mod:`repeatmasker`), as returned by
    :func:`setup.download_repeatmasker_bed`; plain or compressed
    (pandas infers the compression from the extension). Both give
    the same intervals.

Returns
-------
dict[str, tuple[numpy.ndarray, numpy.ndarray]]
    Chromosome -> (merged_starts, merged_ends), 0-based
    half-open (UCSC convention), sorted and overlap-merged --
    the format :func:`flag_repetitive_regions` expects. Reuses
    :func:`wes_target._merge_intervals`, the same interval-merge
    already used for the WES-target gene universe.

Examples
--------
>>> from sigmutsel.qc import load_repeat_intervals
>>> from sigmutsel.setup import download_repeatmasker_bed
>>> intervals = load_repeat_intervals(download_repeatmasker_bed("hg38"))
>>> starts, ends = intervals["chr1"]
```

### `setup.download_repeatmasker_bed`

```python
download_repeatmasker_bed(
    genome_build: str = 'hg38',
    force: bool = False,
    full: bool = False,
)
```

```text
The RepeatMasker regions the call QC and the territory use.

Parameters
----------
genome_build : {"hg38", "hg19"}, default "hg38"
    hg38 matches this package's usual GRCh38 MAF coordinates
    (see the module docstring of :mod:`wes_target` for why the
    rest of the package is hg38-based); hg19 is provided for
    callers working against hg19 coordinates directly.
force : bool, default False
    If True, download even if the file exists.
full : bool, default False
    hg38 only. If True, UCSC's full rmsk table (149 MB, every
    column); otherwise the full table if it is already present,
    else the distributed table of merged intervals (17 MB; see
    :mod:`repeatmasker`), which gives the same intervals. hg19
    always takes the full table.

Returns
-------
Path
    Either UCSC's raw tab-separated ``rmsk.<build>.txt.gz`` dump
    or the distributed BED (``repeatmasker.DISTRIBUTED_NAME``).
    :func:`qc.load_repeat_intervals` reads both.

Notes
-----
Kept compressed: pandas reads ``.gz`` and ``.xz`` files directly,
so there's no need to decompress them to disk.

Examples
--------
>>> from sigmutsel.qc import load_repeat_intervals
>>> from sigmutsel.setup import download_repeatmasker_bed
>>> intervals = load_repeat_intervals(download_repeatmasker_bed("hg38"))
>>> dataset.generate_mutation_db(
...     qc_mode=True, qc_kwargs={"repeat_intervals": intervals})
```

## Calibration steps

### `setup.download_tcga_absolute_purity`

```python
download_tcga_absolute_purity(force: bool = False)
```

```text
Download TCGA's Pan-Cancer Atlas ABSOLUTE purity/ploidy calls.

A single pan-cancer file covering every TCGA cohort and sample --
not cohort-specific, so this only needs downloading once, ever,
not once per cohort or per rerun.

Parameters
----------
force : bool, default False
    If True, download even if the file exists.

Returns
-------
Path
    Path to ``TCGA_mastercalls.abs_tables_JSedit.fixed.txt``. One
    row per sample (15-character barcode, column ``array``), with
    columns including ``purity``, ``ploidy``, and
    ``Subclonal genome fraction``.

Notes
-----
From the GDC PanCanAtlas publication resources (Liu et al. 2018,
Hoadley et al. 2018) -- see
https://gdc.cancer.gov/about-data/publications/PanCan-CellOfOrigin.
Companion file: :func:`download_tcga_absolute_segments`.

Examples
--------
>>> import pandas as pd
>>> from sigmutsel.setup import download_tcga_absolute_purity
>>> purity = pd.read_csv(download_tcga_absolute_purity(), sep="     ")
>>> purity[["array", "purity", "ploidy"]].head(2)
```

### `setup.download_tcga_absolute_segments`

```python
download_tcga_absolute_segments(force: bool = False)
```

```text
Download TCGA's Pan-Cancer Atlas ABSOLUTE copy-number segments.

A single pan-cancer file (~250MB, ~1.9M rows) covering every TCGA
cohort and sample -- not cohort-specific, so this only needs
downloading once, ever, not once per cohort or per rerun.

Parameters
----------
force : bool, default False
    If True, download even if the file exists.

Returns
-------
Path
    Path to ``TCGA_mastercalls.abs_segtabs.fixed.txt``. One row
    per (sample, segment), with ``Sample``, ``Chromosome``,
    ``Start``, ``End``, ``Modal_Total_CN``, and
    ``Cancer_cell_frac_a1``/``_a2`` columns (the latter describe
    the copy-number event's own clonality, not an individual
    point mutation's, which the purity/CN-corrected VAF QC of
    :mod:`sigmutsel.sample_qc` relies on).

Notes
-----
Companion file: :func:`download_tcga_absolute_purity`. Same
source as that function's docstring.

Examples
--------
>>> from sigmutsel.sample_qc import load_copy_number_segments_from_file
>>> from sigmutsel.setup import download_tcga_absolute_segments
>>> segments = load_copy_number_segments_from_file(
...     download_tcga_absolute_segments())
```

### `sample_qc.load_copy_number_segments_from_file`

```python
load_copy_number_segments_from_file(
    segments_path,
    cache_path=None,
    **kwargs,
)
```

```text
Load+process an ABSOLUTE-style segments file, with optional
on-disk caching of the processed structure.

:func:`load_copy_number_segments` itself does no I/O (it takes an
already-loaded DataFrame), by design -- but building its output
from TCGA's ~250MB, ~1.9M-row pan-cancer segments file means
parsing that file *and* a ~240k-group Python-level groupby, real
costs that don't depend on which cohort is being processed. A
per-cohort pipeline invoked as a separate process per cohort (as
this package's typical caller does) would otherwise pay both
costs again for every cohort, even though the underlying segments
never change (TCGA is a closed, fixed dataset -- see the L_low
low-burden-correction plan's data-provenance notes). This wrapper
does the expensive parse+groupby once and pickles the result, so
later calls skip straight to a fast unpickle instead of reading
the raw file at all.

Parameters
----------
segments_path : str or pathlib.Path
    Path to the raw segments file (e.g. from
    :func:`setup.download_tcga_absolute_segments`).
cache_path : str or pathlib.Path or None, default None
    Where to cache the processed structure. If it already exists,
    it's unpickled and returned directly -- `segments_path` is
    never read in that case. If it doesn't exist, the file at
    `segments_path` is read and processed as usual, then pickled
    here for next time. If None, defaults to `segments_path` with
    a ``.pkl`` suffix appended (not replaced, since ABSOLUTE's
    segments filename already ends in ``.txt``).
**kwargs
    Forwarded to :func:`load_copy_number_segments` (its column-name
    parameters) on a cache miss.

Returns
-------
dict
    As returned by :func:`load_copy_number_segments`.

Examples
--------
>>> from sigmutsel.sample_qc import load_copy_number_segments_from_file
>>> from sigmutsel.setup import download_tcga_absolute_segments
>>> segments = load_copy_number_segments_from_file(
...     download_tcga_absolute_segments())  # parsed once, then cached
```

### `sample_qc.flag_low_purity_samples`

```python
flag_low_purity_samples(
    purity_table,
    barcode_column='array',
    purity_column='purity',
    threshold=0.3,
)
```

```text
Flag samples with tumor purity below `threshold`.

Low purity means the somatic-variant caller had less tumor signal
to work with relative to normal-cell contamination, which can
cause real mutations to go undetected -- independent evidence for
"this sample was probably under-sequenced" that doesn't rely on
its own mutation count (see the L_low low-burden-correction
rework this function was built for).

Parameters
----------
purity_table : pandas.DataFrame
    A purity table with a sample-barcode column and a purity
    column, e.g. TCGA's Pan-Cancer Atlas ABSOLUTE purity/ploidy
    file (``TCGA_mastercalls.abs_tables_JSedit.fixed.txt``),
    loaded by the caller -- this function does no I/O itself.
barcode_column : str, default "array"
    Column holding the sample barcode to index by. Must be at
    the same barcode granularity as
    :attr:`MutationDataset.mutation_db`'s
    ``Tumor_Sample_Barcode`` (or a prefix of it -- see
    `barcode_length`).
purity_column : str, default "purity"
    Column holding the purity estimate, 0-1 scale.
threshold : float, default 0.30
    Samples with purity strictly below this are flagged. Default
    based on Yu et al. 2023 (Clin. Transl. Med. 13:e1252,
    doi:10.1002/ctm2.1252; colorectal cancer, MuTect2 calls on
    TCGA and MSK data): false-negative mutation calls rose
    sharply below 30% purity. MuTect2 is one of the callers
    merged into GDC's "Aliquot Ensemble Somatic Variant Merging
    and Masking" MAFs, the usual TCGA input.

Returns
-------
pandas.Series
    Boolean, indexed by `barcode_column`'s values. ``True`` means
    flagged. Samples with missing/NaN purity are *not* flagged
    (no evidence either way, not evidence of a problem).

Examples
--------
The purity table's barcodes are shorter than a MAF's, so map each
tumor to its prefix first:

>>> import pandas as pd
>>> from sigmutsel.sample_qc import flag_low_purity_samples
>>> from sigmutsel.setup import download_tcga_absolute_purity
>>> purity = pd.read_csv(download_tcga_absolute_purity(), sep="     ")
>>> tumors = dataset.mutation_db["Tumor_Sample_Barcode"].unique()
>>> short = pd.Series(tumors, index=tumors).str.slice(
...     0, len(purity["array"].iloc[0]))
>>> low_purity = (short.map(flag_low_purity_samples(purity))
...               .fillna(False).astype(bool))
```

### `sample_qc.flag_vaf_shape_samples`

```python
flag_vaf_shape_samples(
    mutation_db,
    purity_table,
    copy_number_segments,
    barcode_column='array',
    purity_column='purity',
    threshold=0.7,
    depth_col='t_depth',
    alt_col='t_alt_count',
    min_depth=20,
    min_variants=5,
    lift_to_hg19=True,
    chain_blocks=None,
)
```

```text
Flag samples whose VAF distribution doesn't match their purity.

Batch driver: joins local copy number (see
:func:`annotate_local_copy_number`), then applies
:func:`compute_vaf_shape_score` per sample. Requires `depth_col`/
`alt_col` to have survived compaction (see
:func:`load_maf_files.compact_data`).

Parameters
----------
mutation_db : pandas.DataFrame
    A dataset's compact mutation database, with
    ``Tumor_Sample_Barcode``, ``Chromosome``, ``Start_Position``,
    `depth_col`, `alt_col` columns.
purity_table : pandas.DataFrame
    Same purity table as :func:`flag_low_purity_samples`.
copy_number_segments : dict
    As returned by :func:`load_copy_number_segments`.
barcode_column, purity_column : str
    As in :func:`flag_low_purity_samples`.
threshold : float, default 0.7
    Samples with an estimated 75th-percentile CCF below this are
    flagged.
depth_col, alt_col, min_depth, min_variants
    Passed to :func:`compute_vaf_shape_score`: the read-depth and
    alternate-read columns, the least depth of a variant used,
    and the least number of such variants a sample needs to be
    scored.
lift_to_hg19 : bool, default True
    Lift the calls' GRCh38 positions to hg19 before looking up their
    segments, as TCGA's ABSOLUTE segments are hg19 (the
    homozygous-deletion segments of PTEN and RB1 sit at their hg19
    positions). False compares positions as given.
chain_blocks : pandas.DataFrame, optional
    The hg38-to-hg19 chain already read; defaults to the package's.

Returns
-------
pandas.Series
    Boolean, indexed by ``Tumor_Sample_Barcode``. ``True`` means
    flagged. Samples with no purity estimate or too few
    depth-filtered, copy-number-annotated variants are not
    flagged (no evidence either way).

Examples
--------
>>> from sigmutsel.sample_qc import flag_vaf_shape_samples
>>> tumors = dataset.mutation_db["Tumor_Sample_Barcode"].unique()
>>> vaf_shape = flag_vaf_shape_samples(
...     dataset.mutation_db, purity, segments
... ).reindex(tumors, fill_value=False)
```

### `sample_qc.combine_sample_flags`

```python
combine_sample_flags(*flags, how='any', warn_threshold=0.05)
```

```text
Combine per-sample boolean flag Series into one.

Parameters
----------
*flags : pandas.Series
    Boolean Series to combine, e.g. the outputs of
    :func:`flag_low_purity_samples` and
    :func:`flag_vaf_shape_samples`. Indices need not match --
    combined on their union, treating a sample absent from one
    flag as not-flagged by that check (no evidence, not a
    positive flag). For `warn_threshold` to reflect a real
    fraction of the cohort, pass in flags already restricted/
    reindexed to that cohort's own samples -- an unrestricted
    pan-cancer `flag_low_purity_samples` result combined with a
    single cohort's `flag_vaf_shape_samples` result would dilute
    the denominator with irrelevant samples from other cohorts.
how : {"any", "all"}, default "any"
    "any": flagged if any input flags it. "all": flagged only if
    every input flags it.
warn_threshold : float or None, default 0.05
    Log a warning if more than this fraction of the combined
    index ends up flagged -- a signal that dropping flagged
    samples outright may be discarding too much of the cohort,
    and inverse-variance weighting (see the L_low low-burden-
    correction plan's deferred Option A) may be worth revisiting
    instead. Set to None to disable.

Returns
-------
pandas.Series
    Boolean, indexed by the union of all input indices.

Examples
--------
>>> from sigmutsel.sample_qc import combine_sample_flags
>>> dataset.sample_qc_flags = combine_sample_flags(low_purity, vaf_shape)
>>> excluded = set(dataset.sample_qc_flags[dataset.sample_qc_flags].index)
>>> model.estimate_gamma("TP53", excluded_samples=excluded)
```

### `models.MutationDataset.set_exclude_repeats`

```python
set_exclude_repeats(exclude=True)
```

```text
Move the territory off (or back onto) RepeatMasker regions.

Like :meth:`set_germline_mask`, QC on both sides: the call QC
drops every call in a repeat, so with ``exclude`` the
territory leaves those sites out too
(:func:`channel_universe.territory_masks`). Every call is
placed again (a call in a repeat, rare after the QC, leaves
the universe as ``outside_territory``), the opportunity tables
are rebuilt with the same mask and site weights, and every
table built from the calls is recomputed. Saved with the
dataset; the first call on a dataset takes minutes (it
rebuilds the opportunity).

Parameters
----------
exclude : bool, default True
    Take the repeats out (True) or put them back (False).

Returns
-------
pandas.Series
    Calls per ``universe_reason`` after the change.

Examples
--------
>>> if not (dataset.channel_universe or {}).get("exclude_repeats"):
...     dataset.set_exclude_repeats(True)
```

### `models.MutationDataset.set_exclude_overlaps`

```python
set_exclude_overlaps(exclude=True)
```

```text
Give each coding position and splice site to one gene.

Overlapping genes (readthroughs, novel clone-based genes over a
named one) each counted a shared position in their opportunity,
while the MAF credited its calls to only one of them. With
``exclude``, a position belongs to its owner
(:func:`channel_universe.position_owners`): it leaves the
other genes' territory, and calls labelled with a losing gene
move to the owner (:func:`channel_universe.credit_calls_to_owners`).
Every call is placed again and the tables rebuilt, as in
:meth:`set_exclude_repeats`.

Parameters
----------
exclude : bool, default True
    One gene per position (True), or every overlapping gene
    counting it (False).

Returns
-------
(pandas.Series, int)
    Calls per ``universe_reason`` after the change, and how
    many calls moved to their position's owner.

Examples
--------
>>> reasons, moved = dataset.set_exclude_overlaps(True)
```

### `spectrum_shrinkage.heldout_spectrum_scores`

```python
heldout_spectrum_scores(
    db,
    sig_matrix,
    kappa_1_grid=(10.0, 30.0, 100.0, 300.0, 1000.0),
    kappa_2_grid=(10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 5000.0),
    floor_signature='SBS5',
    floor_share=0.1,
    folds=5,
    seed=0,
    fit=None,
    workdir=None,
)
```

```text
Each tumor's held-out log-probability under each pair of kappas.

Each tumor's in-universe calls are split at random into ``folds``
parts. For each part, signatures are fitted on the other parts
(``fit(matrix, signatures, workdir)``, by default
SigProfilerAssignment against ``sig_matrix``, every call as in
production) and the part's passenger calls are scored under the
shrunk spectrum, with the pool and the hybrid's own counts taken
from the other parts' passenger calls. Arms: every ``(kappa_1,
kappa_2)`` of the grids plus the floor alone (``kappa_2`` None)
at every ``kappa_1``, and ``"none"``, the signature spectrum as
is. It refits the signatures ``folds`` times: about a minute for a
cohort of a few hundred exomes.

Parameters
----------
db : pandas.DataFrame
    The dataset's ``mutation_db``.
sig_matrix : pandas.DataFrame
    The cohort's signature matrix, indexed by the 96 types
    (``dataset.signature_matrix.set_index("MutationType")``).
kappa_1_grid, kappa_2_grid : sequence of float
    The grids of the two shrinkage strengths.
floor_signature : str, default "SBS5"
    The signature mixed into the cohort pool.
floor_share : float, default 0.1
    Its share of the pool.
folds : int, default 5
    Parts each tumor's calls are split into.
seed : int, default 0
    The split's seed.
fit : callable, optional
    ``fit(matrix, signatures, workdir)`` returning exposures;
    default SigProfilerAssignment.
workdir : str or Path, optional
    Scratch directory for the fits; default a temporary one.

Returns
-------
pandas.DataFrame
    One row per tumor: ``burden`` (its in-universe calls), the
    log-probability under ``"none"`` over the held-out calls it can
    score, ``none_impossible`` (held-out calls it gives
    probability 0), and one column per arm (:func:`arm_label`).
    Every shrunk arm scores every call, since the prior has no
    zero type; compare an arm with ``"none"`` only on tumors with
    ``none_impossible == 0``, or on the arm-to-arm differences.

Examples
--------
>>> from sigmutsel.spectrum_shrinkage import (
...     heldout_spectrum_scores, select_kappas)
>>> signatures = dataset.signature_matrix  # 96 types x signatures
>>> scores = heldout_spectrum_scores(dataset.mutation_db, signatures)
>>> kept = scores.drop(index=scores.index.intersection(excluded))
>>> kept.attrs.update(scores.attrs)
>>> dataset.set_spectrum_shrinkage(select_kappas(kept))
```

### `spectrum_shrinkage.select_kappas`

```python
select_kappas(
    scores,
    family='hybrid',
    outer_folds=5,
    seed=0,
    floor_signature='SBS5',
    floor_share=0.1,
)
```

```text
Choose the kappas, and measure the choice's optimism.

The chosen pair maximizes the summed held-out log-probability over
every tumor. The nested estimate splits the tumors into
``outer_folds`` groups, chooses on all groups but one and scores
the one left out; ``optimism`` is the in-sample gain minus the
nested gain. Gains are against the floor at the smallest
``kappa_1`` of the grid (an arm that scores every call), and
against ``"none"`` on the tumors it can score.

Parameters
----------
scores : pandas.DataFrame
    :func:`heldout_spectrum_scores`'s table (QC-flagged tumors
    dropped, its ``attrs`` kept).
family : {"hybrid", "floor"}, default "hybrid"
    Both kappas, or ``kappa_1`` alone.
outer_folds : int, default 5
    Groups of tumors of the nested estimate.
seed : int, default 0
    The groups' seed.
floor_signature, floor_share
    As in :func:`heldout_spectrum_scores`, recorded with the
    choice.

Returns
-------
SpectrumShrinkage
    With ``selection`` recording the grid, the nested and
    in-sample gains, the optimism and each outer fold's choice.

Examples
--------
>>> chosen = select_kappas(kept)
>>> chosen.label()
'kappa_1 100, kappa_2 1000'
>>> round(chosen.selection["optimism"])  # nats, HNSC
71
```

### `models.MutationDataset.set_spectrum_shrinkage`

```python
set_spectrum_shrinkage(shrinkage)
```

```text
Set (or clear, with None) the per-cohort spectrum shrinkage.

Like :meth:`set_site_weights`, a per-cohort parameter chosen
once and saved with the dataset; no table changes here, and
models built after it take it into their baselines.

Parameters
----------
shrinkage : spectrum_shrinkage.SpectrumShrinkage, dict or None
    The chosen kappas (:func:`spectrum_shrinkage.select_kappas`)
    or their dict; None clears them.

Returns
-------
None

Examples
--------
>>> from sigmutsel.spectrum_shrinkage import (
...     heldout_spectrum_scores, select_kappas)
>>> signatures = dataset.signature_matrix  # 96 types x signatures
>>> scores = heldout_spectrum_scores(dataset.mutation_db, signatures)
>>> dataset.set_spectrum_shrinkage(select_kappas(scores))
>>> dataset.spectrum_shrinkage.label()
'kappa_1 100, kappa_2 1000'
```

### `models.Model.estimate_site_weights`

```python
estimate_site_weights(
    pseudo_count=50.0,
    excluded_samples=None,
    cpg_term=None,
)
```

```text
Fit per-cohort site weights and put them on the opportunity.

Distance-to-junction and strand factors (:mod:`site_weights`)
are fitted on the passenger genes' in-universe coding calls of
both channels, against the expectation of the per-tumor type
rates alone (``sum_j mu_{j,tau} / D_tau`` per substitution,
observability included), so the covariates -- which act per
gene -- play no part. Then :meth:`MutationDataset.set_site_weights`
rebuilds the opportunity. Call it after :meth:`compute_mu_taus`
and **before** :meth:`compute_base_mus`: baselines built earlier
are on the old opportunity. ``cpg_term`` adds the CpG C>T term of
:class:`site_weights.SiteWeights`: ``"flat"`` fits it freely, a
``(mean, cov)`` pair with that Gaussian prior (per-cohort values
shrunk by the between-cohort spread); None (default) has no term.

Parameters
----------
pseudo_count : float, default 50.0
    Expected calls added to every margin of the fit, which
    pulls each weight toward 1 where calls are few.
excluded_samples : collection of str, optional
    Tumors (QC-flagged) left out of the fit.
cpg_term : None, "flat" or (mean, cov)
    The CpG C>T term, as above.

Returns
-------
site_weights.SiteWeights
    The fitted weights, also set on the dataset (saved with
    it); ``n_calls`` is how many calls they were fitted on.

Examples
--------
>>> from sigmutsel import Model
>>> probe = Model(dataset, None, cov_effects_per_sigma=False,
...               prob_g_tau_tau_independent=False)
>>> weights = probe.estimate_site_weights(excluded_samples=excluded)
>>> dataset.save_dataset("hnsc/HNSC_dataset", overwrite=True)
```

### `sample_qc.homozygously_deleted_genes`

```python
homozygously_deleted_genes(
    copy_number_segments,
    models,
    tumors,
    barcode_length=15,
    chain_path=None,
    chain_blocks=None,
)
```

```text
Which genes each tumor has lost both copies of.

A gene is deleted in a tumor when a segment with total copy number
0 overlaps its coding span. Such a tumor cannot carry a point
mutation where the gene is gone, and a tumor suppressor deleted
this way has met its selective pressure, so gamma should not count
the tumor as evidence against selection (see
``Model.estimate_gamma(hold_out_homdel=...)``).

The segments are TCGA's pan-cancer ABSOLUTE calls, in **hg19**
coordinates (checked: homozygous-deletion segments pile up on PTEN
and RB1 at their hg19 positions, not their GRCh38 ones), while the
transcripts are GRCh38. Each gene's coding span -- first and last
coding base of its chosen transcript -- is lifted to hg19 first; a
gene whose ends do not both lift to one chromosome is skipped.

Parameters
----------
copy_number_segments : dict
    As returned by :func:`load_copy_number_segments`.
models : channel_universe.TranscriptModels
    The chosen transcripts.
tumors : iterable of str
    Full tumor barcodes to look up; each is matched to the
    segments by its first `barcode_length` characters. A tumor
    without segments gets no deletions (unknown, not absent).
barcode_length : int, default 15
chain_path : path-like, optional
    hg38-to-hg19 chain; defaults to the package's.
chain_blocks : pandas.DataFrame, optional
    The chain already read (:func:`liftover.read_chain`), in place
    of `chain_path`.

Returns
-------
pandas.DataFrame
    One row per (tumor, deleted gene): ``Tumor_Sample_Barcode``,
    ``ensembl_gene_id``.

Examples
--------
Give the dataset the deletions once; gamma then holds those tumors
out of each gene's absent set:

>>> from sigmutsel.channel_universe import (
...     load_or_build_transcript_models)
>>> from sigmutsel.sample_qc import homozygously_deleted_genes
>>> tumors = dataset.mutation_db["Tumor_Sample_Barcode"].unique()
>>> dataset.homdel_db = homozygously_deleted_genes(
...     segments, load_or_build_transcript_models(), tumors)
```

### `channel_universe.load_or_build_transcript_models`

```python
load_or_build_transcript_models(
    gtf_path=None,
    fasta_paths=None,
    cache_dir=None,
    force=False,
)
```

```text
Transcript models for every coding gene, cached on disk.

The cache lives in ``cache_dir`` (default
:data:`locations.location_channel_universe_dir`) under a
sub-directory named by a fingerprint of the GTF and FASTA and of
this module's build version, so a change to any of them rebuilds
rather than serving a stale table. The first build takes a few
minutes; later calls load the cache.

Parameters
----------
gtf_path : str or Path, optional
    Default the GENCODE v38 GTF of the reference data.
fasta_paths : str, Path or list, optional
    Default the Ensembl CDS FASTA of the reference data.
cache_dir : str or Path, optional
    Default :data:`locations.location_channel_universe_dir`.
force : bool, default False
    Rebuild even if a cache exists.

Returns
-------
TranscriptModels
    One transcript per gene (MANE Select, else canonical, else
    longest), with its coding positions and splice sites.

Examples
--------
>>> from sigmutsel.channel_universe import (
...     load_or_build_transcript_models)
>>> models = load_or_build_transcript_models()
```

### `models.MutationDataset.build_indel_db`

```python
build_indel_db(location_gene_set=None)
```

```text
Read the non-silent coding indels of the dataset's MAFs.

The SBS build keeps single-base calls only, so a tumor whose
hit in a gene is a frameshift or an in-frame indel looks
unmutated there. This reads the same MAFs for those indels
(:func:`.load_maf_files.read_coding_indels`), keeps the
tumors the dataset has, and stores them in :attr:`indel_db`;
nothing else changes, so fitted models stay valid.

Parameters
----------
location_gene_set : str, Path, None or "default", default None
    HGNC table for the gene identifiers. None (the default)
    keeps the MAF's, as :meth:`generate_mutation_db` does by
    default: the hold-out matches indels to genes by these
    identifiers, so they must be the ones :attr:`mutation_db`
    carries. ``"default"`` uses the package's HGNC table, for a
    dataset whose mutation table was built with it.

Returns
-------
pandas.Series
    Calls per ``Variant_Classification``.

Examples
--------
>>> dataset.build_indel_db()
>>> dataset.save_dataset("hnsc/HNSC_dataset", overwrite=True)
```

## The rate model

### `models.Model`

```python
Model(
    dataset: sigmutsel.models.MutationDataset | str | pathlib.Path,
    cov_matrix: pandas.DataFrame | None = None,
    cov_effects_kwargs: dict | None = None,
    L_low: float | None = None,
    L_high: float | None = None,
    cut_at_L_low: bool | None = None,
    cov_effects_per_sigma: bool | None = None,
    prob_g_tau_tau_independent: bool | None = None,
    signature_selection: list | tuple | None = None,
    include_other: bool = False,
)
```

```text
Signature based, mutation and selection model.

Each Model instance represents one specific analysis with a fixed
set of covariates. To test different covariate combinations,
create multiple Model instances.

Parameters
----------
dataset : MutationDataset, str or Path
    The dataset, or a directory saved by
    :meth:`MutationDataset.save_dataset`.
cov_matrix : pandas.DataFrame, optional
    Gene covariates (genes x columns), assigned with
    :meth:`assign_cov_matrix`'s defaults. To reduce them by PCA
    or to pass published components, leave it None and call
    :meth:`assign_cov_matrix` yourself.
cov_effects_kwargs : dict, optional
    Default keyword arguments of :meth:`estimate_cov_effects`
    (the older single-channel model).
L_low, L_high, cut_at_L_low : optional
    A burden correction for low-burden tumors, superseded by
    sample QC (flag such tumors and pass them as
    ``excluded_samples``); leave them None.
cov_effects_per_sigma : bool, optional
    Separate covariate effects per signature (older models).
    Setting it (False or True) computes the per-tumor type rates
    at construction.
prob_g_tau_tau_independent : bool, optional
    Whether the share of a tumor's mutations of type ``tau`` that
    falls in gene ``g`` ignores the tumor's spectrum. **Pass
    False** for the recommended channel model; setting it
    computes the baselines at construction.
signature_selection : list, optional
    Signatures to keep separate (:meth:`aggregate_signatures`).
include_other : bool, default False
    With ``signature_selection``, pool the other signatures into
    one ``"other"`` group instead of dropping them.

Attributes
----------
dataset : MutationDataset
    Reference to the shared mutation dataset.
cov_matrix : pd.DataFrame or None
    Covariate matrix (genes × covariates) for THIS model.
    Should contain only the covariates this model uses.
    If None, this is a baseline model with no covariates.
    If provided during initialization, it will be automatically
    reindexed to match dataset.contexts_by_gene.index via
    assign_cov_matrix().
cov_effects : np.ndarray or None
    Estimated covariate effect coefficients from MAP estimation.
    Lazy-loaded.
cov_effects_posteriors : object or None
    Posterior samples from MCMC (arviz.InferenceData).
    Lazy-loaded.
mu_taus : pd.DataFrame or dict or None
    Baseline mutation rates per type per tumor. Can be:
    - pd.DataFrame for signature-independent models
    - dict of DataFrames for signature-separated models (when
      separate_per_sigma=True)
    Lazy-loaded via compute_mu_taus().
base_mus : dict or pd.DataFrame or None
    Baseline mutation rates per gene per tumor. Can be:
    - pd.DataFrame for signature-independent models
    - dict of DataFrames for signature-separated models
    Lazy-loaded via compute_base_mus().
mu_gs : pd.DataFrame or None
    Mutation rates per gene per sample, incorporating covariate
    effects. Always a DataFrame with genes as index and samples
    as columns, regardless of whether base_mus is
    signature-separated.
    Lazy-loaded via compute_mu_gs().
mu_ms : pd.DataFrame or None
    Mutation rates per variant per sample. Lazy-loaded.
Auto-initialization parameters
    Optional keyword arguments ``L_low``, ``L_high``,
    ``cut_at_L_low``,
    ``cov_effects_per_sigma``,
    ``prob_g_tau_tau_independent``, ``signature_selection``, and
    ``include_other`` can be provided at initialization to
    automatically run the corresponding setup steps (mutation
    burdens, baseline rates, and optional signature
    aggregation). ``include_other`` (default False) is forwarded
    to :meth:`aggregate_signatures`: set it to True to bucket
    every signature not named in ``signature_selection`` into a
    single ``"other"`` group instead of dropping it from
    coefficient fitting entirely.
prob_g_tau_tau_independent : bool or None
    Flag indicating whether base_mus were computed using
    type-independent gene probabilities. Set by
    compute_base_mus().
cov_effects_kwargs : dict
    Keyword arguments for MAP estimation
    (estimate_cov_effects).
passenger_genes_r2 : float or None
    R² value comparing model predictions with observed data of
    passenger genes. Lazy-loaded.
saved_location : str or None
    Filesystem path where the model snapshot was last saved or
    loaded from. None until save_model() or load_model() runs.

Examples
--------
The recommended (consequence-split channel) model, on published
covariate components (``tutorials/selection_in_hnsc.md``):

>>> import pandas as pd
>>> import sigmutselcovs
>>> pcs = sigmutselcovs.fetch_covariate_matrix("HNSC")
>>> files = sigmutselcovs.fetch_covariate_artifacts(
...     "HNSC", artifacts=("pca_indicators",))
>>> design = pd.concat(
...     [pcs.iloc[:, :300], pd.read_parquet(files["pca_indicators"])],
...     axis=1)
>>> model = Model(dataset, None, cov_effects_per_sigma=False,
...               prob_g_tau_tau_independent=False)
>>> model.assign_cov_matrix(design)
>>> model.compute_channel_base_mus()
>>> model.estimate_channel_rg_cov_effects(
...     excluded_samples=excluded, include_drivers=True,
...     sample="full")
>>> model.save_model("hnsc/model")

The older single-channel models:

>>> # Create a baseline model (no covariates)
>>> model_no_cov = Model(dataset, None)
>>> model_no_cov.has_covariates()
False
>>> model_no_cov.n_covariates
0

>>> # Create a simple model with MRT covariate
>>> model = Model(
...     dataset,
...     cov_matrix_full[['mrt']])
>>> model.has_covariates()
True
>>> model.n_covariates
1

>>> # Create a model with multiple covariates
>>> model = Model(
...     dataset,
...     cov_matrix_full[['mrt', 'log1p_gexp', 'log1p_atac']])
>>> model.n_covariates
3
>>> # Note: cov_matrix is automatically reindexed to match
>>> # dataset.contexts_by_gene.index during initialization

>>> # Compute baseline mutation rates
>>> model.compute_mu_taus()
>>> model.compute_base_mus()
>>> model.has_sig_dependent_mus()
False
```

### `models.Model.assign_cov_matrix`

```python
assign_cov_matrix(
    cov_matrix,
    run_pca=False,
    pca_kwargs=None,
    dr_method=None,
    dr_kwargs=None,
    missing_blocks=None,
    incomplete='project',
)
```

```text
Assign covariate matrix, restricting to dataset genes.

This method assigns a covariate matrix to the model after
reindexing it to match the genes in contexts_by_gene.
Optionally applies dimensionality reduction to the covariates.

Note: This method is automatically called during Model
initialization if a cov_matrix is provided to the constructor
(e.g., Model(dataset, cov_matrix)).

This ensures that:
1. Only genes with context information are included
2. Gene order matches dataset.contexts_by_gene.index
3. Missing genes are handled appropriately
4. (Optional) Covariates are transformed via PCA or
   Riemannian STATS

Parameters
----------
cov_matrix : pd.DataFrame
    Covariate matrix with genes as index and covariates as
    columns. Index should be Ensembl gene IDs.
run_pca : bool, default False
    Deprecated shorthand for ``dr_method='pca'``. If True
    and ``dr_method`` is None, PCA is used.
pca_kwargs : dict or None, default None
    Deprecated. Use ``dr_kwargs`` instead. Forwarded to
    :func:`utils.run_pca_on_covariates` when ``run_pca=True``
    and ``dr_kwargs`` is None.
dr_method : str or None, default None
    Dimensionality reduction method. Options:

    - ``'pca'``: sklearn PCA (see
      :func:`utils.run_pca_on_covariates`)
    - ``'riemannian_stats'``: Riemannian STATS (see
      :func:`utils.run_riemannian_stats_on_covariates`)
    - ``None``: no reduction, use covariates as-is

dr_kwargs : dict or None, default None
    Keyword arguments forwarded to the chosen DR function.
    Common options for both methods:
    - n_components : int, number of components to keep
    - columns : list[str], subset of columns to include
    - standardize : bool, default True
    - dropna : str, default 'any'
    Riemannian-only:
    - n_neighbors : int, default 15
    - min_dist : float, default 0.1
    - metric : str, default 'euclidean'
missing_blocks : Mapping[str, str] or pd.Series or None
    Column -> block name, for the covariate-fallback genes'
    block indicators (see :meth:`_fallback_eta`). ``None``
    makes every column one block. Columns absent from the
    mapping form a block ``"other"``. Only used with PCA.
incomplete : {"project", "impute"}, default "project"
    How genes with incomplete covariates enter (PCA only).
    ``"project"``: the PCA, and later ``c``, are fitted on the
    complete genes; each incomplete gene is projected onto that
    basis from the columns it has and shifted by one fitted
    factor per block it misses (:meth:`_fallback_eta`), so the
    complete genes are exactly what they would be without it.
    ``"impute"``: every gene's missing standardized values are
    set to 0 (the column mean), the PCA is fitted on every gene,
    and one 0/1 column per block ("this gene misses part of
    block B") is appended to the components, so ``c`` and the
    blocks' coefficients are fitted on every gene together and
    no gene falls back. Columns with no observed value or a
    single value are left out of the PCA; identical indicator
    columns are kept once.

Returns
-------
pd.DataFrame
    The reindexed (and optionally PCA-transformed) covariate
    matrix that was assigned to self.cov_matrix.

Raises
------
ValueError
    If contexts_by_gene has not been loaded in the dataset.
    Call dataset.generate_contexts_by_gene() or load_dataset() first.

Notes
-----
**Reindexing behavior:**

The input cov_matrix is reindexed using
`dataset.contexts_by_gene.index`. This means:
- Only genes with trinucleotide context information are kept
- Genes are ordered to match contexts_by_gene
- If a gene in contexts_by_gene is missing from cov_matrix,
  it will have NaN values for all covariates

**PCA transformation:**

When run_pca=True, the method performs:
1. Reindex to dataset genes (as above)
2. Run PCA using :func:`utils.run_pca_on_covariates`
3. Replace covariates with principal components (PC1, PC2, ...)

The PCA transformation is useful for:
- Reducing dimensionality when many correlated covariates exist
- Creating orthogonal features for modeling
- Avoiding multicollinearity issues

**Typical workflow:**

1. Generate mutation database and contexts:
    >>> dataset.generate_mutation_db("data/mutations.parquet")
    >>> dataset.generate_contexts_by_gene(...)

2. Create full covariate matrix for all genes:
    >>> cov_matrix_full = pd.DataFrame({
    ...     'mrt': mrt_per_gene,
    ...     'log1p_gexp': np.log1p(gexp_per_gene),
    ...     'log1p_atac': np.log1p(atac_per_gene)})

3. Create model and assign restricted covariate matrix:
    >>> model = Model(dataset)
    >>> model.assign_cov_matrix(cov_matrix_full[['mrt', 'log1p_gexp']])

This ensures the covariate matrix exactly matches the genes
with trinucleotide context information, which determines the
base_mus genes.

Examples
--------
>>> # Basic usage: assign covariates
>>> cov_matrix_full = pd.DataFrame({
...     'mrt': mrt_per_gene,
...     'log1p_gexp': np.log1p(gexp_per_gene)},
...     index=all_gene_ids)
>>>
>>> model = Model(dataset)
>>> model.assign_cov_matrix(cov_matrix_full)
>>>
>>> # Check that genes match
>>> assert (model.cov_matrix.index ==
...         dataset.contexts_by_gene.index).all()
>>>
>>> # Assign different subset of covariates
>>> model.assign_cov_matrix(cov_matrix_full[['mrt']])
>>>
>>> # Run PCA to reduce dimensionality
>>> model_pca = Model(dataset)
>>> model_pca.assign_cov_matrix(
...     cov_matrix_full,
...     run_pca=True,
...     pca_kwargs={'n_components': 3})
>>> # Now model_pca.cov_matrix has columns: PC1, PC2, PC3
>>>
>>> # PCA on subset of columns
>>> model_pca2 = Model(dataset)
>>> model_pca2.assign_cov_matrix(
...     cov_matrix_full,
...     run_pca=True,
...     pca_kwargs={
...         'columns': ['log1p_gexp', 'log1p_atac', 'log1p_h3k4me3'],
...         'n_components': 2,
...         'standardize': True})

See Also
--------
generate_contexts_by_gene : Must be called first on dataset
utils.run_pca_on_covariates : PCA implementation
```

### `models.Model.compute_channel_base_mus`

```python
compute_channel_base_mus(prob_g_tau_tau_independent=None)
```

```text
Split baseline rates into the two channels, per assay.

:meth:`_compute_channel_base_mus_single` for each assay's
tumors against its own territory (:meth:`_by_assay`); without
assays, exactly that method. Call it after
:meth:`assign_cov_matrix` and before
:meth:`estimate_channel_rg_cov_effects`.

Parameters
----------
prob_g_tau_tau_independent : bool, optional
    Default: the model's own setting (pass False to the
    constructor).

Returns
-------
(pandas.DataFrame, pandas.DataFrame)
    The synonymous and non-synonymous baselines, genes x
    tumors, also kept as :attr:`base_mus_syn` and
    :attr:`base_mus_nonsyn`.

Examples
--------
>>> model = Model(dataset, None, cov_effects_per_sigma=False,
...               prob_g_tau_tau_independent=False)
>>> model.assign_cov_matrix(design)
>>> syn, nonsyn = model.compute_channel_base_mus()
```

### `models.Model.estimate_channel_rg_cov_effects`

```python
estimate_channel_rg_cov_effects(
    sample='MAP',
    chains=4,
    burn=1000,
    tol=0.05,
    excluded_samples=None,
    include_drivers=True,
    separate_c='intercept',
    train_genes=None,
    fit_rg=True,
    fit_intercept=True,
    use_silent_channel=True,
    variant_rates=True,
)
```

```text
Fit the full unified model: shared ``c``, ``θ`` and ``r_g``.

Stage 4 of the channel-split model. Adds a per-gene rate
correction ``r_g ~ Gamma(θ, 1/θ)``, shared across both
consequence channels, on top of the two-channel Poisson
likelihood. ``r_g`` is integrated out analytically, so the fit
is over ``c`` and ``θ`` only -- see
:mod:`sigmutsel.estimate_rg` for the closed form and why it
matters.

Afterwards, ``self.rg_theta`` holds the fitted θ and the two
``r_g`` variants are available through
:meth:`compute_r_g_production` and
:meth:`compute_r_g_for_evaluation`. Neither is applied to
``mu_gs`` automatically: which one a number was computed with
is exactly the thing that must never be ambiguous, so applying
``r_g`` is always an explicit act at the call site.

Parameters
----------
sample : {"MAP", "full"} | int, default "MAP"
    ``"MAP"`` for a point estimate, ``"full"`` for 4000
    posterior draws, or an integer for that many draws (at
    least ``chains``). **Unlike**
    :meth:`estimate_channel_cov_effects`, where an integer
    subsamples genes, an integer here is the number of draws
    -- this method has no gene-subsampling mode, and until
    2026-09-19 an integer was silently treated as ``"full"``.
chains, burn, tol, excluded_samples, include_drivers
    As in :meth:`estimate_channel_cov_effects`. The same
    ``excluded_samples`` must be used here and in any later
    ``r_g`` or R² call, since the per-gene statistics are sums
    over whichever samples were kept.
separate_c : bool | str, default "intercept"
    ``False`` (shared), ``"intercept"`` (shared slopes, own
    non-synonymous intercept) or ``True`` (own vector per
    channel); ``r_g`` and θ stay shared in all three. The
    intercept absorbs a calibration offset between channels,
    so testing ``True`` against ``False`` conflates that
    with the slopes -- go through ``"intercept"`` to
    separate them. It is the default because a shared
    intercept leaves the channels miscalibrated against each
    other by 13--18% in every cohort measured; see
    :mod:`sigmutsel.estimate_rg` for the numbers and for the
    caveat that correcting it is not a uniform win.
    The result lands in :attr:`channel_cov_effects` with shape
    ``(2, n_coeffs)`` -- row 0 synonymous, row 1
    non-synonymous -- and **not** in ``cov_effects``, because
    a single ``mu_gs`` is no longer defined by one ``c``. For
    the same reason the usual downstream recomputation
    (``mu_gs``/``mu_ms``/R²) is skipped in this mode; use
    :meth:`channel_rg_log_likelihood_at_fit` to compare the
    two models.
train_genes : collection of str or pd.Index or None, default None
    Forwarded to :meth:`_channel_gene_statistics` -- restricts
    which passenger genes' non-silent-channel data informs the
    fit (gene-level cross-validation's train fold). Pair with
    :meth:`estimate_passenger_genes_r2`'s ``genes`` on the
    held-out complement; see :func:`cross_validation.
    gene_cv_passenger_r2`.
fit_rg : bool, default True
    ``False`` drops ``r_g``/``θ`` and leaves the plain
    two-channel Poisson -- the same likelihood with the Gamma
    marginalisation removed, not a different one. ``rg_theta``
    and both ``r_g`` variants then raise, which is the point:
    an arm without ``r_g`` must not be scored with one (the
    nested model comparisons' arms without ``r_g``).
fit_intercept : bool, default True
    ``False`` pins the shared intercept at 0 rather than
    fitting it, which is the only way to reach ladder arm 0
    (``μ = μ̄``, nothing fitted) -- a zero-column
    ``cov_matrix`` still leaves the prepended intercept free
    and so gives arm 1. ``cov_effects`` keeps its full
    ``1 + n_covariates`` length either way, with a 0 in the
    first slot, so ``compute_mu_gs`` and everything else
    downstream need no special case.
use_silent_channel : bool, default True
    ``False`` fits the **non-synonymous channel alone**, over
    passenger genes only: the silent channel leaves the
    likelihood and the gene set shrinks from ``G`` to ``P``.
    With ``fit_rg=False`` this is a single-channel Poisson GLM
    with offset ``log μ̄^(nonsyn)_g`` -- ladder arms 1p/2ap,
    December's data under a Poisson rather than a Bernoulli.
    ``include_drivers`` has no effect in this mode (there is
    no silent channel for a driver to enter through) and
    ``separate_c`` must be ``False``. Because the data differ,
    the resulting log-likelihood is **not** comparable with a
    two-channel arm's; compare on a common held-out target.
variant_rates : bool, default True
    Whether to recompute the per-variant rates (``mu_ms``)
    after the fit. A gene-level use of the fit -- each fold of
    a held-out-gene cross-validation, say -- never reads them,
    and they are most of the time of a MAP fit (93% of a
    fold's fit in a small cohort). With ``False`` they are not
    computed and ``mu_ms`` is cleared, so no rates from an
    earlier fit survive next to this fit's coefficients; call
    :meth:`compute_mu_ms` later if they are needed.

Returns
-------
np.ndarray | arviz.InferenceData
    The shared coefficient vector (MAP) or its posterior.

Raises
------
ValueError
    If the channel baselines, the channel count matrices, or
    the covariate matrix are missing.
NotImplementedError
    If ``base_mus`` is signature-separated.

Examples
--------
>>> model.assign_cov_matrix(design)
>>> model.compute_channel_base_mus()
>>> model.estimate_channel_rg_cov_effects(
...     excluded_samples=excluded, include_drivers=True, sample="full")
>>> model.rg_theta
```

### `models.Model.compute_r_g_for_evaluation`

```python
compute_r_g_for_evaluation()
```

```text
Per-gene ``r_g`` from the **silent channel only**.

Wraps :func:`estimate_rg.r_g_silent_only_for_evaluation`,
which has no argument through which non-silent data could
reach it. Scale ``μ^(nonsyn)`` by this before scoring against
non-silent counts, and the score is honest by construction.

Covers the covariate-less genes too (see
:attr:`covariate_fallback_genes`), with the same ``theta``.

Returns
-------
pandas.Series
    ``r_g`` per gene (Ensembl id): the posterior mean of the
    gene's rate correction given its silent counts. Multiply
    :attr:`mu_gs` (which does not include it) by it.

Examples
--------
>>> r_g = model.compute_r_g_for_evaluation()
>>> rate = model.mu_gs.loc["ENSG00000141510"] * r_g["ENSG00000141510"]
```

### `models.Model.estimate_passenger_genes_r2`

```python
estimate_passenger_genes_r2(
    sample_weights=None,
    excluded_samples=None,
    target='any',
    gene_scaling=None,
    genes=None,
    return_per_gene=False,
)
```

```text
Estimate R² for passenger gene mutation frequency predictions.

This method evaluates model performance on passenger genes
by comparing predicted mutation frequency (number of samples
with mutations per gene) with observed frequency. Passenger
genes are those not in the Cancer Gene Census, which are
assumed to be under neutral selection.

The R² metric quantifies how well the model's predicted
mutation rates explain the observed mutation frequency across
passenger genes (gene-level evaluation).

If `mu_gs` have not been computed yet, this method
automatically calls `compute_mu_gs()` first. For models
without covariates, an info message is logged to indicate
that baseline mutation rates are being used.

The result is stored in `self.passenger_genes_r2`.

Parameters
----------
sample_weights : pd.Series or None, default None
    Optional per-sample weight (indexed by
    ``Tumor_Sample_Barcode``, matching `genes_present`'s
    columns), applied to both the observed and expected sums
    before computing R². Default (None) weights every sample
    equally (today's behavior). Intended for samples flagged
    by :mod:`sample_qc` as lower-confidence, rather than
    dropping them outright -- see `excluded_samples` for that.
    Samples missing from `sample_weights` get weight 1.
excluded_samples : collection of str or None, default None
    Sample barcodes to exclude entirely from both sums before
    computing R² (e.g. samples flagged by
    :func:`sample_qc.combine_sample_flags`, when dropping
    rather than downweighting). Applied before
    `sample_weights`.
target : {"any", "non_silent", "non_silent_counts"}, default "any"
    Which observation to score against.

    - ``"any"`` (default, today's behavior): "gene mutated at
      all", i.e. `dataset.genes_present`, predicted from the
      merged `mu_gs`.
    - ``"non_silent"``: only non-silent mutations, i.e.
      `dataset.genes_present_non_silent`, predicted from the
      **non-synonymous channel's** rates
      (:meth:`compute_channel_mu_gs`) rather than the merged
      ones -- predicting a non-silent target from a total rate
      would be biased high by construction.

    - ``"non_silent_counts"``: the same channel, but scored
      on mutation *counts* rather than presence --
      `dataset.genes_counts_non_silent` against
      ``Σ_j μ_g^(nonsyn,j)``, with **no** ``1 - exp(-μ)``
      step, since a count target has no censoring. This is
      the headline for a Poisson (count) fit; using the
      presence formula on it would cap every gene's
      prediction at the number of samples and bias exactly
      the high-rate genes the target exists to measure.

    The presence targets are not interchangeable: ≈25% of
    presence events in the "any" target involve a silent
    mutation, which is exactly the part any silent-driven
    per-gene correction can leak into. Report the non-silent
    number whenever the model being scored has seen silent
    counts, and match the target to the likelihood the model
    was fit with.
gene_scaling : pd.Series or None, default None
    Optional per-gene multiplier applied to the rates before
    scoring -- in practice, a ``r_g``. Genes absent from it
    get 1.0.

    There is deliberately no "use r_g" flag: the caller passes
    the ``r_g`` it means, and the two constructors are named
    so the choice is legible at the call site --
    ``gene_scaling=model.compute_r_g_for_evaluation()`` for a
    reportable R², ``compute_r_g_production()`` for the
    published rates. A scaled result is **not stored** in any
    attribute, only returned, so it can never be mistaken for
    the model's own unscaled R².
genes : collection of str or pd.Index or None, default None
    Ensembl gene IDs to restrict scoring to, intersected with
    the usual passenger-gene set. Use together with
    ``train_genes`` on :meth:`estimate_cov_effects` (or the
    channel-model equivalent) to fit on a disjoint training
    gene set and score here on the held-out genes, for
    gene-level cross-validation (see :func:`cross_validation.
    gene_cv_passenger_r2`).
return_per_gene : bool, default False
    If True, also return the per-gene observed/expected pair
    the R² is computed from. Needed to compare two models with
    anything better than a handful of fold-level R² values --
    see the Returns section.

Returns
-------
float
    R² score (coefficient of determination) for passenger
    genes. Values range from -∞ to 1, where:
    - 1.0: Perfect predictions
    - 0.0: Model performs as well as predicting the mean
    - < 0: Model performs worse than predicting the mean
pandas.DataFrame
    Only when ``return_per_gene`` is True: gene-indexed
    ``observed``/``expected`` columns, the two vectors the R²
    summarises. Use these rather than fold-level R² when
    testing whether one model beats another: k-fold CV gives
    only k numbers, and its folds share training data, so a
    t-test over them is both low-powered and anti-conservative.
    Per-gene residuals give ~10^4 observations instead --
    though genes are *not* independent (this pipeline's
    covariates carry real megabase-scale spatial
    autocorrelation), so resample genomic blocks, not
    individual genes.

Raises
------
ValueError
    If base_mus have not been computed (needed to compute
    mu_gs if not already available). Call compute_base_mus()
    first.
ValueError
    If genes_present has not been computed in the dataset.
    Call dataset.compute_gene_presence() first.

Notes
-----
**Method:**

1. Identify passenger genes using Cancer Gene Census
2. Restrict mu_gs and genes_present to passenger genes only
3. For each passenger gene, sum observed presence across all
   samples:
   present_sum_g = Σ_j I[gene g mutated in sample j]
4. For each passenger gene, compute expected number of samples
   where the gene is mutated:
   expected_g = Σ_j (1 - exp(-μ_g^(j)))
5. Compute R² between expected_g and present_sum_g across all
   passenger genes

**Interpretation:**

High R² (close to 1) indicates the model's predicted
mutation rates accurately capture the observed mutation
frequency (number of samples with mutations) for individual
passenger genes. Low or negative R² suggests poor model fit.

This gene-level evaluation assesses whether the model
correctly predicts which genes are mutated more frequently
across samples. Since passenger genes are assumed to be
under neutral selection (not positively or negatively
selected), they provide a clean test set for evaluating the
mutation rate model without confounding selection effects.

**Typical workflow:**

1. Compute baseline mutation rates:
    >>> model.compute_mu_taus()
    >>> model.compute_base_mus()

2. (Optional) Estimate covariate effects:
    >>> model.estimate_cov_effects()  # Also computes R²

3. Evaluate model performance (if not done by estimate_cov_effects):
    >>> r2 = model.estimate_passenger_genes_r2()
    >>> print(f"Passenger genes R²: {r2:.4f}")

Note: You don't need to explicitly call `compute_mu_gs()`
before this method, as it will be called automatically if
needed.

Examples
--------
>>> # Evaluate baseline model (no covariates)
>>> model_no_cov = Model(dataset, None)
>>> model_no_cov.compute_mu_taus()
>>> model_no_cov.compute_base_mus()
>>> # No need to call compute_mu_gs(), it's automatic
>>> r2_baseline = model_no_cov.estimate_passenger_genes_r2()
>>> print(f"Baseline R²: {r2_baseline:.4f}")
>>>
>>> # Evaluate model with covariates
>>> model_with_cov = Model(dataset)
>>> model_with_cov.assign_cov_matrix(cov_matrix)
>>> model_with_cov.compute_mu_taus()
>>> model_with_cov.compute_base_mus()
>>> # estimate_cov_effects() automatically computes R²
>>> model_with_cov.estimate_cov_effects()
>>> r2_with_cov = model_with_cov.passenger_genes_r2  # Already set
>>> print(f"With covariates R²: {r2_with_cov:.4f}")
>>> print(f"Improvement: {r2_with_cov - r2_baseline:.4f}")
>>>
>>> # Compare multiple models
>>> models = {
...     'baseline': model_no_cov,
...     'mrt': model_mrt,
...     'mrt+gexp': model_mrt_gexp}
>>> for name, model in models.items():
...     r2 = model.estimate_passenger_genes_r2()
...     print(f"{name}: R² = {r2:.4f}")

See Also
--------
compute_mu_gs : Must be called first to compute mutation rates
estimate_presence.filter_passenger_genes_ensembl : Identifies
    passenger genes
```

### `models.Model.save_model`

```python
save_model(directory, overwrite=False)
```

```text
Persist this Model's results to disk.

Saves all model components to the specified directory,
creating a structured snapshot that can be reloaded with
:meth:`load_model`.

Parameters
----------
directory : str or Path
    Directory path where model will be saved.
overwrite : bool, default False
    If True, overwrite existing model at this location.
    If False, raises FileExistsError if directory exists.

Directory Structure
-------------------
The saved model creates the following structure::

    directory/
    ├── model_manifest.json       # Metadata and file registry
    ├── cov_matrix.parquet        # Covariate matrix
    ├── pca_basis/                # The PCA's loadings,
    │                             # centres, scales, genes
    ├── mu_taus.parquet           # Per-type mutation rates
    │   └── (or mu_taus/*.parquet for multi-signature)
    ├── base_mus.parquet          # Baseline gene rates
    │   └── (or base_mus/*.parquet for multi-signature)
    ├── cov_effects.npy           # Coefficient estimates
    ├── cov_effects_posteriors.nc # MCMC posterior (if available)
    ├── mu_gs.parquet             # Gene rates with covariates
    ├── mu_ms.parquet             # Variant rates with covariates
    └── gammas/                   # Selection coefficients
        ├── gamma_{variant}.nc    # Per-variant posteriors
        └── gamma_{gene_id}.nc    # Per-gene posteriors

Notes
-----
- Gamma results are saved as individual NetCDF files, one per
  variant or gene, in the ``gammas/`` subdirectory
- File names for gammas use underscores instead of spaces
  (e.g., "BRAF p.V600E" → "gamma_BRAF_p.V600E.nc")
- The manifest tracks all saved files and their locations
- Multi-signature models save separate parquet files per
  signature in subdirectories

See Also
--------
load_model : Reload a saved model from disk

Returns
-------
None

Examples
--------
>>> model.save_model("hnsc/model")
>>> model.save_model("hnsc/model", overwrite=True)  # after new gammas
```

### `models.Model.load_model`

```python
load_model(directory, dataset_dir=None)
```

```text
Load a Model from a directory created by save_model().

Parameters
----------
directory : path-like
    The model directory.
dataset_dir : path-like, optional
    Load the dataset from here instead of the manifest's
    ``dataset_snapshot``. The manifest records an absolute
    path, so a model copied to another machine needs this to
    find its dataset's copy there.

Returns
-------
Model
    With every saved result, and its own copy of the dataset.

Examples
--------
A loaded model reads its own copy of the dataset: hand it the tables
the selection estimates use, which are kept with the dataset.

>>> from sigmutsel import Model, MutationDataset
>>> dataset = MutationDataset.load_dataset("hnsc/HNSC_dataset")
>>> model = Model.load_model("hnsc/model")
>>> for table in ("multi_base_db", "homdel_db", "indel_db"):
...     setattr(model.dataset, table, getattr(dataset, table))
```

## Rates

### `site_rates.SiteRates.from_model`

```python
from_model(
    model,
    r_g='evaluation',
    transcript_models=None,
    coding_in='dataset',
    splice_in='dataset',
    masked_keys='dataset',
    genome_dir='default',
    verify=True,
    assay=None,
    site_context=None,
)
```

```text
The site rates of a fitted channel-split model.

Parameters
----------
model : Model
    With channel baselines (:meth:`Model.compute_channel_base_mus`)
    on a dataset built with the channel universe, tau-dependent
    ``p_{g tau}`` and a signature-summed ``mu_taus``.
r_g : {"evaluation", "none"}
    The gene rate correction in the multiplier. ``"evaluation"``
    (silent-channel ``r_g``, point estimate) is the one gamma
    uses; ``"none"`` reproduces ``mu_ms``. The production
    ``r_g`` is not offered: it is informed by the non-silent
    counts, so it absorbs selection, and no test covers it
    here.
transcript_models : TranscriptModels, optional
    Default: the cached models the dataset's opportunity uses.
coding_in, splice_in : "dataset", None or boolean arrays
    Territory masks; "dataset" reads the dataset's territory.
masked_keys : "dataset", None or int64 array
    Germline-mask alleles; "dataset" loads the dataset's mask.
genome_dir : "default", None or path
    Genome for splice-query contexts; "default" is the installed
    one (:func:`channel_universe.sigprofiler_genome_dir`).
verify : bool
    Check that the site table rebuilds the dataset's four
    channel tables (to 1e-9 relative); raises if not.
assay : str, optional
    On a dataset with assays, this assay's tumors alone. The
    default is all of them: an :class:`AssaySiteRates` when
    there are several assays, one :class:`SiteRates` when
    there is one.
site_context : site_context.ContextFactors, optional
    The pentanucleotide context factor in the placement mode
    (:meth:`with_placement`), each assay on its own table.

Returns
-------
SiteRates or AssaySiteRates
    One :class:`AssaySiteRates` when the dataset's tumors are on
    several assays, else one :class:`SiteRates`.

Examples
--------
About a minute and 2 GB on top of the model; queries then take
about a millisecond per variant.

>>> from sigmutsel.site_rates import SiteRates
>>> site_rates = SiteRates.from_model(model, r_g="evaluation")
```

### `site_rates.SiteRates.variant_rates`

```python
variant_rates(variants, genes=None, per_tumor=True, chunk=50000)
```

```text
Rates of protein (or splice) changes, summed over routes.

Parameters
----------
variants : sequence of str
    ``"GENE p.X"`` labels, or bare changes (``"p.G12D"``) when
    ``genes`` is given.
genes : sequence of str, optional
    Ensembl id (or symbol) per variant; default: the label's
    first word, looked up as an id and then as a GENCODE symbol.
per_tumor : bool
    Return the variants x tumors rates too.

Returns
-------
(pandas.DataFrame, pandas.DataFrame or None, pandas.DataFrame or None)
    ``info`` per variant (channel, n_routes, n_masked_routes,
    routes, route_weights, context, rate_total,
    observable_rate_total, or ``error``), then the per-tumor
    mutation rates and observable rates.

Examples
--------
>>> info, rates, observable = site_rates.variant_rates(
...     ["PIK3CA p.E545K", "PIK3CA p.H1047R"])
>>> info[["channel", "n_routes", "routes"]]  # indexed by variant
>>> observable.sum(axis=1)  # tumors expected to carry each, if neutral
```

### `site_rates.SiteRates.snv_rates`

```python
snv_rates(chrom, pos, alt, per_tumor=True, observable=False)
```

```text
Rates of genomic SNVs (plus-strand ``alt``), vectorised.

A position coding in several genes gives one row per gene; a
position that is none of the table's elements gives none.

Parameters
----------
chrom : sequence of str
    Chromosomes, with or without ``chr``.
pos : sequence of int
    1-based GRCh38 positions (a MAF's ``Start_Position``).
alt : sequence of str
    The alternate base on the plus strand.
per_tumor : bool, default True
    Also return the per-tumor rates.
observable : bool, default False
    Per-tumor rates of an observable call (the ones to compare
    with calls) instead of mutation rates.

Returns
-------
(pandas.DataFrame, pandas.DataFrame or None)
    ``info``: query, ensembl_gene_id, channel, type, weight,
    capture, observable, context, rate_total,
    observable_rate_total; and, with ``per_tumor``, rows x
    tumors mutation rates, or observable rates (each times
    ``capture`` and ``observable``) with ``observable=True``.

Examples
--------
PIK3CA E545K is chr3:179218303 G>A on GRCh38:

>>> info, per_tumor = site_rates.snv_rates(
...     ["chr3"], [179218303], ["A"], observable=True)
>>> info[["ensembl_gene_id", "channel", "observable_rate_total"]]
```

## Selection

### `models.Model.estimate_gene_tumor_dispersion`

```python
estimate_gene_tumor_dispersion(
    excluded_samples=None,
    strata=None,
    min_genes=15,
)
```

```text
Fit the gene-tumor dispersion trend on passenger genes.

Uses how passenger genes' non-silent mutations are allocated
across tumors, against the non-synonymous channel's per-tumor
rates for a channel-split model and the merged baseline
otherwise -- the same rates non-silent gammas are scored
against. Driver genes are excluded:
their allocation reflects selection, not rate. The rate
model's own fit is not touched -- the allocation likelihood
conditions on each gene's total, so ``phi`` is estimable from
the per-tumor rates alone. See :mod:`.gene_tumor_dispersion`.

Parameters
----------
excluded_samples : collection of str or None
    Tumors to leave out, as in :meth:`estimate_gamma`.
strata, min_genes
    Passed to
    :func:`.gene_tumor_dispersion.fit_gene_tumor_dispersion_trend`.

Returns
-------
dict
    The trend, also stored as :attr:`gene_tumor_dispersion_trend`.

Examples
--------
Fit it once, before any gamma with ``gene_tumor_dispersion="fitted"``:

>>> trend = model.estimate_gene_tumor_dispersion(excluded_samples=excluded)
>>> trend["method"], trend["slope"]
```

### `models.Model.estimate_gamma`

```python
estimate_gamma(
    item,
    level=None,
    upper_bound_prior=None,
    store=True,
    non_silent=True,
    excluded_samples=None,
    use_mu_posterior=False,
    r_g_variant='none',
    gene_tumor_dispersion=None,
    hold_out_same_gene_samples=True,
    hold_out_multi_base=True,
    hold_out_homdel=True,
    hold_out_indel=True,
    channel=None,
    absence_floor=None,
    presence_clip=True,
    spectrum_draws=False,
    shared_mu_uncertainty=False,
    mu_uncertainty=None,
    gamma_prior='uniform',
    rng=None,
)
```

```text
Estimate selection coefficient for a variant or gene.

Parameters
----------
item : str
    Variant identifier (e.g., "ZZZ3 p.Y721H"), gene name
    (e.g., "BRAF"), or ensembl_gene_id
    (e.g., "ENSG00000157764").
level : str or None, optional
    Type of item: 'variant', 'gene', or None (auto-detect).
    Default None.
upper_bound_prior : float or None, optional
    Upper bound for gamma prior. If None (default), uses
    :func:`estimate_gammas.estimate_gamma_from_mus`'s own
    default -- that function also auto-expands the bound if
    the posterior turns out to be bound-limited, so an
    explicit value here is only needed to change the
    starting point.
store : bool, optional
    Whether to store result in self.gammas. Default True.
non_silent : bool, optional
    For genes, whether to use non-silent mutations only.
    Default True.
excluded_samples : collection of str or None, default None
    Tumor barcodes to drop from both the "present" and
    "absent" masks before estimation. See
    `_estimate_gamma_variant`/`_estimate_gamma_gene` for
    details.
use_mu_posterior : bool, default False
    Propagate the stage-1 mu posterior into gamma's credible
    interval as a "cut" (see
    :func:`estimate_gammas.estimate_gamma_from_mus`'s "mu
    posterior cut" note and
    :meth:`compute_mu_g_posterior_draws`/
    :meth:`compute_mu_m_posterior_draws`) instead of treating
    each tumor's mu as a fixed point estimate. Requires a
    channel-split model (:meth:`has_channel_base_mus`) with
    a covariate-effect posterior already fit
    (``estimate_channel_rg_cov_effects(sample="full")``);
    for a gene item, also requires ``non_silent=True`` --
    there is no mu-posterior equivalent for the merged/
    silent-channel rate.
r_g_variant : {"none", "production", "evaluation"}, default "none"
    Which ``r_g`` feeds the mu draws when
    ``use_mu_posterior=True``; ignored otherwise. There is
    deliberately no default other than ``"none"``:
    ``"production"`` (both channels) partly absorbs selection
    itself and would bias gamma downward -- see
    :meth:`compute_mu_g_posterior_draws`.
spectrum_draws : bool, default False
    With ``use_mu_posterior=True``, also draw each tumor's
    spectrum from its hybrid spectrum's Dirichlet, so the cut
    carries the spectrum's uncertainty into gamma (see
    :meth:`compute_mu_g_posterior_draws`); ignored otherwise.
shared_mu_uncertainty : bool, default False
    With ``use_mu_posterior=True``, keep the uncertainty the
    tumors' mu draws share (covariate scale, ``r_g``) as one
    latent instead of one independent latent per tumor, which
    averages it away (see
    :func:`.estimate_gammas.estimate_gamma_from_mus`); ignored
    otherwise. The same as ``mu_uncertainty="shared"``.
mu_uncertainty : {"independent", "shared", "draws"} or None
    How the cut carries the rate draws into gamma (see
    :func:`.estimate_gammas.estimate_gamma_from_mus`): one
    lognormal per tumor, plus a shared latent, or the draws
    themselves (the exact cut, on a grid, no MCMC). ``None``
    reads ``shared_mu_uncertainty``. "draws" without
    ``use_mu_posterior`` is the grid with the point rates.
gamma_prior : {"uniform", "jeffreys", "loguniform"}
    Prior on gamma (see
    :func:`.estimate_gammas.estimate_gamma_from_mus`); the
    default "uniform" is the one cancereffectsizeR parity is
    checked with. With few carriers, "jeffreys" when the
    interval is read against gamma = 1, "uniform" when the
    estimate itself is reported (see there).
rng : numpy.random.Generator, optional
    Draws the stage-1 posterior draws need (the ``r_g`` draws)
    and the exact cut's gamma samples. Default: unseeded for
    the former, ``constants.random_seed`` for the latter, as
    before. One generator per item
    (:func:`.parallel.item_rng`) makes a gamma independent of
    the order items are fitted in, which is what lets
    :meth:`estimate_gammas` run them in parallel.
gene_tumor_dispersion : None, float or "fitted", default None
    Gene-tumor dispersion in the presence likelihood (see
    :func:`estimate_gammas.estimate_gamma_from_mus`). ``None``
    keeps the dispersion-free likelihood; a float is used as
    ``phi`` directly; ``"fitted"`` takes ``phi`` for the
    gene's own non-silent mutation count from
    :attr:`gene_tumor_dispersion_trend` (see
    :meth:`estimate_gene_tumor_dispersion`). For a gene, only with
    ``non_silent=True``. For a variant, the variant inherits
    its gene's per-tumor dispersion: the multiplier acts on the
    (gene, tumor) pair, so each tumor's Gamma shape is
    ``phi_gene * p_gene,j`` with ``p`` the gene's allocation of
    rate across tumors and ``phi_gene`` taken at the gene's own
    mutation count.
hold_out_same_gene_samples : bool, default True
    Variants only (ignored for a gene). Leave out of the
    "without" set every tumor that carries a *different*
    non-silent mutation in the variant's gene, rather than
    counting it as evidence against selection: under mutual
    exclusivity such a tumor may have met the same selective
    pressure by another hit. The "with" set is untouched. See
    :meth:`_estimate_gamma_variant`.
hold_out_multi_base : bool, default True
    Genes and variants. Leave out of the "without" set every
    tumor with a non-silent multi-base event (a doublet, or a
    cluster within 2 bp) in the gene
    (:meth:`MutationDataset.multi_base_hits`). The rates are
    SBS rates and gamma is about SBS hits; such a tumor has met
    the gene's selective pressure by an event the SBS model does
    not describe, so its lack of an SBS hit is not evidence
    against selection. Only that gene's (and its variants')
    likelihood changes. Recorded as
    ``n_tumors_held_out_multi_base`` (and in
    ``n_tumors_held_out``). Without
    :attr:`MutationDataset.multi_base_db` it holds out nobody.
hold_out_homdel : bool, default True
    Genes and variants. Leave out of the "without" set every
    tumor with a homozygous deletion of the gene
    (:meth:`MutationDataset.homdel_hits`): it has no
    opportunity for a point mutation where the gene is gone,
    and a tumor suppressor lost this way has met its pressure.
    Recorded as ``n_tumors_held_out_homdel``. Without
    :attr:`MutationDataset.homdel_db` it holds out nobody.
hold_out_indel : bool, default True
    Genes and variants. Leave out of the "without" set every
    tumor with a non-silent coding insertion or deletion in
    the gene (:meth:`MutationDataset.indel_hits`): the rates
    are single-base rates, and a tumor that reached the gene
    by a frameshift or an in-frame indel has met its
    pressure by an event they do not describe. With
    ``hold_out_same_gene_samples`` this gives a variant the
    same absent set as cancereffectsizeR's same-gene
    hold-out, which reads every call in the gene, except that
    a tumor whose only hit there is synonymous stays in.
    Recorded as ``n_tumors_held_out_indel``. Without
    :attr:`MutationDataset.indel_db` it holds out nobody.
channel : str or None, default None
    Genes only, channel-split models only. None (or
    ``"nonsyn"``) is the gene gamma on its whole non-synonymous
    rate. Otherwise one consequence channel
    (:data:`GAMMA_CHANNELS`): ``"mis"``, ``"plof"`` (predicted
    loss-of-function: nonsense plus essential splice), ``"non"``, ``"spl"``, or ``"syn"``
    (a negative control, expected near 1). Presence counts that
    channel's calls, the rate is the gene's on that channel, a
    dispersed fit inherits the gene's per-tumor dispersion, and
    no other channel's hits are held out. Stored under
    ``"<gene_id>__<channel>"``. A gene gamma averages over every
    site of the channel, so for an oncogene report variant
    gammas; the channel gammas are for tumor suppressors and for
    telling the two apart. ``"esyn"`` is the exon-end
    synonymous channel.
absence_floor : float or None, default None
    ``u``, the least probability that a tumor does not carry
    the mutation whatever its rate: the declared robust
    likelihood of :func:`.estimate_gammas.presence_probability`.
    None uses :data:`.estimate_gammas.ABSENCE_FLOOR`; 0 keeps
    only the numerical clip, as before it existed.
presence_clip : bool, default True
    The numerical clip on P(present) (see
    :func:`.estimate_gammas.estimate_gamma_from_mus`). False
    scores the exact likelihood: with
    ``gene_tumor_dispersion=None``, ``use_mu_posterior=False``
    and ``absence_floor=0`` it is cancereffectsizeR's.

Returns
-------
dict
    Estimation results for the item.

Examples
--------
>>> # Auto-detect variant
>>> model.estimate_gamma("ZZZ3 p.Y721H")
>>> # Auto-detect gene by name
>>> model.estimate_gamma("BRAF")
>>> # Auto-detect gene by ensembl_gene_id
>>> model.estimate_gamma("ENSG00000157764")
>>> # Explicit level specification
>>> model.estimate_gamma("BRAF", level='gene', non_silent=True)
>>> # Mu-posterior cut, evaluation r_g (channel-split model)
>>> model.estimate_gamma(
...     "BRAF", use_mu_posterior=True, r_g_variant="evaluation"
... )
```

### `models.Model.estimate_gammas`

```python
estimate_gammas(requests, *, seed=None, n_jobs=None)
```

```text
Several gammas, one :meth:`estimate_gamma` call each, in parallel.

Each request is a dict of :meth:`estimate_gamma`'s arguments
with the item under ``"item"`` (``store`` and ``rng`` are set
here). The calls are independent, so they run across
``n_jobs`` forked processes (:func:`.parallel.map_ordered`),
and each result is stored in :attr:`gammas` under the key
:meth:`estimate_gamma` gives it, as a serial call would.

With ``seed``, every item draws from its own generator
(:func:`.parallel.item_rng`, keyed by the item and its
channel), so its gamma is the same however many processes run
and in whatever order: ``n_jobs=1`` and ``n_jobs=8`` give equal
posteriors. Without it the draws are unseeded, as in a plain
:meth:`estimate_gamma` call.

Parameters
----------
requests : list of dict
seed : int, optional
n_jobs : int, optional
    Default :data:`.constants.n_jobs` (1, the serial path).

Returns
-------
list of (str or None, object)
    Per request, in order: the key the gamma was stored under
    and its result, or ``None`` and the exception
    :meth:`estimate_gamma` raised (nothing is stored for it).
    Each call's run-history entry is recorded on this model
    either way.

Examples
--------
>>> settings = dict(excluded_samples=excluded, use_mu_posterior=True,
...                 mu_uncertainty="draws", r_g_variant="evaluation",
...                 gene_tumor_dispersion="fitted")
>>> requests = [dict(item=g, level="gene", **settings)
...             for g in ("TP53", "PIK3CA", "NOTCH1")]
>>> results = model.estimate_gammas(requests, seed=0, n_jobs=3)
>>> [key for key, _ in results]
```

### `estimate_gammas.summarize_gamma`

```python
summarize_gamma(result, prob=0.89)
```

```text
Point estimate and interval of a gamma, or an upper bound only.

A gamma fitted with no tumor carrying the mutation -- typical of a
consequence channel of an oncogene, such as the predicted
loss-of-function channel of NRAS -- is informed only by the tumors that do not carry it.
Its posterior is then the prior cut down by the absences: it says
how large gamma can be, not where it is, and its mean depends on
the prior's shape. Such a gamma is reported as an upper bound
alone: the ``(1 + prob) / 2`` quantile, the upper end of the
equal-tailed ``prob`` interval it would otherwise have had.

Parameters
----------
result : arviz.InferenceData
    A gamma posterior. Whether it had carriers is read from
    ``posterior.attrs["n_tumors_with"]``; a posterior without
    that count (fitted before it was recorded) is summarized as
    an estimate.
prob : float, default=0.89
    Probability mass of the equal-tailed interval.

Returns
-------
dict
    ``mean``, ``lower`` and ``upper`` of gamma and
    ``upper_bound_only`` (bool). With ``upper_bound_only``,
    ``mean`` and ``lower`` are NaN and ``upper`` is the bound.

Examples
--------
>>> from sigmutsel.estimate_gammas import summarize_gamma
>>> settings = dict(excluded_samples=excluded, use_mu_posterior=True,
...                 mu_uncertainty="draws", r_g_variant="evaluation",
...                 gene_tumor_dispersion="fitted")
>>> result = model.estimate_gamma("TP53", level="gene", **settings)
>>> summarize_gamma(result)
>>> plof = model.estimate_gamma("PIK3CA", level="gene", channel="plof",
...                             **settings)
>>> summarize_gamma(plof)["upper_bound_only"]  # if no tumor carries one
```

### `models.Model.gene_selection_table`

```python
gene_selection_table(
    genes=None,
    prior=None,
    slab_coef=None,
    guard=True,
    likelihoods=None,
    excluded_samples=None,
    fdr=0.1,
    n_draws=200,
    seed=0,
    gene_tumor_dispersion='fitted',
    threshold=2.0,
    store=True,
    n_jobs=None,
)
```

```text
Whether and how strongly each gene is selected, in one table.

**The call** is the null test (:meth:`gene_selection_null_test`)
with Benjamini-Hochberg over the genes in the table: ``called``
when ``q < fdr``. **The strength** is gamma's posterior under
the empirical-Bayes spike + slab prior (:mod:`.selection`),
fitted on the same genes (or given), with the guard. The flat
prior's interval is kept beside it, as an estimate only: read
across many genes it calls neutral genes above one.

Parameters
----------
genes : iterable, optional
    Default: every gene with complete covariates in the
    ``r_g`` fit, drivers included.
prior : selection.SpikeSlabPrior, optional
    Use this prior instead of fitting one.
slab_coef : array, optional
    A pooled slab (:func:`.selection.fit_pooled_spike_slab`):
    only this cohort's spike is fitted.
guard : bool, default True
    See :func:`.selection.fit_spike_slab`.
likelihoods : selection.GeneSelectionLikelihoods, optional
    From :meth:`gene_selection_likelihoods` (computed if None,
    with ``excluded_samples``, ``n_draws``, ``seed`` and
    ``gene_tumor_dispersion``).
fdr : float, default 0.1
threshold : float, default 2.0
    ``eb_p_above`` is the posterior P(gamma > threshold).
store : bool, default True
    Keep the table in :attr:`selection_table` and the prior in
    :attr:`selection_prior` (both saved with the model).
n_jobs : int, optional
    Processes for the per-gene work (the likelihoods, the
    marginals the prior is fitted on, the posteriors); default
    :data:`.constants.n_jobs`, 1. The table does not depend on
    it.

Returns
-------
pandas.DataFrame
    Indexed by gene_id: ``n_tumors_with``, ``expected_with``,
    ``p_mid``, ``q``, ``called``, ``eb_median``, ``eb_lower``,
    ``eb_upper`` (89%), ``eb_p_above``, ``flat_median``,
    ``flat_lower``, ``flat_upper``, ``upper_bound_only`` (no
    carrier: the flat interval bounds gamma from above only),
    and the fit's accounting (``n_tumors_without``,
    ``n_tumors_held_out``, ``phi``, ``covariate_fallback``).

Examples
--------
>>> model.estimate_gene_tumor_dispersion(excluded_samples=excluded)
>>> table = model.gene_selection_table(
...     excluded_samples=excluded, fdr=0.1, n_jobs=4)
>>> table[table.called].sort_values("q")[
...     ["n_tumors_with", "expected_with", "q", "eb_median"]].head()
```

## Every other public name

One line each; `help(name)` or the source has the rest.

| name | summary |
|---|---|
| `capture.CaptureTerritory` | One assay's capture territory (see the module docstring). |
| `capture.CaptureTerritory.from_dict` |  |
| `capture.CaptureTerritory.g_at` | The capture-GC factor at GC fractions ``gc``. |
| `capture.CaptureTerritory.label` | Short content hash (BED content included), for cache paths. |
| `capture.CaptureTerritory.to_dict` |  |
| `capture.capture_weights` | Per-site observability ``T`` under ``capture``. |
| `capture.exon_gc` | GC fraction of each exon (over its A/C/G/T coding bases). |
| `capture.exon_index` | Exon number of every coding flat position (-1 for padding). |
| `capture.splice_flat_index` | Flat index of the coding base each splice site hangs off. |
| `channel_universe.TranscriptModels` | Chosen transcripts as flat arrays, one block per gene. |
| `channel_universe.TranscriptModels.build` | Assemble models from a selection, GTF features and FASTA. |
| `channel_universe.TranscriptModels.load` |  |
| `channel_universe.TranscriptModels.neighbours` | Coding-strand left and right neighbour of every position. |
| `channel_universe.TranscriptModels.save` |  |
| `channel_universe.agreement_table` | Cross-tabulate the MAF's class against our channel. |
| `channel_universe.channel_opportunity` | Cached channel opportunity tables, optionally subset to genes. |
| `channel_universe.classify_calls` | Place each SNV call on its gene's chosen transcript. |
| `channel_universe.compute_channel_opportunity` | Per-gene, per-type site counts of the five channels. |
| `channel_universe.contexts_from_channels` | Genes x 32 contexts from the channel tables. |
| `channel_universe.credit_calls_to_owners` | Give each call on a contested position to the position's owner. |
| `channel_universe.exon_end_positions` | Per flat position: within ``bases`` of an exon-exon junction. |
| `channel_universe.gene_priority` | Rank of every gene when two claim one position (0 is strongest). |
| `channel_universe.model_calls` | The calls a model input reads, optionally one consequence scope. |
| `channel_universe.position_owners` | Which gene owns each coding position and essential splice site. |
| `channel_universe.read_cds_fasta` | Map stable transcript id -> (versioned id, upper-case sequence). |
| `channel_universe.read_coding_transcripts` | Coding transcripts and their CDS/stop features from a GTF. |
| `channel_universe.repeat_masks` | Coding positions and splice sites inside a RepeatMasker region. |
| `channel_universe.resolve_keep_ids_for_universe` | Gene ids from a ``restrict_to_db`` argument, or None for all. |
| `channel_universe.select_transcripts` | Choose one transcript per gene. |
| `channel_universe.sigprofiler_genome_dir` | SigProfilerMatrixGenerator's installed reference, or None. |
| `channel_universe.splice_flank_composition` | Base composition at donor +3 and acceptor -3, read from a genome. |
| `channel_universe.territory_masks` | Which coding positions and splice sites lie in the territory. |
| `compound_variants.compound_presence` | Whether each tumor carries any member of a compound. |
| `compound_variants.compound_rates` | Per-tumor rate of acquiring any member of a compound. |
| `compound_variants.define_compound_variants` | Group variants into compound variants. |
| `compound_variants.summarize_compounds` | One row per compound: size, and the genes it spans. |
| `compute_alphas.estimate_alphas` | Return per-sample signature probabilities (`alphas`). |
| `compute_mutation_burden.count_mutation_burden` | Count total and synonymous mutations per sample. |
| `compute_mutation_burden.estimate_ell_hats` | Estimate adjusted mutation burdens (ell_hat) per sample. |
| `consequence_contexts_by_gene.compute_consequence_contexts_by_gene` | Split each gene's SBS opportunities into syn/non-syn channels. |
| `constants.extract_context` | Extract trinucleotide context from mutation type. |
| `constants.reverse_complement` | Return the reverse complement of a DNA sequence. |
| `contexts_by_gene.compute_contexts_by_gene` | Build a 32-context table for the longest transcript of each gene. |
| `contexts_by_gene.extract_ensg` | Return the stable ENSG (no version) of a FASTA header, or None. |
| `contexts_by_gene.load_or_generate_contexts_by_gene` | Load or generate trinucleotide context counts by gene. |
| `contexts_by_gene.normalize_fasta_paths` | Normalise the *fasta_files* argument to a list of Paths. |
| `contexts_by_gene.resolve_keep_ids` | Turn a *restrict_to_db* argument into a set of Ensembl IDs. |
| `contexts_by_gene.select_longest_sequences` | Map each Ensembl gene ID to its longest transcript sequence. |
| `cross_validation.channel_gene_cv_passenger_r2` | Gene-level k-fold CV for the consequence-split ``r_g`` model. |
| `cross_validation.gene_cv_passenger_r2` | Gene-level k-fold cross-validated passenger-gene R² (single channel). |
| `download_tcga_data.main` | CLI entrypoint for downloading TCGA mutation data. |
| `download_tcga_data.parse_args` | Parse CLI arguments for download helper. |
| `download_tcga_data.process_tcga_maf_downloads` | Move MAFs downloaded by gdc-client into ``destination_dir``. |
| `estimate_covariates_effect.estimate_all_cov_effects` | Estimate covariate-effect over all (or selected) covariates. |
| `estimate_covariates_effect.estimate_channel_covariates_effect` | Fit one shared ``c`` against two consequence channels jointly. |
| `estimate_covariates_effect.estimate_covariates_effect` | Estimate covariate effects (with intercept) in a Bernoulli model. |
| `estimate_covariates_effect.load_or_estimate_all_cov_effects` | Load or estimate covariate effects with caching support. |
| `estimate_gammas.ConvergenceError` | MCMC sampling completed but convergence diagnostics are bad. |
| `estimate_gammas.estimate_gamma_from_mus` | Estimate gamma from mu values using a Poisson observation model. |
| `estimate_gammas.log_absence_probability` | ``log P(tumor lacks the variant)``, exactly, with no clip. |
| `estimate_gammas.presence_probability` | P(tumor carries the variant), given gamma and its rate. |
| `estimate_mus.compute_mu_g_channel_per_tumor` | Per-gene rate for one consequence channel (syn or non-syn). |
| `estimate_mus.compute_mu_g_per_tumor` | Compute baseline per-gene expected mutation rate per tumor. |
| `estimate_mus.compute_mu_m_per_tumor` | Compute per-variant expected mutation rate per tumor. |
| `estimate_mus.compute_mu_tau_per_tumor` | Compute per-tumor per-type baseline mutation rates. |
| `estimate_mus.compute_mus_per_gene_per_sample` | Return per-gene, per-sample mutation rates. |
| `estimate_mus.compute_n_taus` | Compute counts per mutation type from contexts or a MAF. |
| `estimate_mus.covariate_complete_genes` | Genes whose covariate row is complete (no NaN). |
| `estimate_mus.route_weights_of` | One site weight per route of a variant, aligned with its types. |
| `estimate_mus.type_denominators` | ``D_tau``: the per-type opportunity over the gene universe. |
| `estimate_mus.variant_site_denominators` | Per-(gene, type) site counts that turn a gene rate into a site rate. |
| `estimate_presence.aid_target_ensembl_ids` | Ensembl gene ids of :func:`aid_target_symbols`'s genes. |
| `estimate_presence.aid_target_symbols` | Symbols of the genes activation-induced cytidine deaminase targets. |
| `estimate_presence.artifact_prone_ensembl_ids` | Ensembl gene ids of :func:`artifact_prone_symbols`'s genes. |
| `estimate_presence.artifact_prone_symbols` | Symbols of genes prone to mapping and calling artifacts. |
| `estimate_presence.cancer_gene_ensembl_ids` | Ensembl gene ids of :func:`cancer_gene_symbols`'s genes. |
| `estimate_presence.cancer_gene_symbols` | Symbols of every known cancer gene: the Cancer Gene Census, IntOGen's compendium of mutational driver genes and the driver genes of three published screens, unioned. |
| `estimate_presence.compute_genes_counts` | Build a matrix of mutation counts per gene per tumor. |
| `estimate_presence.compute_genes_present` | Build a 0/1 matrix of gene presence per tumor. |
| `estimate_presence.compute_variants_present` | Build a 0/1 matrix of variant present per tumor. |
| `estimate_presence.filter_giant_non_driver_genes` | Return giant genes with no evidence of being cancer drivers. |
| `estimate_presence.filter_passenger_genes` | Return a list of passenger genes present in the mutation data. |
| `estimate_presence.filter_passenger_genes_ensembl` | Return Ensembl IDs of passenger genes in `db`. |
| `estimate_presence.filter_silent_variants` | Return variants (as an Index) that are annotated as 'Silent' in `db`. |
| `estimate_presence.histone_ensembl_ids` | Ensembl gene ids of :func:`histone_symbols`'s genes. |
| `estimate_presence.histone_symbols` | Symbols of every gene in HGNC's five histone gene groups. |
| `estimate_presence.non_passenger_ensembl_ids` | Ensembl gene ids of :func:`non_passenger_symbols`'s genes. |
| `estimate_presence.non_passenger_symbols` | Everything passenger fits exclude. |
| `estimate_rg.channel_rg_log_likelihood` | Marginal log-likelihood of the two channels with ``r_g`` integrated out. |
| `estimate_rg.estimate_channel_rg_effect` | Fit shared ``c`` and ``θ`` with ``r_g`` marginalized out. |
| `estimate_rg.fit_fallback_shifts` | Block shifts ``d`` for genes whose covariates are incomplete. |
| `estimate_rg.r_g_draws_for_evaluation` | Draws from ``r_g``'s exact posterior, silent channel **only**. |
| `estimate_rg.r_g_draws_production` | Draws from ``r_g``'s exact posterior, **both** channels. |
| `estimate_rg.r_g_production` | Posterior mean of ``r_g`` using **both** channels (production). |
| `estimate_rg.r_g_silent_only_for_evaluation` | Posterior mean of ``r_g`` from the silent channel **only**. |
| `figures.comparison_with_observed` | Make an expected vs observed scatter plot for mutation counts. |
| `figures.plot_posteriors_vs_counts` | Plot posterior mean ± 94 % HDI against observed sample counts. |
| `figures.plot_sig_decomposition_with_burden` | Heatmap of signature attribution probabilities ordered by burden. |
| `figures.plot_signature_correlations` | Plot correlations with bars and mutation counts. |
| `figures.plot_signature_profile` | Draw one SBS96 profile as bars, grouped by substitution. |
| `figures.sbs96_order` | Sort SBS96 mutation types by substitution, then by context. |
| `gene_tumor_dispersion.dirichlet_multinomial_logpmf` | Per-gene Dirichlet-Multinomial log-likelihood of an allocation. |
| `gene_tumor_dispersion.fit_gene_tumor_dispersion_trend` | Fit ``log phi = intercept + slope * log N_g``, if the data support it. |
| `gene_tumor_dispersion.fit_phi` | Maximum-likelihood ``phi`` for a set of genes' allocations. |
| `gene_tumor_dispersion.phi_for_count` | ``phi`` for a gene with ``n`` mutations; ``inf`` means none. |
| `germline_mask.allele_keys` | One ``int64`` per (chromosome, position, alternate base). |
| `germline_mask.build_gnomad_snv_table` | Stream gnomAD v2.1.1 exomes and keep every SNV with its AF. |
| `germline_mask.calls_on_masked_alleles` | Boolean Series: which calls of a MAF-style table are masked. |
| `germline_mask.default_table_path` | Where the full gnomAD SNV table is cached (under the data dir). |
| `germline_mask.distributed_table_path` | Where the distributed (AF > 5e-5) table is cached. |
| `germline_mask.download_germline_mask_table` | Download the distributed table (15 MB) and check its SHA-256. |
| `germline_mask.load_germline_mask` | Sorted allele keys of the gnomAD SNVs with ``AF > af_threshold``. |
| `germline_mask.mask_label` | Short name of a mask, used in cache paths and manifests. |
| `germline_mask.resolve_table` | The table that serves ``af_threshold``, fetched if need be. |
| `germline_mask.write_distributed_table` | Cut the distributed table out of a full one (for maintainers). |
| `liftover.in_intervals` | Whether each 1-based position lies in the merged intervals. |
| `liftover.lift_positions` | Map 1-based positions through chain blocks. |
| `liftover.merged_bed_intervals` | Per-chromosome disjoint intervals of a BED, optionally padded. |
| `liftover.read_chain` | Parse a (gzipped) UCSC chain file into aligned blocks. |
| `load_maf_files.build_id83_lookup` | Create a DataFrame with all mutation type. |
| `load_maf_files.compact_data` | Reduce MAF DataFrame to minimal columns to continue analysis. |
| `load_maf_files.create_mutation_type_column` | Add a mutation type column in COSMIC-style. |
| `load_maf_files.create_mutation_type_column_id` | Annotate a mutation Data-frame with its COSMIC ID-83 type. |
| `load_maf_files.create_mutation_type_column_snp` | Add a mutation type column in COSMIC-style SBS96 format. |
| `load_maf_files.filter_db` | Filter a MAF-style DataFrame to retain only `variant_type`. |
| `load_maf_files.generate_compact_db` | Generate a compact mutation database from MAF files. |
| `load_maf_files.load_or_generate_compact_db` | Load or generate a compact database of mutations. |
| `load_maf_files.load_validate_compact_all_maf_files_parallel` | Load and process all MAF files in a directory in parallel. |
| `load_maf_files.process_single_maf` | Process a single MAF file into a cleaned and compact DataFrame. |
| `load_maf_files.read_coding_indels` | Insertions and deletions that change a protein, from MAF files. |
| `load_maf_files.reduce_context_to_trinucleotides` | Trim CONTEXT column to the central trinucleotide. |
| `load_maf_files.select_coding_indels` | The non-silent coding indel rows of a MAF table. |
| `load_maf_files.validate_alleles_snv` | Validate SNV alleles in a MAF dataframe. |
| `load_maf_files.validate_chromosome` | Validate 'Chromosome' column and filter out invalid entries. |
| `load_maf_files.validate_context_snv` | Validate the 'CONTEXT' column for expected format and characters. |
| `load_maf_files.validate_full` | Run all validation checks on a MAF DataFrame. |
| `load_maf_files.validate_reference_matches_context_snv` | Validate that Reference_Allele matches center of CONTEXT. |
| `load_signature_matrix.load_signature_matrix` | Load a mutational signature matrix from file. |
| `locations.check_data_file` | Check if a data file exists, provide helpful error if not. |
| `locations.get_data_dir` | Get the package data directory. |
| `locations.list_data_files` | List all reference data files and their availability. |
| `models.AssayView` | A :class:`MutationDataset` as one assay's tumors see it. |
| `models.Model.aggregate_signatures` | Aggregate signature-separated base_mus into chosen signatures. |
| `models.Model.channel_rg_log_likelihood_at_fit` | Marginal log-likelihood at the fitted ``c`` and ``θ``. |
| `models.Model.compute_base_mus` | Compute baseline mutation rates per gene per tumor. |
| `models.Model.compute_channel_mu_gs` | Covariate-scaled per-gene rates for one consequence channel. |
| `models.Model.compute_mu_g_posterior_draws` | Per-tumor mu posterior draws for one gene's non-silent rate. |
| `models.Model.compute_mu_gs` | Compute per-gene, per-sample mutation rates. |
| `models.Model.compute_mu_m_posterior_draws` | Per-tumor rate draws of one variant, per assay. |
| `models.Model.compute_mu_ms` | Per-variant rates (:meth:`_compute_mu_ms_single`), per assay. |
| `models.Model.compute_mu_taus` | Compute mutation burden (total mutations) per tumor. |
| `models.Model.compute_r_g_production` | Per-gene ``r_g`` from **both** channels -- the paper number. |
| `models.Model.compute_rg_fallback_statistics` | Gather the ``r_g`` statistics of the covariate-less genes. |
| `models.Model.copy` | Create a copy of the model with shared dataset and base results. |
| `models.Model.estimate_channel_cov_effects` | Fit one shared ``c`` against both consequence channels. |
| `models.Model.estimate_cov_effects` | Estimate covariate effect coefficients via MAP or MCMC. |
| `models.Model.estimate_fallback_shifts` | Fit ``d_B`` for the covariate-fallback genes. |
| `models.Model.estimate_gamma_compound` | Estimate one gamma shared by several variants. |
| `models.Model.gamma_sample_accounting` | Tumor counts behind each stored gamma, as a table. |
| `models.Model.gene_selection_likelihoods` | Each gene's gamma likelihood on the grid, over stage-1 draws. |
| `models.Model.gene_selection_null_test` | Upper mid-p of each gene's carrier count at gamma = 1. |
| `models.Model.has_base_mus` | Check if baseline mutation rates have been computed. |
| `models.Model.has_channel_base_mus` | Whether both consequence channels' baselines exist. |
| `models.Model.has_cov_effects` | Check if covariate effects have been estimated. |
| `models.Model.has_cov_effects_posteriors` | Check if covariate-effect posterior samples exist. |
| `models.Model.has_covariates` | Check if model uses covariates (not a baseline model). |
| `models.Model.has_mu_gs` | Check if gene-level mutation rates have been computed. |
| `models.Model.has_mu_ms` | Check if variant-level mutation rates have been computed. |
| `models.Model.has_mu_taus` | Check if mutation burdens have been computed. |
| `models.Model.has_sig_dependent_mus` | Check if base_mus are signature-dependent (dict). |
| `models.Model.is_submodel_of` | Check if this model is a submodel of another model. |
| `models.Model.is_supermodel_of` | Check if this model is a supermodel of another model. |
| `models.Model.plot_gamma_results` | Plot posterior vs counts for selection results. |
| `models.Model.plot_signature_correlations` | Plot signature correlations with covariates. |
| `models.Model.set_site_context` | Turn the context factor's placement mode on (or off, None). |
| `models.Model.signature_effect_shares` | Share of this cohort's selection due to each signature. |
| `models.Model.with_covariates_removed` | Return a copy of the model without the specified covariates. |
| `models.MutationDataset.assay_groups` | Assay name -> the given tumors on it (order kept), or None. |
| `models.MutationDataset.assay_view` | The dataset as assay ``name`` sees it (:class:`AssayView`). |
| `models.MutationDataset.build_multi_base_db` | Recover the multi-base rows of a dataset built without them. |
| `models.MutationDataset.classify_mutation_db` | Place every call in the channel universe. |
| `models.MutationDataset.compute_gene_counts_channels` | Compute per-channel mutation *count* matrices. |
| `models.MutationDataset.compute_gene_presence` | Compute gene presence matrix from mutation database. |
| `models.MutationDataset.compute_gene_presence_non_silent` | Compute non-silent gene presence matrix. |
| `models.MutationDataset.compute_gene_presence_silent` | Compute silent (synonymous) gene presence matrix. |
| `models.MutationDataset.compute_variants_present` | Compute variant presence matrix. |
| `models.MutationDataset.counts_by` | Tumor counts per variant or gene, split by tumor group. |
| `models.MutationDataset.generate_consequence_contexts_by_gene` | Generate synonymous/non-synonymous opportunity counts. |
| `models.MutationDataset.generate_contexts_by_gene` | Generate trinucleotide context counts by gene. |
| `models.MutationDataset.generate_mutational_matrices` | Generate mutational matrices using SigProfilerMatrixGenerator. |
| `models.MutationDataset.generate_variant_db` | Generate variant database from the mutation database. |
| `models.MutationDataset.genes_present_channel` | 0/1 genes x tumors: a call of one consequence channel. |
| `models.MutationDataset.has_channel_counts` | Check if both channels' count matrices exist. |
| `models.MutationDataset.has_channel_universe` | Whether the mutation table carries the channel universe. |
| `models.MutationDataset.has_consequence_contexts_by_gene` | Whether the consequence-split opportunity tables exist. |
| `models.MutationDataset.has_gene_presence` | Check if gene presence matrix has been computed. |
| `models.MutationDataset.has_mutation_db` | Check if mutation database has been loaded. |
| `models.MutationDataset.has_mutational_matrices` | Check if mutational matrices have been generated. |
| `models.MutationDataset.has_non_silent_presence` | Check if non-silent gene presence has been computed. |
| `models.MutationDataset.has_silent_presence` | Check if silent gene presence has been computed. |
| `models.MutationDataset.has_variants` | Check if variants have been loaded. |
| `models.MutationDataset.homdel_hits` | Tumors with a homozygous deletion of a gene. |
| `models.MutationDataset.indel_hits` | Tumors with a non-silent coding indel in a gene. |
| `models.MutationDataset.multi_base_hits` | Tumors with a non-silent multi-base event in a gene. |
| `models.MutationDataset.record_qc_steps` | Add rows to :attr:`qc_ledger`, replacing their steps. |
| `models.MutationDataset.relabel_exon_end_synonymous` | Move exon-end synonymous calls and sites to the ``esyn`` channel. |
| `models.MutationDataset.run_signature_decomposition` | Run COSMIC signature decomposition on mutational matrices. |
| `models.MutationDataset.set_assays` | Give each tumor the capture territory of its assay. |
| `models.MutationDataset.set_capture` | Set (or clear, with None) the capture territory. |
| `models.MutationDataset.set_germline_mask` | Set (or clear, with None) the germline mask. |
| `models.MutationDataset.set_site_weights` | Set (or clear, with None) site weights on the opportunity. |
| `mutational_matrices_generator.mutational_matrices_generation` | Load or generate mutational matrices from MAF/VCF files. |
| `mutational_matrices_generator.run_mutational_matrices_generator` | Generate mutational matrices. |
| `parallel.item_rng` | A generator for one item, the same whatever order items run in. |
| `parallel.map_ordered` | ``[func(x) for x in items]``, across ``n_jobs`` forked processes. |
| `parallel.numpy_backed` | ``posterior`` with every array a NumPy array. |
| `parallel.resolve_n_jobs` | ``n_jobs``, or :data:`.constants.n_jobs`; -1 means every core. |
| `provenance.check_provenance` | Warn if `manifest` was written by a different build. |
| `provenance.format_run_history` | Return `history` as lines, for printing or logging. |
| `provenance.git_commit` | Return ``git describe`` of the tree this package runs from. |
| `provenance.package_provenance` | Return the provenance stamp to write into a manifest. |
| `provenance.package_version` | Return ``sigmutsel.__version__``, or ``"unknown"``. |
| `provenance.record_call` | Record calls to a state-mutating method in ``_run_history``. |
| `provenance.record_event` | Append a history entry for something that is not a call. |
| `qc.apply_qc` | Run the full QC sequence and return a problem-tagged DataFrame. |
| `qc.check_sample_overlap` | Flag sample pairs with suspiciously high mutation overlap. |
| `qc.detect_mnv_dbs` | Tag adjacent same-sample SNVs likely from one real event. |
| `qc.flag_artifact_signature_mutations` | Tag mutations dominantly attributed to a technical-artifact signature. |
| `qc.flag_exact_duplicates` | Tag exact-duplicate mutation records. |
| `qc.flag_germline_variants` | Tag likely-germline sites by gnomAD population frequency. |
| `qc.flag_repetitive_regions` | Tag rows overlapping a RepeatMasker-annotated region. |
| `qc.flag_treatment_signature_samples` | Tag every mutation of a sample with implausible treatment load. |
| `qc.summarize_problems` | Return a ``{problem: count}`` summary, for logging. |
| `qc.validate_full_with_problems` | Like :func:`load_maf_files.validate_full`, but tags, not drops. |
| `qc_ledger.concat` | Concatenate ledgers, sorted by tumor and production order. |
| `qc_ledger.empty_ledger` | A ledger with no rows and the ledger's columns. |
| `qc_ledger.ledger_rows` | Ledger rows from per-tumor counts. |
| `qc_ledger.load_rows` | The load-time ledger rows of one MAF file. |
| `qc_ledger.replace_steps` | *ledger* with *steps* replaced by *rows* (a step rerun). |
| `qc_ledger.step_order` | The production order of a step (see :data:`STEP_ORDER`). |
| `qc_ledger.totals` | Per-step totals over tumors, in production order. |
| `repeatmasker.download_distributed_table` | Download the distributed table to ``dest`` and check its SHA-256. |
| `repeatmasker.write_distributed_table` | Cut the distributed table out of the full rmsk dump (for maintainers). |
| `sample_qc.annotate_local_copy_number` | Add a local total-copy-number column to `mutation_db`. |
| `sample_qc.compute_vaf_shape_score` | Score how well one sample's VAF pattern matches its purity. |
| `sample_qc.flag_unverified_samples` | Flag samples with no purity estimate available at all. |
| `sample_qc.load_copy_number_segments` | Load a copy-number segment table into a per-(sample, chromosome) lookup structure. |
| `selection.FactoredGeneLikelihood` | One gene's presence likelihood of ``u = log gamma``, on every draw. |
| `selection.FactoredGeneLikelihood.loglik` | ``(K, U)`` log-likelihood per draw on the grid (or its ``mask``). |
| `selection.FactoredGeneLikelihood.marginal` | ``log mean_d L_d(u)``: the likelihood with the draws averaged. |
| `selection.GeneSelectionLikelihoods` | Every gene's likelihood on the grid, as the selection table needs it. |
| `selection.GeneSelectionLikelihoods.marginal` | ``(G, U)`` marginal log-likelihoods on :func:`eb_grid`. |
| `selection.SpikeSlabPrior` | A fitted spike + slab prior on ``u = log gamma``. |
| `selection.SpikeSlabPrior.from_dict` |  |
| `selection.SpikeSlabPrior.interval` | The prior's own central interval of gamma. |
| `selection.SpikeSlabPrior.to_dict` | JSON-ready form (:meth:`from_dict` reads it back). |
| `selection.benjamini_hochberg` | Benjamini-Hochberg q-values; NaN p-values stay NaN, untested. |
| `selection.cut_posterior` | Posterior mass on the grid: draws' normalised posteriors, averaged. |
| `selection.eb_grid` | The empirical-Bayes prior's grid, a sub-grid of :func:`flat_grid`. |
| `selection.eb_mask` | Points of ``grid`` (default :func:`flat_grid`) inside the EB range. |
| `selection.fit_pooled_spike_slab` | One slab for several cohorts, a spike per cohort. |
| `selection.fit_spike_slab` | Fit one cohort's spike + slab prior by maximum marginal likelihood. |
| `selection.flat_grid` | The grid the likelihood is computed on (the flat prior's). |
| `selection.guard_triggers` | The guard's trigger: the prior's or its narrow component's interval. |
| `selection.log_absent_null` | ``log P(absent)`` per tumor at gamma = 1, as the null test uses it. |
| `selection.log_gamma_grid` | Grid of ``u = log gamma`` from ``lo`` to ``hi``, both included. |
| `selection.poisson_binomial_tail` | ``P(N < n)`` and ``P(N = n)``, mixed over the rows of ``log_absent``. |
| `selection.posterior_summary` | Central interval, median, mean and P(gamma > threshold) of a mass. |
| `selection.prior_excludes_one` | True when the prior's own central ``prob`` interval excludes gamma = 1. |
| `selection.r_posterior_nodes` | Quadrature nodes and weights for a Gamma(shape, rate) ``r_g``. |
| `selection.slab_basis` | Cubic B-spline basis on ``grid``, columns centred and unit-norm. |
| `selection.spike_excludes_one` | True when the narrow component's own central interval excludes one. |
| `selection.spike_slab_log_density` | ``log g`` on ``grid`` for given parameters (log scale centre and sd). |
| `selection.upper_mid_p` | Upper mid-p of ``n`` carriers: ``P(N > n) + P(N = n) / 2``. |
| `setup.build_full_germline_table` | Build the full gnomAD SNV table (every AF > 0). |
| `setup.download_cds_fasta` | Download Ensembl CDS FASTA file. |
| `setup.download_file` | Download a file from URL to destination. |
| `setup.download_gencode_gtf` | Download GENCODE GTF annotation file. |
| `setup.download_germline_mask` | Download the germline-mask table (15 MB, gnomAD AF > 5e-5). |
| `setup.download_hgnc` | Download HGNC complete set. |
| `setup.download_liftover_chain` | Download UCSC's hg38-to-hg19 liftOver chain (~1.2 MB). |
| `setup.download_wes_target_bed` | Download MC3's TCGA WES capture-kit target BED. |
| `setup.main` | Main entry point for command-line usage. |
| `signature_attribution.assign_signatures_per_gene_id` | Assign signature probabilities per gene ID. |
| `signature_attribution.compute_signature_effect_shares` | Share of a cohort's selection attributable to each signature. |
| `signature_attribution.compute_signature_probabilities` | Per-mutation P(sigma\|tau,j) for every signature. |
| `signature_attribution.compute_signature_probability_mass` | Per-mutation probability mass on a set of signatures. |
| `signature_decomposition.build_sbs96_matrix_from_mutation_db` | Write an SBS96 mutational-matrix file from a mutation database. |
| `signature_decomposition.compute_prevalence_overrides` | Signatures recurring often enough in a cohort's own fit to override the per-cancer-type table's tissue-absence exclusion. |
| `signature_decomposition.resolve_exclusion_list` | Resolve the final list of signatures to exclude for a cancer type. |
| `signature_decomposition.run_signature_decomposition` | Fits COSMIC mutational signatures to input mutation data. |
| `signature_decomposition.signature_decomposition` | Load or generate signature decomposition results. |
| `site_context.ContextFactors` | Per-process flank factors ``F_k(tau, c)``. |
| `site_context.ContextFactors.from_dict` |  |
| `site_context.ContextFactors.group_of` | The group index of a signature, or -1 (factor 1). |
| `site_context.ContextFactors.membership` | ``(S, K + 1)`` 0/1; the last column is the unit group. |
| `site_context.ContextFactors.to_dict` |  |
| `site_context.PlacementContext` | The placement mode, bound to one site table and its tumors. |
| `site_context.PlacementContext.group_opportunity` | ``(R, 17)`` group weights; zeros where there is no group. |
| `site_context.PlacementContext.multipliers` | Placement multipliers ``(R, J)`` of routes. |
| `site_context.as_factors` | A :class:`ContextFactors` from itself, a dict or a JSON path. |
| `site_context.cohort_context_table` | The opportunity mode: one ``(96, 16)`` table for a cohort. |
| `site_context.context_factor` | The context factor ``sum_k mix_{tau k} psi_k(tau, c)``. |
| `site_context.context_opportunity` | Summed weights by type and outer bases, ``(96, 17)``. |
| `site_context.context_table_at` | ``table[types, c]``, 1 where ``c`` is :data:`NO_CONTEXT`. |
| `site_context.group_mix` | Each signature group's share of each type, per tumor or pooled. |
| `site_context.normalised_flank` | ``psi_k(tau, c)``: each group's factors over their territory mean. |
| `site_context.placement_multipliers` | The placement mode: per route and tumor, ``(R, J)``. |
| `site_rates.AssaySiteRates` | The rates of a cohort whose tumors are on several assays. |
| `site_rates.AssaySiteRates.aggregate_rates` | :meth:`SiteRates.aggregate_rates`, summed over the assays. |
| `site_rates.AssaySiteRates.parts` | (assay name, its :class:`SiteRates`) per assay. |
| `site_rates.AssaySiteRates.snv_rates` | :meth:`SiteRates.snv_rates`, rows x all tumors. |
| `site_rates.AssaySiteRates.variant_rates` | :meth:`SiteRates.variant_rates`, variants x all tumors. |
| `site_rates.SiteRates` | Per-tumor rate of any variant, from a factored store. |
| `site_rates.SiteRates.aggregate_rates` | Summed rates per (gene, channel, label), without sites x tumors. |
| `site_rates.SiteRates.opportunity` | Weighted element counts per (gene, channel, label) and type. |
| `site_rates.SiteRates.parts` | (name, SiteRates) per assay; here, one. |
| `site_rates.SiteRates.variant_routes` | The routes of one change on the gene's chosen transcript. |
| `site_rates.SiteRates.verify_opportunity` | Raise unless the table rebuilds the model's channel tables. |
| `site_rates.SiteRates.with_placement` | These rates with the context factor in the placement mode. |
| `site_rates.SiteTable` | Every opportunity element of a gene universe, as flat arrays. |
| `site_rates.SiteTable.build` | Enumerate the elements of ``genes`` (stable Ensembl ids). |
| `site_rates.SiteTable.genomic_positions` | All flat coding positions at (chrom, pos): (query, flat). |
| `site_rates.SiteTable.position_rows` | Row of each flat position in the table, or -1. |
| `site_rates.SiteTable.weights` | ``upsilon_o`` before the mask, (n, 3), for ``rows``. |
| `site_rates.SiteTable.with_capture` | This table under another capture; the arrays are shared. |
| `site_rates.site_table_of` | The :class:`SiteTable` of a dataset, or of one assay's view. |
| `site_rates.tumor_type_rates` | The type rates ``mu_bar^j_tau / D_tau``, tumors x 96. |
| `site_weights.SiteWeights` | Per-cohort factors of the opportunity (see the module docstring). |
| `site_weights.SiteWeights.from_dict` |  |
| `site_weights.SiteWeights.has_context` |  |
| `site_weights.SiteWeights.has_cpg` |  |
| `site_weights.SiteWeights.label` | Short content hash, for cache paths. |
| `site_weights.SiteWeights.to_dict` |  |
| `site_weights.distance_bin` | Bin index into :data:`DIST_LABELS`. |
| `site_weights.fit_site_weights` | Fit distance and strand factors by raking two margins. |
| `site_weights.genomic_context5` | Outer bases of the pentamers at genomic positions, 0..15 or 16. |
| `site_weights.junction_distance` | Per flat position: distance along the CDS to the nearest junction. |
| `site_weights.position_bins` | Distance bin and strand orientation of every flat position. |
| `site_weights.position_context5` | Outer bases of every flat position's pentamer, 0..15 or 16. |
| `site_weights.position_gc` | Exon GC fraction minus 0.5 at every flat position (0 for padding). |
| `site_weights.splice_context5` | :func:`genomic_context5` of every ``models.splice`` row (cached). |
| `site_weights.variant_route_contexts` | Per variant, the outer bases of each route's pentamer. |
| `site_weights.variant_route_weights` | Per variant, the weight of each of its routes, in route order. |
| `site_weights.weights_at` | Weights of substitutions: bins/orientations per site, types per alt. |
| `spectrum_shrinkage.SpectrumShrinkage` | Per-cohort parameters of the floor and the hybrid. |
| `spectrum_shrinkage.SpectrumShrinkage.from_dict` |  |
| `spectrum_shrinkage.SpectrumShrinkage.label` |  |
| `spectrum_shrinkage.SpectrumShrinkage.to_dict` |  |
| `spectrum_shrinkage.arm_label` | Column name of one (kappa_1, kappa_2) arm in the score table. |
| `spectrum_shrinkage.floor_prior` | ``e = (1 - share) pool + share s^floor`` over the types. |
| `spectrum_shrinkage.floored_spectra` | ``q = (n p + kappa_1 e) / (n + kappa_1)``, row by row. |
| `spectrum_shrinkage.hybrid_spectra` | ``pi = (n_jt + kappa_2 q) / (n_j. + kappa_2)``, row by row. |
| `spectrum_shrinkage.parse_arm_label` | Inverse of :func:`arm_label`. |
| `spectrum_shrinkage.passenger_calls` | In-universe calls outside every non-passenger gene. |
| `spectrum_shrinkage.shrink_spectra` | Shrunk spectra (tumors x types) for one cohort. |
| `spectrum_shrinkage.type_counts` | Tumors x types call counts (zeros where a tumor has none). |
| `split_maf_file.split_maf_file` | Split a multi-sample MAF file into per-sample MAF files. |
| `tcga_sample_selection.catalog_maf_files` | Map every ``.maf`` file in a directory to its parsed barcode. |
| `tcga_sample_selection.filter_by_sample_type` | Keep only MAF files whose sample-type code is in *keep_sample_types*. |
| `tcga_sample_selection.read_maf_tumor_sample_barcode` | Read the ``Tumor_Sample_Barcode`` of a single-aliquot MAF file. |
| `tcga_sample_selection.select_one_per_case` | Resolve cases with more than one remaining MAF file. |
| `update_gene_names.update_genes_with_gene_set` | Update gene symbols and Ensembl IDs to HGNC-approved values. |
| `utils.project_onto_pca` | Place raw covariate rows in an existing PCA basis. |
| `utils.run_pca_on_covariates` | Compute PCA over gene-level covariates and return scores. |
| `utils.run_riemannian_stats_on_covariates` | Compute Riemannian principal components over gene covariates. |
| `variant_annotation.annotate_variants_with_types` | Annotate variants with the mutational type(s). |
| `variant_annotation.build_arrays_for_cov_effect_estimation` | Build arrays needed for the covariates regression. |
| `variant_annotation.extract_variants_from_db` | Extract unique variants from a mutation DataFrame. |
| `variant_annotation.load_or_generate_variants_with_types` | Load or generate a DataFrame of unique variants with type annotations. |
| `wes_target.compute_wes_target_gene_ids` | Compute the WES-target gene-ID set from source files. |
| `wes_target.get_wes_target_gene_ids` | Cached WES-target gene-ID set, downloading source files if needed. |
