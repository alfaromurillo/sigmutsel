# Selection in head and neck cancer: a first-week tutorial

A mutation that is common in tumors is not necessarily one that
helps tumors grow. It may simply be a mutation that *happens* often:
some DNA sites mutate tens of times faster than others, because of
their sequence context, the mutational processes active in the
tumor (tobacco smoke, the APOBEC family of DNA-editing enzymes,
ultraviolet light, ...)
and where the gene sits in the genome. **sigmutsel** separates the
two. For every tumor it estimates how often each site would mutate
with no selection (the **neutral mutation rate**, `mu`), and then
how many times more often a gene or a variant is found mutated than
that rate predicts (the **selection coefficient**, `gamma`):
`gamma = 1` is neutral, `gamma = 10` means the mutation is found in
tumors ten times as often as its rate alone would put it there.

In this tutorial you will:

1. download a public cohort of The Cancer Genome Atlas (TCGA): HNSC,
   head and neck squamous cell carcinoma, 410 tumors; and choose the
   right files;
2. build a mutation dataset, with its quality control and the
   mutational-signature decomposition of each tumor;
3. fit the neutral rate model, with gene covariates from Zenodo;
4. read rates per tumor, per gene and per variant;
5. estimate selection for genes and variants, with intervals;
6. make the selection call for every gene (is it selected?) and
   estimate its strength (how strongly?);
7. work through two small research projects: **(a)** a frequent
   mutation that is mostly mutation rate against a rarer one that is
   strongly selected, and **(b)** comparing the strength of selection
   between two groups of tumors (smokers and never-smokers).

Each step says what it does in plain words, what to check, and the
mistakes people make. You need no prior experience with the
package; some Python and pandas, and the idea of a probability
distribution, are enough.

## Before you start

**Install** (Python 3.12, Linux or macOS; a fresh virtual
environment is strongly recommended):

```sh
python3.12 -m venv sigmutsel_env
source sigmutsel_env/bin/activate
pip install git+https://github.com/alfaromurillo/sigmutsel.git \
    git+https://github.com/alfaromurillo/sigmutselcovs.git
sigmutsel-setup        # reference data + the GRCh38 genome, once
sigmutsel-check-data   # every line should say "Found"
```

`sigmutsel-setup` downloads about 300 MB of reference files and
installs SigProfilerMatrixGenerator's GRCh38 genome (about 3 GB; on
our connection it took 18 minutes, most of it the genome). **Do not
skip the genome**: without it nothing fails, but every step that
reads the genome falls back to defaults and the rates come out
different (you would see a `RuntimeWarning: No installed GRCh38
genome`).

To run this file as a notebook, open it in Jupyter with `jupytext`
installed (`pip install jupytext notebook`) or in VS Code; or run it
as a plain script, `python selection_in_hnsc.py`.

**What it costs.** Measured on a desktop (16 cores, of which the fit
uses 4; 100 Mb/s connection). The outputs shown on this page come
from that run.

| step | time | peak memory | disk |
|---|---|---|---|
| download and choose the mutation files (MAFs) | 1 min | small | 180 MB |
| build the dataset (quality control, signatures, opportunities) | 6 min | 12 GB | 1 GB |
| calibration steps (sample quality control, spectra, site weights) | 16 min | 16-20 GB | |
| fit the rate model (Hamiltonian Monte Carlo, 4 chains) | 47 min (300 components; 4 min on a GPU) | under 10 GB | 0.6 GB |
| gammas (each) | 4-8 s | | |
| selection table, every gene | 5 min (4 processes) | | |

**Memory.** One step, the site weights, needs 16-20 GB whatever the
cohort (it holds every coding site of the exome at once; 16 GB in the
run shown here, 20 GB in a run as a script); the rest fits in 12 GB.
On a 16 GB laptop set `SITE_WEIGHTS = False` in step 3: everything
runs, and the rates lose the correction for how detection varies along
each exon (fine for learning the package; use a larger machine for
results you report). The first sample quality control (QC) parses
TCGA's copy-number file (3 minutes); later runs reuse it.

## 0. Set up the session

Three things before any analysis: the sampler's chains, the
libraries, and where the tutorial writes its files.

### Chains, cores and `XLA_FLAGS`

The slow step, the fit of the rate model (section 5), samples the
model's posterior with Hamiltonian Monte Carlo (NUTS, the No-U-Turn
Sampler) in **four chains**: four independent runs from different
starting points, 1,000 draws each, pooled into the 4,000 draws the
selection estimates use. Four rather than one is how convergence is
checked: if the chains agree, they found the same posterior (the R-hat
statistic of section 5, which compares the spread between chains with
the spread within them, measures it); if they disagree, none of them
can be trusted.

The chains can run **at the same time**, one per processor (CPU) core,
which makes the fit several times faster. The sampler (NumPyro, built
on JAX, a library for numerical computing) does that only if JAX sees
four *devices*. A device here is not hardware: on a CPU, JAX shows one
device unless it is told to split the CPU into several, which the
environment variable `XLA_FLAGS` below does (XLA is the compiler JAX
runs on). It must be set **before anything imports JAX** (sigmutsel
imports it), so it is the first thing the tutorial runs.

- **A machine with 4 or more cores** (most laptops and desktops
  today): keep the setting; the chains run in parallel.
- **Fewer than 4 cores**: the four devices share the cores you have,
  so running them together gains little over one after another;
  the results are the same.
- **Without the setting** (or with fewer devices than chains), JAX
  shows one device, NumPyro warns "There are not enough devices to
  run parallel chains: expected 4 but got 1. Chains will be drawn
  sequentially." and runs the four chains one after another. **The
  posterior is the same**; the fit takes several times longer (in
  one measured case, a small cohort's fit, 639 s instead of 66 s).
  Do not ask for 2 devices: anything below 4 falls back to one at a
  time.

The cell below sets it and prints how many cores this Python may
use; section 5 checks, just before the fit, that JAX sees the four
devices. (JAX is not imported here: imported this early, it warns
when the signature step starts worker processes.)

> **If you have an NVIDIA GPU (graphics processor; optional).** Only
> the fit of section 5 gains from it; everything else stays on the
> CPU, and no call changes. Three things, before sigmutsel is
> imported:
>
> 1. install JAX's build for CUDA (NVIDIA's GPU software) at the
>    version sigmutsel is tested with: `pip install
>    "jax[cuda12]==0.11.0"`;
> 2. run the four chains together on the one GPU: add
>    `os.environ["SIGMUTSEL_CHAIN_METHOD"] = "vectorized"` to the cell
>    below (left at its default, `"parallel"`, NumPyro warns that
>    there are not enough devices and runs the chains one after
>    another);
> 3. do not set `JAX_PLATFORMS=cpu`. `XLA_FLAGS` then does not matter:
>    it splits only the CPU.
>
> `jax.devices()` should then list a CUDA device (section 5's count
> shows 1, which is right here). For this cohort, on one NVIDIA A100,
> the sampling took 4 minutes instead of 47 on four CPU cores (11x),
> with the same posterior; the per-variant rates computed at the end
> of the step stay on the CPU.


```python
import os

# Four JAX devices, one per chain: before anything imports JAX.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

# The cores this Python may use (Linux; elsewhere, all the machine's).
cores = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else os.cpu_count()
)
print(f"CPU cores: {cores}")
```

    CPU cores: 8


### Import the libraries

sigmutsel prints only warnings, the messages worth reading. To
follow the long steps as they run, add
`logging.basicConfig(level=logging.INFO)` (after `import logging`)
to this cell: every step then reports what it is doing.


```python
from pathlib import Path  # file paths that work on every system

import matplotlib.pyplot as plt  # figures
import numpy as np  # arrays and numerical functions
import pandas as pd  # tables (DataFrames): every result is one

import sigmutsel
from sigmutsel import Model, MutationDataset  # the two main objects

print("sigmutsel", sigmutsel.__version__)
```

    sigmutsel 0.1.1.post234


### Where the tutorial writes its files

Everything the tutorial downloads and computes goes in one
directory, about 2 GB: the MAF files, the dataset, the fitted model.
A rerun finds them there and skips what is done.


```python
PROJECT = "HNSC"  # the TCGA cohort

# A relative path like "hnsc" is relative to the directory you run
# from. For a fixed place, Path.home() / "sigmutsel_hnsc" works
# everywhere:
#   Linux    /home/<you>/sigmutsel_hnsc
#   macOS    /Users/<you>/sigmutsel_hnsc
#   Windows  C:\Users\<you>\sigmutsel_hnsc
# or name any directory, e.g. Path("/data/sigmutsel_hnsc"), or on
# Windows Path("D:/sigmutsel_hnsc") (forward slashes work there too).
# sigmutsel is tested on Linux and macOS; on Windows run it in WSL,
# where paths are Linux paths.
WORK = Path("hnsc")
WORK.mkdir(exist_ok=True)
```

## 1. Download the mutation calls and choose the files

The GDC (Genomic Data Commons of the US National Cancer Institute)
publishes TCGA's somatic mutation calls as one MAF file (Mutation
Annotation Format: a tab-separated table of mutations)
per sequenced *aliquot*. `download_project_maf_files` finds a
project's open-access MAFs, downloads them over HTTPS and unpacks
them into one flat directory. It needs no account and no extra
program.


```python
from sigmutsel.download_tcga_data import download_project_maf_files

raw_maf_dir = WORK / "all_maf_files_raw"
mafs = download_project_maf_files(PROJECT, raw_maf_dir)
print(len(mafs), "MAF files")
```

    511 MAF files


**Rerunning is safe.** Running a cell again (a *rerun*: after a
lost connection, a restarted kernel, or the next day) does not
repeat the work: the files already in `raw_maf_dir`, checked against
the GDC's sizes, are skipped, and only what is missing is fetched.
Calling the download a second time shows it: it lists the GDC's
files again, finds them all on disk and returns in seconds. Most
steps below behave the same way: they look in `WORK` first.


```python
import time

start = time.time()
mafs = download_project_maf_files(PROJECT, raw_maf_dir)  # a rerun
print(f"{len(mafs)} MAF files in {time.time() - start:.0f} s")
```

    511 MAF files in 2 s


**Choose the files before anything reads them.** One aliquot per
file means the directory also holds recurrent and metastatic
samples, several aliquots of the same patient and patients treated
before the sample was taken. Counted as separate tumors, a patient
with four aliquots makes each of their mutations look recurrent.
`select_tcga_maf_files` keeps primary tumors (TCGA sample types 01
and 03), the newest aliquot of each patient, and drops patients the
GDC records as treated before collection. **Its defaults keep every
aliquot**, so pass the options below.


```python
from sigmutsel.tcga_sample_selection import select_tcga_maf_files

maf_dir = WORK / "all_maf_files"  # the MAFs the dataset reads
# The first time, this directory does not exist yet: the files are
# chosen and it is created. On a rerun it exists and is left as it is.
if not maf_dir.is_dir():
    chosen = select_tcga_maf_files(
        raw_maf_dir,
        duplicate_policy="newest",
        exclude_prior_treatment=True,
        project=PROJECT,
    )
    maf_dir.mkdir()
    for f in chosen:  # links, so the downloads stay where they are
        (maf_dir / f.name).symlink_to(f.resolve())
print(len(list(maf_dir.glob("*.maf"))), "files chosen")
```

    410 files chosen


## 2. Build the dataset

A `MutationDataset` turns the MAFs into everything the model needs.
Four things happen here, and they take about six minutes:

- **Call quality control** (`qc_mode=True`): calls in repeat regions
  (where reads map badly), duplicated calls, and multi-base events
  (doublets, kept aside because the model is about single-base
  substitutions) are removed, and calls on common germline alleles
  (frequency above `1.5e-4` in gnomAD, the Genome Aggregation
  Database) are masked, because the GDC's
  own filter already removes most calls there.
- **Mutational signatures**: each tumor's mutations are decomposed
  into the single-base substitution (SBS) signatures of COSMIC, the
  Catalogue Of Somatic Mutations In Cancer (SBS4 tobacco, SBS2 and
  SBS13 APOBEC, SBS1 and SBS5 clock-like, ...). The decomposition runs
  twice: first unrestricted, to find sequencing-artifact mutations and
  treatment-exposed tumors; then restricted to the signatures known in
  this cancer type (`exclude_signature_subgroups=PROJECT`).
- **The channel universe**: each call is placed on its gene's
  transcript and classified as synonymous, missense, nonsense or
  splice, and each gene's **opportunity** (how many sites could
  mutate to each consequence, in each trinucleotide context) is
  counted.
- **The gene universe** (`gene_universe="wes_target"`, the default,
  written out below so you see it): every gene of the exome
  capture, mutated in this cohort or not.

The built dataset is saved, so a rerun of this cell loads it in
seconds.


```python
from sigmutsel.qc import load_repeat_intervals
from sigmutsel.setup import download_repeatmasker_bed

dataset_dir = WORK / f"{PROJECT}_dataset"
if dataset_dir.is_dir():
    dataset = MutationDataset.load_dataset(dataset_dir)
else:
    dataset = MutationDataset(maf_dir, signature_class="SBS")
    dataset.generate_mutation_db(
        qc_mode=True,
        qc_kwargs={
            "repeat_intervals": load_repeat_intervals(
                download_repeatmasker_bed("hg38")
            )
        },
        germline_mask_af=1.5e-4,
    )
    dataset.run_two_pass_signature_decomposition(
        exclude_signature_subgroups=PROJECT,
        treatment_naive=True,
        genome_build="GRCh38",
        exome=True,
        cosmic_version=3.6,
        artifact_threshold=0.5,
        treatment_load_threshold=0.2,
        prevalence_override_min_fraction=0.15,
        min_burden_for_diagnostics=100,
    )
    # Keeps the calls the signature step cleaned (it rebuilds them
    # from the MAFs only if none are loaded).
    dataset.build_full_dataset(gene_universe="wes_target")
    dataset.save_dataset(dataset_dir)

print(
    dataset.n_samples,
    "tumors,",
    len(dataset.model_db),
    "modelled calls",
)
```

    392 tumors, 59453 modelled calls


**What to check.** The number of tumors is a little below the number
of files: tumors whose every call was removed by the QC, or that the
signature step flagged as treatment-exposed, are gone.
`dataset.model_db` is the table of calls the model counts (one row
per call, with its gene, consequence and context); `mutation_db`
also holds the calls outside coding territory, which the model does
not use. Always take calls from `model_db`.


```python
calls = dataset.model_db
calls[
    [
        "Tumor_Sample_Barcode",
        "gene",
        "variant",
        "Variant_Classification",
    ]
].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tumor_Sample_Barcode</th>
      <th>gene</th>
      <th>variant</th>
      <th>Variant_Classification</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TCGA-F7-A50I-01A-11D-A28R-08</td>
      <td>DISP3</td>
      <td>DISP3 p.V814L</td>
      <td>Missense_Mutation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TCGA-F7-A50I-01A-11D-A28R-08</td>
      <td>CSMD2</td>
      <td>CSMD2 p.I900T</td>
      <td>Missense_Mutation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TCGA-F7-A50I-01A-11D-A28R-08</td>
      <td>DNALI1</td>
      <td>DNALI1 p.K32=</td>
      <td>Silent</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TCGA-F7-A50I-01A-11D-A28R-08</td>
      <td>HIVEP3</td>
      <td>HIVEP3 p.P1225L</td>
      <td>Missense_Mutation</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TCGA-F7-A50I-01A-11D-A28R-08</td>
      <td>ADGRL4</td>
      <td>ADGRL4 p.H498Y</td>
      <td>Missense_Mutation</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What each quality-control step removed, summed over the tumors (the
# dataset keeps a ledger of it: one row per tumor and step).
STEP_LABELS = {
    "not_snv": "not a single-base substitution (indels)",
    "invalid_chromosome": "invalid chromosome",
    "invalid_allele": "invalid allele",
    "invalid_context": "invalid context",
    "duplicate_record": "duplicate record",
    "germline_variant_site": "at a common germline site",
    "merged_into_dbs_variant": "part of a doublet (set aside)",
    "merged_with_nearby_variant": "merged with a nearby call",
    "repetitive_region": "in a repeat region",
    "compact_failed": "no gene",
    "treatment_signature_sample": "tumor exposed to treatment",
    "artifact_signature_mutation": "sequencing-artifact signature",
}
ledger = dataset.qc_ledger
removed = (
    ledger[
        (ledger["level"] == "calls") & (ledger["step"] != "maf_rows")
    ]
    .groupby(["order", "step"])["n"]
    .sum()
    .droplevel("order")  # sorted in the order the steps run
)
removed = removed[removed > 0]
total = ledger.loc[ledger["step"] == "maf_rows", "n"].sum()

fig, ax = plt.subplots(figsize=(8, 0.35 * len(removed) + 1))
labels = [
    STEP_LABELS.get(step, step.replace("_", " "))
    for step in removed.index
]
ax.barh(labels, removed.to_numpy(), color="C0")
ax.invert_yaxis()  # the first step on top
ax.set_xscale("log")
ax.set_xlabel("calls removed or set aside, all tumors")
fig.tight_layout()
plt.show()
print(f"{total:,} calls in the MAF files, {len(calls):,} modelled")
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_17_0.png)
    


    69,489 calls in the MAF files, 59,453 modelled


**Reading the figure.** One bar per quality-control step, in the order
the steps run, on a logarithmic axis: what each removed from the MAF
files' calls, over all tumors. Calls that are not single-base
substitutions (insertions and deletions) are not modelled. Calls that
are part of a doublet are set aside, not discarded: the selection
estimates use them later. The last two bars come from the signature
step: every call of a tumor it found treatment-exposed, and the calls
it attributed to a sequencing-artifact signature. A step that removes
a large share in your own data is the first thing to look into.

## 3. Calibration steps (once per cohort)

Five more steps finish the dataset. They are separate because each
needs data or choices of its own; all are saved with the dataset.

**Sample QC.** A tumor sample with little tumor in it (low purity),
or whose variant allele fractions do not look like a tumor's, gives
too few calls and would look like a low-rate tumor. Such tumors are
flagged, from TCGA's purity and copy-number calls (made with the
ABSOLUTE method; public,
downloaded once), and left out of every fit below through
`excluded_samples`. This step is TCGA-specific: for other data, use
your own purity estimates or skip it.


```python
from sigmutsel.sample_qc import (
    combine_sample_flags,
    flag_low_purity_samples,
    flag_vaf_shape_samples,
    load_copy_number_segments_from_file,
)
from sigmutsel.setup import (
    download_tcga_absolute_purity,
    download_tcga_absolute_segments,
)

purity = pd.read_csv(download_tcga_absolute_purity(), sep="\t")
segments = load_copy_number_segments_from_file(
    download_tcga_absolute_segments()
)
tumors = dataset.mutation_db["Tumor_Sample_Barcode"].unique()
# The purity table uses shorter barcodes than the MAFs.
short = pd.Series(tumors, index=tumors).str.slice(
    0, len(purity["array"].iloc[0])
)
low_purity = (
    short.map(flag_low_purity_samples(purity))
    .fillna(False)
    .astype(bool)
    .rename("low_purity")
)
vaf_shape = flag_vaf_shape_samples(
    dataset.mutation_db, purity, segments
).reindex(tumors, fill_value=False)
dataset.sample_qc_flags = combine_sample_flags(low_purity, vaf_shape)
EXCLUDED = set(dataset.sample_qc_flags[dataset.sample_qc_flags].index)
print(f"{len(EXCLUDED)} of {len(tumors)} tumors flagged and left out")
```

    15.3% of samples flagged (60/392), above the 5% warn_threshold -- dropping this many samples may be discarding too much of the cohort; consider inverse-variance weighting instead.


    60 of 392 tumors flagged and left out



```python
# Purity against burden, one point per tumor, the flagged ones in red.
calls_per_tumor = dataset.mutation_db.groupby(
    "Tumor_Sample_Barcode"
).size()
tumor_purity = short.map(
    purity.drop_duplicates("array").set_index("array")["purity"]
)
flagged = dataset.sample_qc_flags.reindex(tumors, fill_value=False)
fig, ax = plt.subplots(figsize=(5.5, 4))
for keep, color, label in (
    (True, "C0", "kept"),
    (False, "C3", "flagged"),
):
    group = tumors[(~flagged.to_numpy()) == keep]
    ax.scatter(
        tumor_purity.reindex(group),
        calls_per_tumor.reindex(group),
        s=10 if keep else 16,
        alpha=0.6 if keep else 0.9,
        color=color,
        label=label,
    )
ax.axvline(0.30, color="0.6", lw=1, ls="--")  # the purity threshold
ax.set_yscale("log")
ax.set_xlabel("tumor purity (ABSOLUTE)")
ax.set_ylabel("calls per tumor")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_21_0.png)
    


**Reading the figure.** Each point is a tumor (those without a purity
estimate are not shown). Left of the dashed line, purity below 0.30:
so little of the sample is tumor that many of its mutations fall
below the caller's detection, and the tumor would pass for a slowly
mutating one. Red points right of the line are flagged by the shape
of their allele fractions instead (their calls do not look like a
tumor's, given its purity and copy number). Both kinds stay out of
every fit through `EXCLUDED`.

**The rest**, in one cell:

- repeat regions leave the *opportunity* too, not only the calls
  (a site where no call can be made must not count as a site that
  could mutate);
- each genomic position is assigned to one gene, where genes
  overlap;
- each tumor's mutational **spectrum** (the share of each of the 96
  trinucleotide substitution types) is estimated by shrinking its
  own counts toward the cohort's, by an amount chosen on held-out
  mutations (`select_kappas`); this matters most for tumors with few
  mutations;
- **site weights**: detection is lower near exon ends and differs
  by DNA strand, and the weights correct the opportunity for it,
  fitted on passenger genes;
- homozygous deletions and coding indels are recorded, so that a
  tumor that lost a gene, or hit it by an indel, is not counted as
  evidence that the gene is not selected.


```python
from sigmutsel.channel_universe import load_or_build_transcript_models
from sigmutsel.sample_qc import homozygously_deleted_genes
from sigmutsel.spectrum_shrinkage import (
    heldout_spectrum_scores,
    select_kappas,
)

if not (dataset.channel_universe or {}).get("exclude_repeats"):
    dataset.set_exclude_repeats(True)
if not (dataset.channel_universe or {}).get("exclude_overlaps"):
    dataset.set_exclude_overlaps(True)
if dataset.spectrum_shrinkage is None:
    signatures = dataset.signature_matrix
    if (
        "MutationType" in signatures.columns
    ):  # a freshly built dataset
        signatures = signatures.set_index("MutationType")
    scores = heldout_spectrum_scores(dataset.mutation_db, signatures)
    kept = scores.drop(index=scores.index.intersection(EXCLUDED))
    kept.attrs.update(scores.attrs)
    dataset.set_spectrum_shrinkage(select_kappas(kept))
SITE_WEIGHTS = True  # False on a 16 GB machine (see "Memory" above)
if SITE_WEIGHTS and dataset.site_weights is None:
    # A model only for its tumors' type rates, which the weights are
    # fitted against; it is not kept.
    Model(dataset).estimate_site_weights(excluded_samples=EXCLUDED)
if dataset.homdel_db is None:
    dataset.homdel_db = homozygously_deleted_genes(
        segments, load_or_build_transcript_models(), tumors
    )
if dataset.indel_db is None:
    dataset.build_indel_db()
dataset.save_dataset(dataset_dir, overwrite=True)
print("spectrum shrinkage:", dataset.spectrum_shrinkage.label())
```

    spectrum shrinkage: kappa_1 100, kappa_2 1000


## 4. The tumors: burden and signatures

Before fitting anything, look at the data. Each tumor's number of
calls (its **mutation burden**) and the share of them its signature
decomposition gives to APOBEC (SBS2 + SBS13) and to tobacco (SBS4):


```python
burden = calls.groupby("Tumor_Sample_Barcode").size()
exposures = dataset.sig_assignments.reindex(burden.index).fillna(0)
share = exposures.div(exposures.sum(axis=1), axis=0)
apobec = share.filter(regex=r"^SBS(2|13)$").sum(axis=1)
tobacco = share.get("SBS4", pd.Series(0.0, index=share.index))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, s, label in (
    (axes[0], apobec, "APOBEC (SBS2+SBS13)"),
    (axes[1], tobacco, "tobacco (SBS4)"),
):
    ax.scatter(s, burden, s=8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel(f"share of the tumor's calls: {label}")
axes[0].set_ylabel("calls per tumor")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_26_0.png)
    


**Reading the figure.** Each point is a tumor. Burdens span two orders
of magnitude (from a handful of calls to over a thousand), so a single
"cohort rate" would describe almost no tumor: the model gives each
tumor its own rate, from its own burden and spectrum. A group of
tumors gets most of its mutations from APOBEC (left, the points to the
right), another from tobacco (right). APOBEC prefers a C in a TCW
context (a T, the C, then W = A or T), and two of the most common
mutations of this cancer, PIK3CA E545K and E542K, are C>T changes in
exactly that context. Keep that in mind for project (a).

The package draws every signature at once, tumor by tumor
(`figures.plot_sig_decomposition_with_burden`):


```python
from sigmutsel.figures import plot_sig_decomposition_with_burden

# The package draws at the size of a slide; a higher resolution
# makes it readable here.
with plt.rc_context({"figure.dpi": 200}):
    fig = plot_sig_decomposition_with_burden(
        burden, dataset.sig_assignments, show=True
    )
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_29_0.png)
    


**Reading the figure.** One row per tumor, sorted by burden (left, on
a logarithmic axis; the lowest at the top). Each column of the
heatmap is a signature, labelled for the ten most active; the colour
is the share of the tumor's calls the decomposition gives it. Most
tumors are a mix of a few signatures: the clock-like SBS1 and SBS5
in most tumors, plus APOBEC or tobacco in some. Tumors with
few calls have noisy shares, the reason their rates lean on the
whole cohort's spectrum (the spectrum shrinkage of section 3).

## 5. Fit the neutral rate model

The rate of gene `g` in tumor `j` combines:

- the tumor's **spectrum and burden** (from the signatures);
- the gene's **opportunity** (its sites, by context and consequence,
  with the site weights);
- a **covariate** term `exp(c . x_g)`: genes that replicate late,
  sit in closed chromatin or are expressed less mutate faster. The
  vector `x_g` holds these properties of gene `g`, its
  **covariates** (below, summarized as principal components), and
  the vector `c`, the **coefficients**, says how much each one
  raises or lowers the rate; `c` is estimated in this step;
- a **channel factor**: non-synonymous sites get their own
  intercept;
- a per-gene **rate correction** `r_g`, learned from the gene's own
  synonymous (silent) mutations, which selection barely touches.

The covariates come ready-made from Zenodo through **sigmutselcovs**:
the principal components of about 5,000 per-gene measurements
(expression, histone marks, chromatin accessibility, replication
timing) for this tissue, plus 0/1 indicators for genes missing part
of them. The production model for HNSC uses the first 300
components (`fitted_n_components` in the record's manifest, chosen
by held-out cross-validation); a smaller number fits faster.


```python
import json

import sigmutselcovs

components = sigmutselcovs.fetch_covariate_matrix(PROJECT)
files = sigmutselcovs.fetch_covariate_artifacts(
    PROJECT, artifacts=("pca_indicators", "pca_manifest")
)
indicators = pd.read_parquet(files["pca_indicators"])
N_COMPONENTS = json.loads(Path(files["pca_manifest"]).read_text())[
    "fitted_n_components"
]
design = pd.concat(
    [components.iloc[:, :N_COMPONENTS], indicators], axis=1
)
genes = dataset.contexts_by_gene.index
print(
    f"{N_COMPONENTS} components; {genes.isin(design.index).sum()} of "
    f"{len(genes)} dataset genes have covariates"
)
```

    300 components; 19147 of 19147 dataset genes have covariates


Now the model. `assign_cov_matrix` takes the design as given (no
second principal component analysis). `compute_channel_base_mus`
computes each tumor's type rates and each gene's baseline rate on both
channels, the share of a tumor's mutations that falls in a gene
following the tumor's own spectrum. `estimate_channel_rg_cov_effects`
fits the covariate effects `c`, the channel intercept and the
dispersion `theta` of `r_g` by Hamiltonian Monte Carlo (NUTS, 4
chains, in parallel if JAX sees 4 devices, checked first);
`sample="full"` keeps the posterior draws, which the selection
estimates below need. This fit is the slow step; the fitted model is
saved. If it prints "There are not enough devices to run parallel
chains", the chains are running one after another: the result is the
same, only slower (see "Chains, cores and `XLA_FLAGS`").

The page you are reading was run on one GPU (the box in section 0),
so below it prints `JAX devices: 1`; on a CPU it should print 4.


```python
import jax  # the numerical library the sampler runs on

print(
    f"JAX devices: {jax.local_device_count()}"
)  # 4: chains in parallel

model_dir = WORK / f"model_nc{N_COMPONENTS}"
if model_dir.is_dir():
    model = Model.load_model(model_dir)
    # A saved model loads its own copy of the dataset: hand it the
    # tables the selection estimates read.
    for table in ("multi_base_db", "homdel_db", "indel_db"):
        setattr(model.dataset, table, getattr(dataset, table))
    model.dataset.sample_qc_flags = dataset.sample_qc_flags
else:
    model = Model(dataset)
    model.assign_cov_matrix(design)
    model.compute_channel_base_mus()
    model.estimate_channel_rg_cov_effects(
        excluded_samples=EXCLUDED, include_drivers=True, sample="full"
    )
    model.save_model(model_dir)
print(f"theta = {model.rg_theta:.2f}")
```

    JAX devices: 1


    theta = 8.53


**What to check.**

- **Convergence.** The posterior's R-hat: every parameter's should
  be below 1.01 (much above means the chains disagree; rerun with
  more draws, or check the data).
- **`theta`**, the inverse dispersion of `r_g`: large `theta` means
  the covariates leave little gene-to-gene variation unexplained.
- **Fit to held-out data.** The honest check of a rate model is how
  well it predicts mutation counts it has not seen. Here we compute
  the R² of passenger genes' non-silent counts using `r_g` from their
  silent counts only (`compute_r_g_for_evaluation`): the non-silent
  counts were not used to correct the gene's rate, so the model is
  predicting them, not fitting them.


```python
import arviz as az

posterior = model.cov_effects_posteriors
rhat = az.rhat(posterior)
print("max R-hat:", float(max(rhat[v].max() for v in rhat.data_vars)))
r2, per_gene = model.estimate_passenger_genes_r2(
    excluded_samples=EXCLUDED,
    target="non_silent_counts",
    gene_scaling=model.compute_r_g_for_evaluation(),
    return_per_gene=True,  # also the observed/expected pair per gene
)
print(f"R² of passenger genes' non-silent counts: {r2:.3f}")
```

    max R-hat: 1.0058080440396662


    R² of passenger genes' non-silent counts: 0.741



```python
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(
    per_gene["expected"],
    per_gene["observed"],
    s=3,
    alpha=0.3,
    color="C0",
    linewidths=0,
)
top = (
    max(per_gene["observed"].max(), per_gene["expected"].max()) * 1.5
)
ax.plot([0.01, top], [0.01, top], color="0.4", lw=1, ls="--")
ax.set_xscale("log")
# symlog: logarithmic above 1, linear below, so 0 counts can show.
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim(-0.3, top)
ax.set_xlabel("expected non-silent mutations (model)")
ax.set_ylabel("observed non-silent mutations")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_37_0.png)
    


**Reading the figure.** One point per passenger gene, the pairs the
R² above is computed from: its non-silent mutations in the kept
tumors against the number the model expects, with `r_g` from the
gene's silent mutations only. The dashed line is "observed =
expected". A good rate model puts the cloud along it at every scale;
the spread around it shrinks, relative to the count, as the counts
grow, because small counts are noisy. A cloud tilted away from the
line, or bent at one end, means the rates are off for a whole class
of genes. The few genes at the top right weigh most in the R², which
is why it should not be read alone.


```python
from scipy import stats

r_g = model.compute_r_g_for_evaluation()
theta = model.rg_theta
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.hist(
    r_g,
    bins=np.linspace(0, 4, 81),
    density=True,
    color="C0",
    alpha=0.6,
    label="genes' r_g",
)
grid = np.linspace(0.01, 4, 400)
ax.plot(
    grid,
    stats.gamma(a=theta, scale=1 / theta).pdf(grid),
    color="0.2",
    label=f"Gamma(theta, 1/theta), theta = {theta:.1f}",
)
ax.set_xlabel("r_g (gene's rate correction, from silent mutations)")
ax.set_ylabel("density")
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_39_0.png)
    


**Reading the figure.** The curve is the spread of `r_g` the model
allows before seeing a gene's silent mutations: centred at 1 (no
correction), and the narrower the larger `theta`. The bars are each
gene's estimate after seeing them. They gather more tightly at 1 than
the curve, because a gene with few silent mutations has little to
say and keeps a correction near 1; the genes far from 1 are those
whose many silent mutations show their rate really is higher or
lower than the covariates predict.

**Common mistakes at this step.**

- Using the *production* `r_g` (`compute_r_g_production`) to judge
  the fit: it is learned from both channels, so it partly predicts
  its own target and the R² looks better than it is.
- Reading R² as the measure of a rate model: a few heavily mutated
  genes carry most of its sum of squares. Compare models by held-out
  likelihood instead (the package's `cross_validation` module).
- Passing a different `excluded_samples` to later steps than to the
  fit: every per-gene sum is over the kept tumors.

## 6. Rates: per tumor, per gene, per variant

**Per gene.** `model.mu_gs` is a genes x tumors table: the expected
number of non-synonymous mutations of each gene in each tumor, from
the spectrum, opportunity, covariates and channel factor. It does
**not** include the gene's own correction `r_g`; multiply by it
yourself (`compute_r_g_for_evaluation` is the one learned from
silent mutations only, the one every selection estimate below uses).


```python
# A call's channel: syn (synonymous), esyn (synonymous at an exon end,
# which can act on splicing), mis (missense), non (nonsense), spl
# (essential splice). The non-silent channels are the last three.
NON_SILENT = ["mis", "non", "spl"]
r_g = model.compute_r_g_for_evaluation()
symbol = calls.drop_duplicates("ensembl_gene_id").set_index("gene")[
    "ensembl_gene_id"
]
kept = [t for t in model.mu_gs.columns if t not in EXCLUDED]
rows = []
for gene in ("TP53", "PIK3CA", "NOTCH1", "TTN", "MUC16"):
    gid = symbol[gene]
    rate = model.mu_gs.loc[gid, kept] * r_g.get(gid, 1.0)
    rows.append(
        {
            "gene": gene,
            "r_g": r_g.get(gid, np.nan),
            "median rate per tumor": rate.median(),
            "expected carriers": (1 - np.exp(-rate)).sum(),
            "observed carriers": calls.loc[
                (calls.gene == gene)
                & calls.channel.isin(NON_SILENT)
                & ~calls.Tumor_Sample_Barcode.isin(EXCLUDED),
                "Tumor_Sample_Barcode",
            ].nunique(),
        }
    )
pd.DataFrame(rows).set_index("gene").round(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>r_g</th>
      <th>median rate per tumor</th>
      <th>expected carriers</th>
      <th>observed carriers</th>
    </tr>
    <tr>
      <th>gene</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TP53</th>
      <td>1.198</td>
      <td>0.003</td>
      <td>1.302</td>
      <td>187</td>
    </tr>
    <tr>
      <th>PIK3CA</th>
      <td>1.015</td>
      <td>0.009</td>
      <td>4.264</td>
      <td>50</td>
    </tr>
    <tr>
      <th>NOTCH1</th>
      <td>1.182</td>
      <td>0.022</td>
      <td>10.601</td>
      <td>42</td>
    </tr>
    <tr>
      <th>TTN</th>
      <td>1.052</td>
      <td>0.451</td>
      <td>136.191</td>
      <td>130</td>
    </tr>
    <tr>
      <th>MUC16</th>
      <td>0.717</td>
      <td>0.204</td>
      <td>78.954</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>



**Reading the table.** "Expected carriers" is how many of the kept
tumors would carry at least one non-synonymous mutation in the gene if
it were neutral (a tumor with rate `mu` carries one with probability
`1 - exp(-mu)`). TTN and MUC16 are among the most mutated genes of
every cancer, and their observed counts are near what their length and
late replication predict: they are long, not (much) selected. TP53 is
mutated in about 190 of the 332 kept tumors where neutral evolution
would put a non-synonymous mutation in about one. This comparison of
observed with expected is the whole idea; `gamma` makes it precise.

**Per variant.** `SiteRates` gives the rate of any single-nucleotide
variant, protein change or splice change in every tumor, whether it
was observed or not, without building a variants x tumors table
(here every possible single-nucleotide variant of the exome, about
105 million, by 392 tumors: some 330 GB of memory). A protein
change's rate is the sum over every single-base route to it
(PIK3CA E545K has one; a change with two possible codons has two).
The *observable* rate is the one to compare with calls (a germline-
masked allele can mutate but cannot be called).


```python
from sigmutsel.site_rates import SiteRates

site_rates = SiteRates.from_model(model, r_g="evaluation")
VARIANTS = [
    "PIK3CA p.E545K",
    "PIK3CA p.E542K",
    "PIK3CA p.H1047R",
    "HRAS p.G12S",
    "TP53 p.R248Q",
    "TP53 p.R175H",
]
info, _, observable = site_rates.variant_rates(VARIANTS)
info[["channel", "n_routes", "routes"]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>channel</th>
      <th>n_routes</th>
      <th>routes</th>
    </tr>
    <tr>
      <th>variant</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PIK3CA p.E545K</th>
      <td>mis</td>
      <td>1</td>
      <td>T[C&gt;T]A</td>
    </tr>
    <tr>
      <th>PIK3CA p.E542K</th>
      <td>mis</td>
      <td>1</td>
      <td>T[C&gt;T]A</td>
    </tr>
    <tr>
      <th>PIK3CA p.H1047R</th>
      <td>mis</td>
      <td>1</td>
      <td>A[T&gt;C]G</td>
    </tr>
    <tr>
      <th>HRAS p.G12S</th>
      <td>mis</td>
      <td>1</td>
      <td>C[C&gt;T]G</td>
    </tr>
    <tr>
      <th>TP53 p.R248Q</th>
      <td>mis</td>
      <td>1</td>
      <td>C[C&gt;T]G</td>
    </tr>
    <tr>
      <th>TP53 p.R175H</th>
      <td>mis</td>
      <td>1</td>
      <td>G[C&gt;T]G</td>
    </tr>
  </tbody>
</table>
</div>



`observable` is variants x tumors. Its sum over the kept tumors is
the number of tumors expected to carry the variant by mutation
alone; the ratio of observed to expected is a first, crude look at
selection.


```python
carriers = (
    calls[~calls.Tumor_Sample_Barcode.isin(EXCLUDED)]
    .groupby("variant")["Tumor_Sample_Barcode"]
    .nunique()
)
kept_obs = observable.loc[:, observable.columns.isin(kept)]
rates = pd.DataFrame(
    {
        "observed carriers": carriers.reindex(VARIANTS).fillna(0),
        "expected carriers": kept_obs.sum(axis=1).to_numpy(),
        "median rate per tumor": kept_obs.median(axis=1).to_numpy(),
    },
    index=VARIANTS,
)
rates["observed / expected"] = (
    rates["observed carriers"] / rates["expected carriers"]
)
rates.style.format("{:.3g}")
```




<style type="text/css">
</style>
<table id="T_f79a9">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f79a9_level0_col0" class="col_heading level0 col0" >observed carriers</th>
      <th id="T_f79a9_level0_col1" class="col_heading level0 col1" >expected carriers</th>
      <th id="T_f79a9_level0_col2" class="col_heading level0 col2" >median rate per tumor</th>
      <th id="T_f79a9_level0_col3" class="col_heading level0 col3" >observed / expected</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f79a9_level0_row0" class="row_heading level0 row0" >PIK3CA p.E545K</th>
      <td id="T_f79a9_row0_col0" class="data row0 col0" >12</td>
      <td id="T_f79a9_row0_col1" class="data row0 col1" >0.00278</td>
      <td id="T_f79a9_row0_col2" class="data row0 col2" >4.55e-06</td>
      <td id="T_f79a9_row0_col3" class="data row0 col3" >4.32e+03</td>
    </tr>
    <tr>
      <th id="T_f79a9_level0_row1" class="row_heading level0 row1" >PIK3CA p.E542K</th>
      <td id="T_f79a9_row1_col0" class="data row1 col0" >13</td>
      <td id="T_f79a9_row1_col1" class="data row1 col1" >0.00288</td>
      <td id="T_f79a9_row1_col2" class="data row1 col2" >4.7e-06</td>
      <td id="T_f79a9_row1_col3" class="data row1 col3" >4.52e+03</td>
    </tr>
    <tr>
      <th id="T_f79a9_level0_row2" class="row_heading level0 row2" >PIK3CA p.H1047R</th>
      <td id="T_f79a9_row2_col0" class="data row2 col0" >6</td>
      <td id="T_f79a9_row2_col1" class="data row2 col1" >0.000667</td>
      <td id="T_f79a9_row2_col2" class="data row2 col2" >1.29e-06</td>
      <td id="T_f79a9_row2_col3" class="data row2 col3" >9e+03</td>
    </tr>
    <tr>
      <th id="T_f79a9_level0_row3" class="row_heading level0 row3" >HRAS p.G12S</th>
      <td id="T_f79a9_row3_col0" class="data row3 col0" >3</td>
      <td id="T_f79a9_row3_col1" class="data row3 col1" >0.00223</td>
      <td id="T_f79a9_row3_col2" class="data row3 col2" >5.13e-06</td>
      <td id="T_f79a9_row3_col3" class="data row3 col3" >1.34e+03</td>
    </tr>
    <tr>
      <th id="T_f79a9_level0_row4" class="row_heading level0 row4" >TP53 p.R248Q</th>
      <td id="T_f79a9_row4_col0" class="data row4 col0" >4</td>
      <td id="T_f79a9_row4_col1" class="data row4 col1" >0.00209</td>
      <td id="T_f79a9_row4_col2" class="data row4 col2" >4.81e-06</td>
      <td id="T_f79a9_row4_col3" class="data row4 col3" >1.91e+03</td>
    </tr>
    <tr>
      <th id="T_f79a9_level0_row5" class="row_heading level0 row5" >TP53 p.R175H</th>
      <td id="T_f79a9_row5_col0" class="data row5 col0" >6</td>
      <td id="T_f79a9_row5_col1" class="data row5 col1" >0.00237</td>
      <td id="T_f79a9_row5_col2" class="data row5 col2" >5.84e-06</td>
      <td id="T_f79a9_row5_col3" class="data row5 col3" >2.53e+03</td>
    </tr>
  </tbody>
</table>




**Reading the table.** A variant's rate is tiny (a few in a million
per tumor), so even the most frequent variants are expected in a small
fraction of one tumor across the whole cohort (0.003 tumors for
E545K). The ratio is thousands: these variants are among the most
strongly selected mutations in this cancer. The rates of variants of
the same gene differ: E545K and E542K are C>T changes in a TCA
context, APOBEC's favourite, and mutate about three and a half times faster than
H1047R (an A>G change). Project (a) is about exactly that difference.


```python
# Each tumor's rate for two PIK3CA hotspots, coloured by APOBEC's share
# of its calls (section 4).
kept_tumors = [t for t in kept if t in observable.columns]
x = observable.loc["PIK3CA p.H1047R", kept_tumors]
y = observable.loc["PIK3CA p.E545K", kept_tumors]
fig, ax = plt.subplots(figsize=(5.5, 4.5))
points = ax.scatter(
    x,
    y,
    c=apobec.reindex(kept_tumors),
    cmap="viridis",
    vmin=0,
    vmax=1,
    s=10,
)
fig.colorbar(points, ax=ax, label="APOBEC share of the tumor's calls")
low = min(x[x > 0].min(), y[y > 0].min()) * 0.8
high = max(x.max(), y.max()) * 1.2
ax.plot([low, high], [low, high], color="0.4", lw=1, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("rate of PIK3CA H1047R in the tumor")
ax.set_ylabel("rate of PIK3CA E545K in the tumor")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_49_0.png)
    


**Reading the figure.** One point per tumor; the dashed line is
"the same rate". The two hotspots are in the same gene, so the gene's
covariates and `r_g` are the same for both: the difference is the
tumor's spectrum alone. Rates span two orders of magnitude between
tumors, with the burden. Even tumors with little APOBEC (dark) sit
above the line, two to three times: C>T changes are common in every
tumor. APOBEC-rich ones (yellow) sit ten times above it or more,
because E545K is a C>T change in APOBEC's TCW context and H1047R an
A>G change it does not make. A cohort-wide rate would hide both
differences.

## 7. Selection coefficients for genes and variants

`estimate_gamma` estimates `gamma` for a gene (all its
non-synonymous sites together), a variant, or one consequence
channel of a gene. It compares, tumor by tumor, the probability
that a tumor carries the mutation given its rate and `gamma`, with
whether it does.

**The package's defaults are not the recommended settings** (they
are kept for reproducibility of older analyses). Pass these settings:


```python
model.estimate_gene_tumor_dispersion(excluded_samples=EXCLUDED)

GAMMA_SETTINGS = dict(
    excluded_samples=EXCLUDED,  # the QC-flagged tumors stay out
    use_mu_posterior=True,  # carry the rate model's uncertainty...
    mu_uncertainty="draws",  # ...exactly, over its posterior draws
    r_g_variant="evaluation",  # r_g from silent mutations only
    gene_tumor_dispersion="fitted",  # tumors differ beyond their rate
)
```

What each setting does:

- `use_mu_posterior=True, mu_uncertainty="draws"`: the rates are
  estimates, not known numbers. With these, gamma's interval
  averages over the rate model's posterior draws (the *cut*), so it
  is wide when the rates are uncertain. Without them the interval
  is too narrow.
- `r_g_variant="evaluation"`: the gene's rate correction from its
  silent mutations. The other one (`"production"`) has seen the
  non-silent mutations, so it would absorb part of the selection it
  is meant to measure.
- `gene_tumor_dispersion="fitted"`: mutations of a gene cluster in
  some tumors more than their rates predict. The dispersion `phi`
  is fitted on passenger genes (`estimate_gene_tumor_dispersion`,
  run above) and switched off where it does not improve held-out
  fit.

Tumors with a homozygous deletion of the gene, a coding indel in
it, or a doublet there are held out automatically (they met the
gene's pressure by an event the rates do not describe). For a
variant, tumors with *another* non-silent mutation in the same gene
are held out too.


```python
from sigmutsel.estimate_gammas import summarize_gamma


def gamma_row(item, **kwargs):
    """One gamma, summarized: posterior mean, 89% interval, carriers."""
    result = model.estimate_gamma(item, **GAMMA_SETTINGS, **kwargs)
    attrs = result.posterior.attrs
    return {
        "item": item
        + (f" ({kwargs['channel']})" if "channel" in kwargs else ""),
        **summarize_gamma(result),
        "carriers": attrs["n_tumors_with"],
        "tumors informative": attrs["n_tumors_informative"],
    }


GENES = [
    "TP53",
    "PIK3CA",
    "CDKN2A",
    "NOTCH1",
    "FAT1",
    "CASP8",
    "HRAS",
    "NSD1",
    "TTN",
    "MUC16",
]
gene_gammas = pd.DataFrame(
    [gamma_row(g, level="gene") for g in GENES]
).set_index("item")
gene_gammas.round(2)
```

    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  636  229      360     1000      3973      3883  1.00       3.7     4.3


    Gamma posterior summary:
           mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  18.9  7.1       10       32      4055      3872  1.00      0.11    0.12


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  230  119      100      430      3821      3906  1.00       1.9       3


    Gamma posterior summary:
          mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  7.3  2.66      4.1       12      4098      3655  1.00     0.042   0.049


    Gamma posterior summary:
          mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  4.3  1.41      2.5      6.8      3808      3911  1.00     0.023   0.025


    Gamma posterior summary:
          mean  sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma   31  14       15       57      3661      3511  1.00      0.23    0.29


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  45.5  21.5       21       85      3959      3873  1.00      0.34    0.41


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  8.4  3.9      4.2       16      3932      3822  1.00     0.062   0.093


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  1.463  0.265      1.1      1.9      4078      3841  1.00    0.0041  0.0031


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  1.127  0.284     0.74      1.6      3868      3755  1.00    0.0045  0.0039





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>lower</th>
      <th>upper</th>
      <th>upper_bound_only</th>
      <th>carriers</th>
      <th>tumors informative</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TP53</th>
      <td>636.11</td>
      <td>363.20</td>
      <td>1037.03</td>
      <td>False</td>
      <td>187</td>
      <td>104</td>
    </tr>
    <tr>
      <th>PIK3CA</th>
      <td>18.94</td>
      <td>10.14</td>
      <td>31.64</td>
      <td>False</td>
      <td>50</td>
      <td>280</td>
    </tr>
    <tr>
      <th>CDKN2A</th>
      <td>230.50</td>
      <td>102.87</td>
      <td>434.52</td>
      <td>False</td>
      <td>52</td>
      <td>197</td>
    </tr>
    <tr>
      <th>NOTCH1</th>
      <td>7.30</td>
      <td>4.07</td>
      <td>12.09</td>
      <td>False</td>
      <td>42</td>
      <td>276</td>
    </tr>
    <tr>
      <th>FAT1</th>
      <td>4.30</td>
      <td>2.52</td>
      <td>6.81</td>
      <td>False</td>
      <td>47</td>
      <td>258</td>
    </tr>
    <tr>
      <th>CASP8</th>
      <td>31.00</td>
      <td>15.33</td>
      <td>56.69</td>
      <td>False</td>
      <td>27</td>
      <td>302</td>
    </tr>
    <tr>
      <th>HRAS</th>
      <td>45.51</td>
      <td>20.68</td>
      <td>84.94</td>
      <td>False</td>
      <td>17</td>
      <td>313</td>
    </tr>
    <tr>
      <th>NSD1</th>
      <td>8.42</td>
      <td>4.16</td>
      <td>15.51</td>
      <td>False</td>
      <td>33</td>
      <td>284</td>
    </tr>
    <tr>
      <th>TTN</th>
      <td>1.46</td>
      <td>1.08</td>
      <td>1.92</td>
      <td>False</td>
      <td>130</td>
      <td>198</td>
    </tr>
    <tr>
      <th>MUC16</th>
      <td>1.13</td>
      <td>0.74</td>
      <td>1.63</td>
      <td>False</td>
      <td>62</td>
      <td>268</td>
    </tr>
  </tbody>
</table>
</div>



**Reading the table.** `mean`, `lower` and `upper` are the posterior
mean of gamma and its 89% interval; `tumors informative` are the
tumors without the mutation that entered the fit (the held-out ones
are not). TP53's gamma, about 600, says a TP53 non-silent mutation is
found in tumors about 600 times more often than its rate predicts: it
is the **strength of selection** on the gene (relative to neutral
evolution), comparable to other genes' gammas. MUC16 sits at 1; TTN at
about 1.5, with an interval just above 1: a modest excess, and TTN is
not called in section 8: among 19,000 genes some intervals exclude 1
by chance, and TTN, the longest gene, is always among the genes one
looks at first.

Two cautions. A *gene* gamma averages over all the gene's
non-synonymous sites: for an oncogene, a few selected hotspots
among hundreds of neutral sites give a modest gene gamma (PIK3CA,
HRAS), and its variant gammas are the effect sizes to report. And
one interval above 1 is not a selection call: across 19,000 genes
many neutral genes have intervals above 1 by chance. The call is
section 8.

**Per channel.** For a tumor suppressor, the predicted
loss-of-function channel (`"plof"`: nonsense and essential splice)
carries the selection, for an oncogene the missense channel
(`"mis"`); the synonymous channel (`"syn"`) is a
control, expected near 1.


```python
channel_gammas = pd.DataFrame(
    [
        gamma_row(g, level="gene", channel=ch)
        for g in ("TP53", "NOTCH1", "PIK3CA")
        for ch in ("mis", "plof", "syn")
    ]
).set_index("item")
channel_gammas.round(2)
```

    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  369  134      200      610      3778      3788  1.00       2.2     2.7


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  9e+02  4e+02      520     1600      3866      3803  1.00       5.8     7.9


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma   11  7.8      2.7       25      3551      3660  1.00      0.13    0.19


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  6.07  2.22      3.4       10      3898      3999  1.00     0.036   0.042


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  22.1  10.7      9.5       41      4096      3971  1.00      0.17    0.23


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  2.04  1.07     0.78      4.1      4054      3960  1.00     0.017   0.019


    Gamma posterior summary:
           mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  21.7  8.2       12       37      3909      3608  1.00      0.13    0.17


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  2.8  3.2     0.13      8.6      3747      3670  1.00     0.052   0.092


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  2.66  2.13     0.43      6.6      4286      4083  1.00     0.033   0.042





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>lower</th>
      <th>upper</th>
      <th>upper_bound_only</th>
      <th>carriers</th>
      <th>tumors informative</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TP53 (mis)</th>
      <td>369.14</td>
      <td>204.63</td>
      <td>607.51</td>
      <td>False</td>
      <td>143</td>
      <td>147</td>
    </tr>
    <tr>
      <th>TP53 (plof)</th>
      <td>947.62</td>
      <td>519.40</td>
      <td>1570.15</td>
      <td>False</td>
      <td>55</td>
      <td>229</td>
    </tr>
    <tr>
      <th>TP53 (syn)</th>
      <td>11.04</td>
      <td>2.69</td>
      <td>24.95</td>
      <td>False</td>
      <td>2</td>
      <td>330</td>
    </tr>
    <tr>
      <th>NOTCH1 (mis)</th>
      <td>6.07</td>
      <td>3.36</td>
      <td>10.01</td>
      <td>False</td>
      <td>34</td>
      <td>284</td>
    </tr>
    <tr>
      <th>NOTCH1 (plof)</th>
      <td>22.14</td>
      <td>9.47</td>
      <td>40.60</td>
      <td>False</td>
      <td>8</td>
      <td>308</td>
    </tr>
    <tr>
      <th>NOTCH1 (syn)</th>
      <td>2.04</td>
      <td>0.78</td>
      <td>4.07</td>
      <td>False</td>
      <td>5</td>
      <td>327</td>
    </tr>
    <tr>
      <th>PIK3CA (mis)</th>
      <td>21.71</td>
      <td>11.90</td>
      <td>36.58</td>
      <td>False</td>
      <td>50</td>
      <td>280</td>
    </tr>
    <tr>
      <th>PIK3CA (plof)</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.65</td>
      <td>True</td>
      <td>0</td>
      <td>330</td>
    </tr>
    <tr>
      <th>PIK3CA (syn)</th>
      <td>2.66</td>
      <td>0.43</td>
      <td>6.64</td>
      <td>False</td>
      <td>1</td>
      <td>331</td>
    </tr>
  </tbody>
</table>
</div>




```python
CHANNEL_COLORS = {"mis": "C0", "plof": "C3", "syn": "0.5"}
fig, ax = plt.subplots(figsize=(6, 3.6))
y = np.arange(len(channel_gammas))[::-1]
x_min = 0.1  # the left edge of the axis
for yi, (item, r) in zip(y, channel_gammas.iterrows()):
    color = CHANNEL_COLORS[item.split("(")[1].rstrip(")")]
    if r["upper_bound_only"]:
        # No carrier: only an upper bound, drawn as a dotted line from
        # the axis edge to the bound.
        ax.plot([x_min, r["upper"]], [yi, yi], ls=":", color=color)
        ax.plot(r["upper"], yi, marker="|", ms=8, color=color)
    else:
        ax.plot([r["lower"], r["upper"]], [yi, yi], color=color)
        ax.plot(r["mean"], yi, "o", ms=4, color=color)
# A thin line between genes (three channels each).
for yi in y[2::3][:-1]:
    ax.axhline(yi - 0.5, color="0.9", lw=0.8)
ax.axvline(1, color="0.6", lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlim(x_min, channel_gammas["upper"].max() * 2)
ax.set_yticks(y, channel_gammas.index)
ax.set_xlabel("gamma (posterior mean, 89% interval)")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_57_0.png)
    


**Reading the figure.** The same table, on a logarithmic axis, with
the channels in colour (missense blue, pLoF red, synonymous grey).
The red pLoF points sit far right for the tumor suppressors (TP53,
NOTCH1); for the oncogene PIK3CA the blue missense point is the
selected one and the pLoF channel is only a dotted upper bound. The
grey synonymous controls should cross the dashed line at 1 (TP53's
does not: see the table's reading above).

**Reading the table.** TP53 is selected on both channels (its hotspot
missense mutations and its truncations). NOTCH1, a tumor suppressor
in this cancer, is selected mostly on `plof` (about 22, against 6 on
missense). PIK3CA, an oncogene, is selected on missense, and its
`plof` channel has no carrier: it is reported as an **upper bound
only** (`mean` and `lower` are NaN), how large gamma can be, not
where it is. The synonymous channel is the control: neutral by
design, so its gamma should cover 1. NOTCH1's and PIK3CA's do. TP53's
does not (about 11, on two carriers): with two calls the interval is
wide, and the usual suspects are synonymous changes that act on
splicing, or chance; it is a flag to look at the two calls, not a
verdict on the rates. A synonymous
gamma well above 1 in *many* genes would mean the rates are off.

**Variants.**


```python
variant_gammas = pd.DataFrame(
    [gamma_row(v, level="variant") for v in VARIANTS]
).set_index("item")
variant_gammas.round(0)
```

    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  7e+03  3e+03     3000    13000      4454      4146  1.00        47      53


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  7e+03  4e+03     3300    14000      3864      3846  1.00        57      81


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  1e+04  8e+03     5300    28000      3964      3840  1.00       130     250


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  2e+03  1e+03      620     4800      3967      4023  1.00        23      36


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  9e+03  6e+03     3100    19000      4085      3918  1.00        89     150


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  1e+04  7e+03     4900    26000      3811      3727  1.00       120     230





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>lower</th>
      <th>upper</th>
      <th>upper_bound_only</th>
      <th>carriers</th>
      <th>tumors informative</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PIK3CA p.E545K</th>
      <td>6781.0</td>
      <td>3011.0</td>
      <td>12515.0</td>
      <td>False</td>
      <td>12</td>
      <td>280</td>
    </tr>
    <tr>
      <th>PIK3CA p.E542K</th>
      <td>7345.0</td>
      <td>3297.0</td>
      <td>13659.0</td>
      <td>False</td>
      <td>13</td>
      <td>280</td>
    </tr>
    <tr>
      <th>PIK3CA p.H1047R</th>
      <td>14122.0</td>
      <td>5300.0</td>
      <td>27768.0</td>
      <td>False</td>
      <td>6</td>
      <td>280</td>
    </tr>
    <tr>
      <th>HRAS p.G12S</th>
      <td>2199.0</td>
      <td>625.0</td>
      <td>4815.0</td>
      <td>False</td>
      <td>3</td>
      <td>313</td>
    </tr>
    <tr>
      <th>TP53 p.R248Q</th>
      <td>9480.0</td>
      <td>3114.0</td>
      <td>19231.0</td>
      <td>False</td>
      <td>4</td>
      <td>104</td>
    </tr>
    <tr>
      <th>TP53 p.R175H</th>
      <td>12836.0</td>
      <td>4931.0</td>
      <td>25751.0</td>
      <td>False</td>
      <td>6</td>
      <td>104</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(6, 4.5))
table = pd.concat([gene_gammas, variant_gammas])
y = np.arange(len(table))[::-1]
ax.errorbar(
    table["mean"],
    y,
    xerr=[
        table["mean"] - table["lower"],
        table["upper"] - table["mean"],
    ],
    fmt="o",
    ms=4,
    capsize=2,
)
ax.axvline(1, color="0.6", lw=1, ls="--")
ax.set_xscale("log")
ax.set_yticks(y, table.index)
ax.set_xlabel("gamma (posterior mean, 89% interval)")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_61_0.png)
    


**Reading the figure.** The horizontal axis is logarithmic: the dashed
line is neutral (gamma = 1). Genes sit between about 1 and several
hundred (TP53, CDKN2A); hotspot variants at thousands, because a
variant gamma measures selection on one site, not averaged over a
gene's hundreds of sites. Intervals are wide for variants with few
carriers (HRAS G12S has three).

## 8. Is a gene selected, and how strongly?

Two questions, two answers, in one table over every gene:

- **The call.** A test of the null hypothesis that the gene is
  neutral (`gamma = 1`): how surprising its number of carrier tumors
  is under its rates, integrating over the uncertainty of `r_g`
  (`p_mid`, the mid-p-value: the probability of more carriers than
  observed plus half that of exactly as many). Then the
  Benjamini-Hochberg correction over all genes gives `q`, the
  adjusted p-value: calling the genes with `q < 0.1` (`called`)
  keeps the expected share of false calls among them, the false
  discovery rate, at most 10%.
- **The strength.** Gamma's posterior under an **empirical-Bayes (EB)
  prior**, one estimated from the data: a spike near 1 (most genes are
  neutral) plus a slab for the selected ones, both fitted on this
  cohort's genes. This prior shrinks the gammas of genes with little
  information toward 1, so they can be compared across genes: the
  columns `eb_median` and `eb_lower` to `eb_upper`, its 89% interval;
  `eb_p_above` is the posterior probability that gamma exceeds 2. The
  estimate under a flat prior, one that favours no value of gamma
  (columns `flat_*`), is kept beside it, as an estimate only, never as
  a call.

It takes a few minutes (it runs every gene; `n_jobs` processes).


```python
selection = model.gene_selection_table(
    excluded_samples=EXCLUDED, fdr=0.1, n_jobs=4
)
gene_of = calls.drop_duplicates("ensembl_gene_id").set_index(
    "ensembl_gene_id"
)["gene"]
selection.insert(0, "gene", gene_of.reindex(selection.index))
called = selection[selection.called].sort_values("q")
print(f"{len(called)} of {len(selection)} genes called at q < 0.1")
called[
    [
        "gene",
        "n_tumors_with",
        "expected_with",
        "q",
        "eb_median",
        "eb_lower",
        "eb_upper",
        "flat_median",
    ]
].head(20).round(3)
```

    22 of 19147 genes called at q < 0.1





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gene</th>
      <th>n_tumors_with</th>
      <th>expected_with</th>
      <th>q</th>
      <th>eb_median</th>
      <th>eb_lower</th>
      <th>eb_upper</th>
      <th>flat_median</th>
    </tr>
    <tr>
      <th>gene_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ENSG00000064012</th>
      <td>CASP8</td>
      <td>27</td>
      <td>1.167</td>
      <td>0.000</td>
      <td>25.824</td>
      <td>13.801</td>
      <td>51.602</td>
      <td>27.932</td>
    </tr>
    <tr>
      <th>ENSG00000147889</th>
      <td>CDKN2A</td>
      <td>52</td>
      <td>0.448</td>
      <td>0.000</td>
      <td>196.668</td>
      <td>92.754</td>
      <td>397.152</td>
      <td>202.390</td>
    </tr>
    <tr>
      <th>ENSG00000141510</th>
      <td>TP53</td>
      <td>187</td>
      <td>0.989</td>
      <td>0.000</td>
      <td>574.524</td>
      <td>349.884</td>
      <td>1025.585</td>
      <td>580.555</td>
    </tr>
    <tr>
      <th>ENSG00000174775</th>
      <td>HRAS</td>
      <td>17</td>
      <td>0.480</td>
      <td>0.000</td>
      <td>35.929</td>
      <td>17.343</td>
      <td>76.285</td>
      <td>40.549</td>
    </tr>
    <tr>
      <th>ENSG00000121879</th>
      <td>PIK3CA</td>
      <td>50</td>
      <td>3.418</td>
      <td>0.000</td>
      <td>17.016</td>
      <td>9.848</td>
      <td>31.254</td>
      <td>17.757</td>
    </tr>
    <tr>
      <th>ENSG00000116044</th>
      <td>NFE2L2</td>
      <td>24</td>
      <td>1.433</td>
      <td>0.000</td>
      <td>17.878</td>
      <td>9.489</td>
      <td>36.693</td>
      <td>19.474</td>
    </tr>
    <tr>
      <th>ENSG00000148400</th>
      <td>NOTCH1</td>
      <td>42</td>
      <td>7.374</td>
      <td>0.000</td>
      <td>6.462</td>
      <td>3.914</td>
      <td>11.673</td>
      <td>6.701</td>
    </tr>
    <tr>
      <th>ENSG00000109670</th>
      <td>FBXW7</td>
      <td>22</td>
      <td>2.341</td>
      <td>0.000</td>
      <td>9.972</td>
      <td>5.508</td>
      <td>18.969</td>
      <td>10.820</td>
    </tr>
    <tr>
      <th>ENSG00000234745</th>
      <td>HLA-B</td>
      <td>11</td>
      <td>0.632</td>
      <td>0.000</td>
      <td>17.337</td>
      <td>7.923</td>
      <td>39.381</td>
      <td>20.816</td>
    </tr>
    <tr>
      <th>ENSG00000165671</th>
      <td>NSD1</td>
      <td>33</td>
      <td>5.579</td>
      <td>0.000</td>
      <td>7.173</td>
      <td>3.985</td>
      <td>13.895</td>
      <td>7.530</td>
    </tr>
    <tr>
      <th>ENSG00000206503</th>
      <td>HLA-A</td>
      <td>10</td>
      <td>0.612</td>
      <td>0.000</td>
      <td>16.198</td>
      <td>7.703</td>
      <td>35.625</td>
      <td>19.764</td>
    </tr>
    <tr>
      <th>ENSG00000083857</th>
      <td>FAT1</td>
      <td>47</td>
      <td>14.311</td>
      <td>0.001</td>
      <td>3.893</td>
      <td>2.332</td>
      <td>6.763</td>
      <td>3.977</td>
    </tr>
    <tr>
      <th>ENSG00000129474</th>
      <td>AJUBA</td>
      <td>11</td>
      <td>1.360</td>
      <td>0.003</td>
      <td>8.193</td>
      <td>2.939</td>
      <td>17.223</td>
      <td>9.532</td>
    </tr>
    <tr>
      <th>ENSG00000163513</th>
      <td>TGFBR2</td>
      <td>11</td>
      <td>1.374</td>
      <td>0.003</td>
      <td>7.962</td>
      <td>3.257</td>
      <td>16.773</td>
      <td>9.244</td>
    </tr>
    <tr>
      <th>ENSG00000079999</th>
      <td>KEAP1</td>
      <td>12</td>
      <td>1.747</td>
      <td>0.006</td>
      <td>6.844</td>
      <td>1.058</td>
      <td>13.925</td>
      <td>7.790</td>
    </tr>
    <tr>
      <th>ENSG00000136238</th>
      <td>RAC1</td>
      <td>7</td>
      <td>0.569</td>
      <td>0.007</td>
      <td>10.879</td>
      <td>1.039</td>
      <td>25.172</td>
      <td>14.418</td>
    </tr>
    <tr>
      <th>ENSG00000172819</th>
      <td>RARG</td>
      <td>9</td>
      <td>1.032</td>
      <td>0.008</td>
      <td>8.268</td>
      <td>1.037</td>
      <td>18.446</td>
      <td>10.061</td>
    </tr>
    <tr>
      <th>ENSG00000100393</th>
      <td>EP300</td>
      <td>21</td>
      <td>5.196</td>
      <td>0.044</td>
      <td>4.337</td>
      <td>1.034</td>
      <td>8.519</td>
      <td>4.584</td>
    </tr>
    <tr>
      <th>ENSG00000167548</th>
      <td>KMT2D</td>
      <td>35</td>
      <td>11.237</td>
      <td>0.044</td>
      <td>3.593</td>
      <td>1.055</td>
      <td>6.536</td>
      <td>3.689</td>
    </tr>
    <tr>
      <th>ENSG00000130538</th>
      <td>OR11H1</td>
      <td>6</td>
      <td>0.622</td>
      <td>0.065</td>
      <td>6.944</td>
      <td>1.008</td>
      <td>19.864</td>
      <td>11.720</td>
    </tr>
  </tbody>
</table>
</div>



**Reading the table.** Rows are genes by their Ensembl id (`ENSG...`),
with the symbol in `gene`. `n_tumors_with` are the carriers and
`expected_with` the carriers expected under neutrality. The genes at
the top are the known drivers of head and neck cancer (TP53, CDKN2A,
PIK3CA, CASP8, HRAS, NFE2L2, NOTCH1, FBXW7, NSD1, FAT1, HLA-A/B, ...).
Compare `eb_median` with `flat_median`: for a strongly selected gene
with many carriers they agree; for a gene with few carriers the EB
estimate is pulled toward 1. The pull is intended: across thousands of
genes, the flat estimates of genes with little data scatter widely by
chance, and the prior, learned from all genes, says how much.

**What to check.** Known passengers (TTN, MUC16) should not be called,
and are not. Near the bottom of the list there are genes you will not
find in a cancer gene list: OR11H1 (an olfactory receptor), and past
the 20 rows shown (`called.tail(3)`) NEK5 and CGB8 (one of a cluster
of near-identical copies). OR11H1 and CGB8 belong to multi-copy
families, where reads map to the wrong copy and calls pile up. A
called gene you do not recognise is a gene to examine (its calls, its
rates, its paralogs), not yet a discovery.


```python
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(
    selection.expected_with,
    selection.n_tumors_with,
    s=4,
    color="0.7",
    label="not called",
)
ax.scatter(
    called.expected_with,
    called.n_tumors_with,
    s=12,
    color="C3",
    label="called, q < 0.1",
)
# Label the five strongest calls; the rest are red dots.
for _, r in called.head(5).iterrows():
    ax.annotate(
        r.gene,
        (r.expected_with, r.n_tumors_with),
        fontsize=8,
        xytext=(4, 2),
        textcoords="offset points",
    )
top = max(selection.n_tumors_with.max(), 1) * 1.5
ax.plot([0.01, top], [0.01, top], color="0.4", lw=1, ls="--")
ax.set_xscale("log")
# symlog: logarithmic above 1, linear below, so 0 carriers can show.
ax.set_yscale("symlog", linthresh=1)
ax.set_ylim(-0.3, top)
ax.set_xlabel("expected carrier tumors (neutral)")
ax.set_ylabel("observed carrier tumors")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_66_0.png)
    


**Reading the figure.** One point per gene. The dashed line is
"observed = expected": a neutral gene scatters around it, more widely
at small counts. Called genes sit far above it. Long genes (to the
right) can have many carriers and still sit on the line.

**The EB prior, and what it does.** The table fitted the prior and
kept it on the model (`model.selection_prior`). The figure below
draws it (left) and, for every gene, sets the estimate under the
flat prior against the one under the EB prior (right).


```python
from sigmutsel.selection import SpikeSlabPrior

prior = SpikeSlabPrior.from_dict(model.selection_prior)
# The prior is a mass on a grid of log gamma; divided by the grid
# step it is a density per unit of log gamma. The spike is a normal
# curve with the fitted centre, sd and weight; the slab is the rest.
step = prior.grid[1] - prior.grid[0]
gamma = np.exp(prior.grid)
density = np.exp(prior.log_density) / step
spike = np.exp(
    -0.5 * ((prior.grid - prior.spike_centre) / prior.spike_sd) ** 2
)
spike *= prior.spike_weight / (spike.sum() * step)
slab = np.clip(density - spike, 1e-12, None)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
ax = axes[0]
ax.plot(gamma, slab, color="C0", label="slab: genes away from 1")
ax.plot(gamma, spike, color="0.3", label="spike: neutral genes")
ax.axvline(1, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log")
# Log density: the spike is thousands of times taller than the slab.
ax.set_yscale("log")
ax.set_ylim(1e-5, density.max() * 2)
ax.set_xlim(0.1, 1e4)
ax.set_xlabel("gamma")
ax.set_ylabel("prior density (per unit of log gamma)")
ax.legend(frameon=False, loc="upper right")
ax.text(
    0.03,
    0.04,
    f"spike: {prior.spike_weight:.1%} of the prior's mass",
    transform=ax.transAxes,
    fontsize=8,
)

ax = axes[1]
ax.scatter(
    selection.flat_median,
    selection.eb_median,
    s=4,
    color="0.7",
    label="not called",
)
ax.scatter(
    called.flat_median,
    called.eb_median,
    s=12,
    color="C3",
    label="called, q < 0.1",
)
lims = [0.1, max(selection.flat_median.max(), 10) * 1.5]
ax.plot(lims, lims, color="0.4", lw=1, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("gamma under the flat prior (median)")
ax.set_ylabel("gamma under the EB prior (median)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_68_0.png)
    


**Reading the figure.** Left: the prior puts almost all its mass in
a narrow spike at gamma close to 1, the neutral genes, and spreads
the rest thinly as a slab over values far from 1, mostly above. Its
shape is learned from all the cohort's genes at once. Right: one
point per gene; the dashed line is "the two estimates agree". Genes
with many carriers (the called ones, red) stay on it: their data
outweigh the prior. Genes with few carriers, whose flat estimate can
be anything from 0.1 to tens of thousands, form the horizontal band
at 1: the prior pulls them back, because the data cannot tell them
from the spike. That is what makes the EB estimates comparable
across genes, and why a large flat estimate on its own means little.

## 9. Project (a): frequency is not selection

Suppose a published paper ranked the mutations of head and neck
cancer by how often they occur, and called the most frequent the
most important. PIK3CA has three classic hotspots: E545K and E542K
in the helical domain, H1047R in the kinase domain. Which is the
most frequent, which mutates fastest, and which is the most strongly
selected?


```python
pik3ca = ["PIK3CA p.E545K", "PIK3CA p.E542K", "PIK3CA p.H1047R"]
project_a = rates.loc[
    pik3ca, ["observed carriers", "expected carriers"]
]
project_a = project_a.join(
    variant_gammas.loc[pik3ca, ["mean", "lower", "upper"]]
)
project_a.columns = [
    "carriers",
    "expected (neutral)",
    "gamma",
    "gamma lower",
    "gamma upper",
]
project_a.style.format("{:.3g}")
```




<style type="text/css">
</style>
<table id="T_35d03">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_35d03_level0_col0" class="col_heading level0 col0" >carriers</th>
      <th id="T_35d03_level0_col1" class="col_heading level0 col1" >expected (neutral)</th>
      <th id="T_35d03_level0_col2" class="col_heading level0 col2" >gamma</th>
      <th id="T_35d03_level0_col3" class="col_heading level0 col3" >gamma lower</th>
      <th id="T_35d03_level0_col4" class="col_heading level0 col4" >gamma upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_35d03_level0_row0" class="row_heading level0 row0" >PIK3CA p.E545K</th>
      <td id="T_35d03_row0_col0" class="data row0 col0" >12</td>
      <td id="T_35d03_row0_col1" class="data row0 col1" >0.00278</td>
      <td id="T_35d03_row0_col2" class="data row0 col2" >6.78e+03</td>
      <td id="T_35d03_row0_col3" class="data row0 col3" >3.01e+03</td>
      <td id="T_35d03_row0_col4" class="data row0 col4" >1.25e+04</td>
    </tr>
    <tr>
      <th id="T_35d03_level0_row1" class="row_heading level0 row1" >PIK3CA p.E542K</th>
      <td id="T_35d03_row1_col0" class="data row1 col0" >13</td>
      <td id="T_35d03_row1_col1" class="data row1 col1" >0.00288</td>
      <td id="T_35d03_row1_col2" class="data row1 col2" >7.34e+03</td>
      <td id="T_35d03_row1_col3" class="data row1 col3" >3.3e+03</td>
      <td id="T_35d03_row1_col4" class="data row1 col4" >1.37e+04</td>
    </tr>
    <tr>
      <th id="T_35d03_level0_row2" class="row_heading level0 row2" >PIK3CA p.H1047R</th>
      <td id="T_35d03_row2_col0" class="data row2 col0" >6</td>
      <td id="T_35d03_row2_col1" class="data row2 col1" >0.000667</td>
      <td id="T_35d03_row2_col2" class="data row2 col2" >1.41e+04</td>
      <td id="T_35d03_row2_col3" class="data row2 col3" >5.3e+03</td>
      <td id="T_35d03_row2_col4" class="data row2 col4" >2.78e+04</td>
    </tr>
  </tbody>
</table>





```python
fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
y = np.arange(len(pik3ca))[::-1]
labels = [v.split()[1] for v in pik3ca]
axes[0].barh(y, project_a["carriers"])
axes[0].set_xlabel("carrier tumors")
axes[1].barh(y, project_a["expected (neutral)"])
axes[1].set_xlabel("expected by mutation alone")
axes[2].errorbar(
    project_a["gamma"],
    y,
    xerr=[
        project_a["gamma"] - project_a["gamma lower"],
        project_a["gamma upper"] - project_a["gamma"],
    ],
    fmt="o",
    capsize=2,
)
axes[2].set_xlabel("gamma (89% interval)")
axes[0].set_yticks(y, labels)
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_72_0.png)
    


**Reading the figure.** The two helical-domain hotspots are the
frequent ones (12 and 13 carriers), and the kinase-domain H1047R has
half as many (6). But E545K and E542K mutate about three and a half times faster:
they are C>T changes in APOBEC's TCA context, and APOBEC is active in
many of these tumors (section 4). Once each is divided by its own
neutral rate, H1047R comes out the *most* strongly selected, about
twice the helical ones. Its interval overlaps theirs, so how sure is
that? From the posterior draws:


```python
draws = {
    v: np.asarray(model.gammas[v].posterior["gamma"]).ravel()
    for v in pik3ca
}
rng = np.random.default_rng(0)
for other in ("PIK3CA p.E545K", "PIK3CA p.E542K"):
    ratio = draws["PIK3CA p.H1047R"] / rng.permutation(draws[other])
    lo, hi = np.percentile(ratio, [5.5, 94.5])
    print(
        f"H1047R / {other.split()[1]}: {np.median(ratio):.2f} "
        f"[{lo:.2f}, {hi:.2f}], P(H1047R stronger) = "
        f"{(ratio > 1).mean():.2f}"
    )
```

    H1047R / p.E545K: 2.01 [0.65, 5.94], P(H1047R stronger) = 0.84
    H1047R / p.E542K: 1.89 [0.63, 5.40], P(H1047R stronger) = 0.82


The posterior favours stronger selection on H1047R, but with six
carriers it does not settle it: a frequency ranking puts the three
in the opposite order to the selection ranking, and the honest
statement is "H1047R is at least as strongly selected as the
helical hotspots despite being half as frequent". (The draws of two
gammas are paired at random: they share the rate model's
uncertainty, which this ignores, so the interval is if anything too
wide.) That frequency and strength of selection part ways in head
and neck cancer, and that APOBEC deaminates the E545K and E542K
sites, was shown with cancereffectsizeR's method by Cannataro et al.
(2019, *Oncogene* 38:3475-3487, doi:10.1038/s41388-018-0657-6): read
it beside this project.

**How to turn this into a letter.** State the claim you test (the
paper's ranking by frequency), show frequency, neutral expectation
and gamma side by side as here, for the paper's own variants, and
report gamma with its interval. Check that your conclusion does not
depend on one setting: refit with fewer covariate components, and
with `gene_tumor_dispersion=None`; a robust difference survives
both. And say what gamma is not: it is selection relative to the
variant's own neutral rate, under this rate model; a variant gamma
is conditional on the site-level rate model (sites recur somewhat
more often than it predicts), so read a difference of less than
about a factor of two between two variants with care.

## 10. Project (b): selection in two groups of tumors

Does selection on a gene differ between smokers and never-smokers?
A paper that compares *mutation frequencies* between the groups
cannot tell: smokers' tumors carry more mutations everywhere, so a
gene is mutated more often in them even with the same selection.
Gamma removes that, because each tumor's rate includes its own
burden and spectrum (tobacco's SBS4 included).

**Step 1: the groups**, from the GDC's clinical data (public, no
account): `tobacco_smoking_status` of each patient.


```python
import requests

response = requests.post(
    "https://api.gdc.cancer.gov/cases",
    json={
        "filters": {
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [f"TCGA-{PROJECT}"],
            },
        },
        "fields": "submitter_id,exposures.tobacco_smoking_status",
        "size": 2000,
    },
    timeout=60,
)
status = {
    case["submitter_id"]: (case.get("exposures") or [{}])[0].get(
        "tobacco_smoking_status"
    )
    for case in response.json()["data"]["hits"]
}
patient = pd.Series(kept, index=kept).str.slice(0, 12)
smoking = patient.map(status)
groups = {
    "never": set(smoking.index[smoking == "Lifelong Non-Smoker"]),
    "smoker": set(
        smoking.index[smoking.str.startswith("Current", na=False)]
    ),
}
print(
    {name: len(tumors) for name, tumors in groups.items()},
    "| status unknown:",
    int(
        (
            ~smoking.index.isin(groups["never"] | groups["smoker"])
        ).sum()
    ),
)
```

    {'never': 74, 'smoker': 250} | status unknown: 8


**Step 2: one rate model, two gammas.** The rate model above was
fitted on all tumors; it already gives every tumor its own rates.
A group's gamma is estimated on that group's tumors alone, by
excluding everyone else. **Do not fit a separate rate model per
group**: the covariate effects and `r_g` are properties of genes,
best estimated on all tumors, and a small group's own fit would
make its rates noisier, not more specific.


```python
def group_gamma(item, level, group, **kwargs):
    """Gamma of ``item`` on one group's tumors: its posterior draws."""
    others = set(kept) - groups[group]
    settings = {
        **GAMMA_SETTINGS,
        "excluded_samples": EXCLUDED | others,
    }
    result = model.estimate_gamma(
        item, level=level, store=False, **settings, **kwargs
    )
    return np.asarray(result.posterior["gamma"]).ravel(), result


rows = []
summaries, ratios = (
    {},
    {},
)  # the numbers behind the table, for the figure
for item, level in [
    ("TP53", "gene"),
    ("CDKN2A", "gene"),
    ("PIK3CA", "gene"),
    ("NOTCH1", "gene"),
    ("PIK3CA p.E545K", "variant"),
]:
    draws, row = {}, {"item": item}
    for group in groups:
        draws[group], result = group_gamma(item, level, group)
        s = summarize_gamma(result)
        summaries[item, group] = s
        row[f"{group} gamma"] = s["mean"]
        row[f"{group} 89%"] = f"[{s['lower']:.1f}, {s['upper']:.1f}]"
        row[f"{group} carriers"] = result.posterior.attrs[
            "n_tumors_with"
        ]
    # The ratio of the two gammas, from independent draws of each.
    log_ratio = np.log(draws["smoker"]) - np.log(
        np.random.default_rng(0).permutation(draws["never"])
    )
    lo, hi = np.exp(np.percentile(log_ratio, [5.5, 94.5]))
    ratios[item] = (float(np.exp(np.median(log_ratio))), lo, hi)
    row["smoker / never"] = float(np.exp(np.median(log_ratio)))
    row["ratio 89%"] = f"[{lo:.2f}, {hi:.2f}]"
    row["P(smoker > never)"] = float((log_ratio > 0).mean())
    rows.append(row)
group_table = pd.DataFrame(rows).set_index("item")
group_table
```

    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  466  187      250      820      3915      3767  1.00         3     3.3


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  642  242      360     1100      3937      3557  1.00       3.9     4.1


    Gamma posterior summary:
          mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  400  3e+02      130      850      4210      3824  1.00         4     5.7


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  204  109       87      400      3916      3961  1.00       1.7     2.5


    Gamma posterior summary:
           mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  14.2  7.2      5.8       27      3598      3539  1.00      0.12    0.17


    Gamma posterior summary:
           mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  20.8  8.6       11       35      3937      3988  1.00      0.14    0.21


    Gamma posterior summary:
          mean   sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  9.4  4.3      4.3       17      4104      3921  1.00     0.067   0.082


    Gamma posterior summary:
           mean    sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  6.72  2.46      3.6       11      3615      3746  1.00     0.041   0.041


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  5e+03  4e+03     1200    12000      4069      3935  1.00        62     120


    Gamma posterior summary:
            mean     sd eti89_lb eti89_ub  ess_bulk  ess_tail r_hat mcse_mean mcse_sd
    gamma  8e+03  4e+03     3700    16000      4284      3952  1.00        62      97





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>never gamma</th>
      <th>never 89%</th>
      <th>never carriers</th>
      <th>smoker gamma</th>
      <th>smoker 89%</th>
      <th>smoker carriers</th>
      <th>smoker / never</th>
      <th>ratio 89%</th>
      <th>P(smoker &gt; never)</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TP53</th>
      <td>465.607816</td>
      <td>[245.0, 821.0]</td>
      <td>41</td>
      <td>641.954182</td>
      <td>[355.5, 1100.0]</td>
      <td>141</td>
      <td>1.391743</td>
      <td>[0.62, 3.18]</td>
      <td>0.73475</td>
    </tr>
    <tr>
      <th>CDKN2A</th>
      <td>400.011743</td>
      <td>[130.5, 853.1]</td>
      <td>14</td>
      <td>203.855885</td>
      <td>[87.2, 399.1]</td>
      <td>36</td>
      <td>0.533024</td>
      <td>[0.16, 1.85]</td>
      <td>0.20300</td>
    </tr>
    <tr>
      <th>PIK3CA</th>
      <td>14.196558</td>
      <td>[5.8, 27.3]</td>
      <td>8</td>
      <td>20.795769</td>
      <td>[10.8, 34.9]</td>
      <td>41</td>
      <td>1.515456</td>
      <td>[0.58, 4.14]</td>
      <td>0.75275</td>
    </tr>
    <tr>
      <th>NOTCH1</th>
      <td>9.401658</td>
      <td>[4.3, 17.4]</td>
      <td>11</td>
      <td>6.716676</td>
      <td>[3.6, 11.3]</td>
      <td>30</td>
      <td>0.739648</td>
      <td>[0.29, 1.75]</td>
      <td>0.29650</td>
    </tr>
    <tr>
      <th>PIK3CA p.E545K</th>
      <td>5096.416835</td>
      <td>[1182.8, 11992.6]</td>
      <td>2</td>
      <td>8309.801373</td>
      <td>[3720.7, 15611.6]</td>
      <td>10</td>
      <td>1.854209</td>
      <td>[0.52, 7.74]</td>
      <td>0.77075</td>
    </tr>
  </tbody>
</table>
</div>




```python
from matplotlib import ticker  # axis tick placement and labels

items = list(group_table.index)
y = np.arange(len(items))[::-1]
fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.2), sharey=True, width_ratios=[3, 2]
)
ax = axes[0]
for group, offset, color in [
    ("smoker", 0.15, "C1"),
    ("never", -0.15, "C0"),
]:
    s = [summaries[item, group] for item in items]
    ax.errorbar(
        [x["mean"] for x in s],
        y + offset,
        xerr=[
            [x["mean"] - x["lower"] for x in s],
            [x["upper"] - x["mean"] for x in s],
        ],
        fmt="o",
        ms=4,
        capsize=2,
        color=color,
        label=group,
    )
ax.axvline(1, color="0.6", lw=1, ls="--")
ax.set_xscale("log")
ax.set_yticks(y, items)
ax.set_xlabel("gamma (posterior mean, 89% interval)")
ax.legend(frameon=False, loc="upper left")

ax = axes[1]
median, lower, upper = (
    np.array(v) for v in zip(*(ratios[i] for i in items))
)
ax.errorbar(
    median,
    y,
    xerr=[median - lower, upper - median],
    fmt="o",
    ms=4,
    capsize=2,
    color="0.2",
)
ax.axvline(1, color="0.6", lw=1, ls="--")
ax.set_xscale("log")
# Ticks at powers of 2 (1/4, 1/2, 1, 2, 4), fractions below 1.
ax.xaxis.set_major_locator(ticker.LogLocator(base=2))
ax.xaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"1/{1 / v:g}"
    )
)
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.set_xlabel("smoker / never-smoker gamma (89%)")
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_80_0.png)
    


**Reading the figure.** Left: each gene's (and E545K's) gamma in the
two groups, smokers orange, never-smokers blue; right: the ratio of
the two, with the dashed line at "no difference". The groups'
intervals overlap for every item, and every ratio's interval crosses
1; the never-smokers' intervals are mostly the wider ones, from a
group about a third the size. The width of the ratio's interval is the
answer to "how large a difference could this cohort have missed".

**Reading the table.** No ratio's interval excludes 1. TP53's gamma is
somewhat higher in smokers and CDKN2A's in never-smokers, but the
intervals run from about half to two or three times: with 74
never-smokers (and, for E545K, two carriers among them), this cohort
cannot tell these genes' selection apart between the groups. That
is a result worth reporting as it is ("no evidence of a difference,
and differences up to threefold cannot be excluded"), and it is
what a frequency comparison hides: TP53 is mutated in 56% of the
smokers' tumors and 55% of the never-smokers', and the gamma
comparison says that similar frequency is consistent with similar
selection, given each tumor's rates.

**How the uncertainty is computed, and what it leaves out.** The
ratio's interval comes from pairing independent draws of the two
groups' posteriors. Both gammas share the same rate model, and its
uncertainty (the covariate effects, `r_g`) moves both the same way;
pairing independent draws ignores that shared part, which makes
the ratio's interval somewhat *wider* than it should be. So a ratio
whose interval excludes 1 here is a conservative finding. A
`P(smoker > never)` near 0.5 means no evidence of a difference,
not evidence of no difference: check the interval's width before
writing "the same".

**Common mistakes in group comparisons.**

- Comparing frequencies (or the fraction of tumors mutated) between
  groups whose burdens differ.
- Comparing two groups' gammas from two separately fitted rate
  models: their difference then mixes selection with the two fits'
  differences.
- Forgetting confounders. In head and neck cancer, never-smokers are
  enriched for tumors positive for human papillomavirus (HPV), whose
  biology differs (HPV's E6 protein degrades p53, so TP53 mutations
  are rarely needed). A smoker/never-smoker difference in TP53
  selection may be an HPV difference. Compare within HPV-negative
  tumors before attributing it to tobacco.
- Testing many genes and reporting the few with intervals excluding
  1: correct for the number of genes compared, or decide the genes
  before looking.

## 11. The figures sigmutsel draws

The package has its own plotting functions (in `sigmutsel.figures`,
two of them also as methods of `Model`). Each returns its figure and
saves it when given a path (`save=`, or `save_path=` for the
correlations).

| function | what it draws | here |
|---|---|---|
| `figures.plot_sig_decomposition_with_burden` | each tumor's signature shares, tumors sorted by burden | section 4 |
| `figures.plot_signature_profile` | a 96-type profile: a signature, or a tumor's own spectrum | below |
| `Model.plot_gamma_results` | gammas and their intervals against the carriers | below |
| `figures.comparison_with_observed` | observed against expected counts per gene, with R² | below |
| `Model.plot_signature_correlations` | how each signature's mutations follow gene covariates | below |

**Signature profiles.** A signature is a probability over the 96
mutation types (six substitutions, each in 16 contexts of the bases
on either side). Two of HNSC's, and one tumor's own calls:


```python
from sigmutsel.figures import plot_signature_profile

signatures = dataset.signature_matrix
tumor = burden.idxmax()  # the tumor with the most calls
spectrum = (
    calls.loc[calls["Tumor_Sample_Barcode"] == tumor, "type"]
    .value_counts(normalize=True)
    .reindex(signatures.index, fill_value=0.0)
)
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5))
for ax, profile, label in (
    (axes[0], signatures["SBS4"], "SBS4\n(tobacco)"),
    (axes[1], signatures["SBS2"], "SBS2\n(APOBEC)"),
    (axes[2], spectrum, f"one tumor\n({burden.max()} calls)"),
):
    plot_signature_profile(
        profile, ax=ax, ylabel=label, context_labels=False
    )
fig.tight_layout()
plt.show()
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_84_0.png)
    


**Reading the figure.** Tobacco's SBS4 is mostly C>A (blue); APOBEC's
SBS2 is C>T (red), highest where a T precedes the C (its TCW context).
The tumor's own profile is the mixture its decomposition found, with
the noise of a finite number of calls.

**Gammas against carriers.** The gene gammas of section 7, with 89%
intervals, against the number of tumors carrying a non-silent
mutation in the gene. Passenger genes are open circles and Census
oncogenes diamonds:


```python
gene_keys = [
    k for k in model.gammas if k.startswith("ENSG") and "__" not in k
]
# The package draws at the size of a slide; a higher resolution
# makes it readable here.
with plt.rc_context({"figure.dpi": 200}):
    fig = model.plot_gamma_results(
        keys=gene_keys, level="gene", prob=0.89, show=True
    )
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_86_0.png)
    


**Reading the figure.** Frequency and selection are different axes:
a gene can be mutated in many tumors and sit at 1 (TTN, MUC16: long
genes) or in fewer and sit far above it.

**Observed against expected, on linear axes.** The pairs of section
5's fit check, drawn by the package, with R² and the mean percentage
error written on the figure:


```python
from sigmutsel.figures import comparison_with_observed

# The package draws at the size of a slide; a higher resolution
# makes it readable here.
with plt.rc_context({"figure.dpi": 200}):
    fig = comparison_with_observed(
        per_gene["expected"], per_gene["observed"], show=True
    )
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_88_0.png)
    


**Reading the figure.** On linear axes the few genes with the most
mutations stretch the scale and decide most of R², while the
thousands of genes with a handful of mutations pile up near the
origin; section 5's logarithmic version shows them. Read both.

**Signatures against covariates.** Each gene's calls are attributed to
the signatures (by each tumor's exposures and each signature's
probability of the call's type), and each signature's per-gene totals
are correlated with a few raw covariates, read by name from the
Zenodo record (only those columns are downloaded): mean replication
time, expression in the tumors, and two histone marks from the
Roadmap epigenome of esophagus (E079).


```python
RAW_COLUMNS = {
    "mrt": "replication time (late = 1)",
    "tpm_unstranded": "expression in the tumors",  # already in log
    "e079_h3k36me3_fc_signal_cds_span_logmean": "H3K36me3 (transcribed)",
    "e079_h3k9me3_fc_signal_cds_span_logmean": "H3K9me3 (heterochromatin)",
}
raw = sigmutselcovs.fetch_covariate_matrix(
    PROJECT, "full", columns=list(RAW_COLUMNS)
).rename(columns=RAW_COLUMNS)
# The package draws at the size of a slide; a higher resolution
# makes it readable here.
with plt.rc_context({"figure.dpi": 200}):
    fig, _, _ = model.plot_signature_correlations(
        covariates=raw, top_n=8, show=True
    )
```


    
![png](selection_in_hnsc_files/selection_in_hnsc_90_0.png)
    


**Reading the figure.** For each of the eight signatures with the most
calls, one bar per covariate: the correlation, across genes, between
the signature's calls in the gene and the covariate; the grey line is
the number of calls attributed to the signature (right axis).
Replication time correlates positively with all eight, most for
tobacco's SBS4, SBS100 and the clock-like SBS5, and least for APOBEC's
SBS2 and SBS13, which follow replication timing little. Expression and
H3K36me3, the marks of transcribed genes, correlate negatively with
most (transcribed genes are repaired more); H3K9me3 is mixed. These
raw correlations also carry gene length (longer genes collect more
calls of every signature) and each other, so read them as a first
look; the model uses all covariates together, through their
components.

## 12. Reading the output correctly: a checklist

- **Gamma is the strength of selection**: how many times more
  often a mutation is found in tumors than its neutral rate
  predicts. It is not a probability of being a driver, not the
  mutation's fitness effect in absolute units, and not its
  "impact" on the patient.
- **Frequency is rate times selection.** A frequent mutation can be
  a fast-mutating site under modest selection; a rare one can be a
  slow site under strong selection. Always show the rate.
- **Compare genes with the EB estimates** (`eb_*`), not the flat
  ones; decide "selected or not" with `called` (`q`), not with an
  interval.
- **Gene gamma vs variant gamma.** For oncogenes, read variant (or
  missense-channel) gammas; for tumor suppressors, the `plof`
  channel.
- **A variant seen in one tumor** has a gamma set almost entirely
  by its rate (about 1/rate): ranking singletons by gamma ranks
  them by how slowly they mutate.
- **Near-neutral variant gammas are biased upward** by the
  same-gene hold-out (tumors with another hit in the gene, mostly
  high-burden ones, leave the variant's absent set): read a
  variant gamma of a few as possibly neutral.
- **mu_gs does not include r_g**; multiply by
  `compute_r_g_for_evaluation()` yourself.
- **QC consistency**: pass the same `excluded_samples` everywhere.

## 13. What next: project ideas

Each project below is a summer's work with this tutorial's tools.

1. **A letter on a frequency-based claim.** Find a published paper
   that ranks or compares mutations by frequency (or calls the most
   frequent mutation the main driver) and redo its comparison with
   rates and gammas, as in project (a).
2. **Selection by group.** Smokers and never-smokers in lung
   adenocarcinoma (`LUAD`; a larger cohort: run it on a server),
   young and old patients (`diagnoses.age_at_diagnosis` in the same
   GDC query), sexes, HPV status in head and neck cancer, molecular
   subtypes. Which genes' gammas differ, and which differences
   survive a confounder?
3. **APOBEC hotspots across cancers.** PIK3CA E545K/E542K in
   bladder (`BLCA`), cervical (`CESC`), breast (`BRCA`) and head and
   neck cancer: does the helical/kinase difference hold everywhere?
4. **The synonymous control.** Estimate the `syn` channel gamma for
   the 50 most mutated genes of a cohort. Which are above 1, and
   why (splicing, mapping, a rate error)?
5. **A cohort outside TCGA.** Any GRCh38 MAF works; skip the
   TCGA-specific sample QC, and give covariates with sigmutselcovs
   (`project_genes` places your genes in a TCGA cohort's
   components).

## Troubleshooting

- `RuntimeWarning: No installed GRCh38 genome`: run
  `sigmutsel-setup` (or `python -c "from sigmutsel.setup import
  install_reference_genome; install_reference_genome()"`), then
  rebuild the dataset: one built without the genome has other
  opportunities.
- `FileNotFoundError: ... not found ... python -m sigmutsel setup`:
  reference data missing; `sigmutsel-check-data` lists what is.
- The fit warns "There are not enough devices to run parallel
  chains" (and section 5 printed `JAX devices: 1`): `XLA_FLAGS`
  was set after JAX was imported, which happens when cells run out
  of order. Restart the kernel and run the first cell first; or
  leave it, since the chains then run one after another with the
  same result, only slower.
- Killed / out of memory during the dataset build: the build peaks
  at about 12 GB for this cohort; use a smaller cohort, or a
  machine with more memory.
- A gamma raises `ValueError`: the item is not in the dataset (check
  the spelling, `"GENE p.X123Y"`, and that it has a call in a kept
  tumor) or the gene has no rate.
- `estimate_gamma` returns quickly with a huge gamma and
  `likelihood_saturated` in its attributes: the data cannot bound
  it from above (all carriers, almost no rate); report it as a lower
  bound.
