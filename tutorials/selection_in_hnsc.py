# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Selection in head and neck cancer: a first-week tutorial
#
# A mutation that is common in tumors is not necessarily one that
# helps tumors grow. It may simply be a mutation that *happens* often:
# some DNA sites mutate tens of times faster than others, because of
# their sequence context, the mutational processes active in the
# tumor (tobacco smoke, the APOBEC enzymes, ultraviolet light, ...)
# and where the gene sits in the genome. **sigmutsel** separates the
# two. For every tumor it estimates how often each site would mutate
# with no selection (the **neutral mutation rate**, `mu`), and then
# how many times more often a gene or a variant is found mutated than
# that rate predicts (the **selection coefficient**, `gamma`):
# `gamma = 1` is neutral, `gamma = 10` means the mutation is found in
# tumors ten times as often as its rate alone would put it there.
#
# In this tutorial you will:
#
# 1. download a public TCGA cohort (HNSC, head and neck squamous cell
#    carcinoma, 410 tumors) and select the right files;
# 2. build a mutation dataset, with its quality control and the
#    mutational-signature decomposition of each tumor;
# 3. fit the neutral rate model, with gene covariates from Zenodo;
# 4. read rates per tumor, per gene and per variant;
# 5. estimate selection for genes and variants, with intervals;
# 6. make the selection call for every gene (is it selected?) and
#    estimate its strength (how strongly?);
# 7. work through two small research projects: **(a)** a frequent
#    mutation that is mostly mutation rate against a rarer one that is
#    strongly selected, and **(b)** comparing the strength of selection
#    between two groups of tumors (smokers and never-smokers).
#
# Each step says what it does in plain words, what to check, and the
# mistakes people make. You need no prior experience with the
# package; some Python and pandas, and the idea of a probability
# distribution, are enough.
#
# ## Before you start
#
# **Install** (Python 3.12, Linux or macOS; a fresh virtual
# environment is strongly recommended):
#
# ```sh
# python3.12 -m venv sigmutsel_env
# source sigmutsel_env/bin/activate
# pip install git+https://github.com/alfaromurillo/sigmutsel.git \
#     git+https://github.com/alfaromurillo/sigmutselcovs.git
# sigmutsel-setup        # reference data + the GRCh38 genome, once
# sigmutsel-check-data   # every line should say "Found"
# ```
#
# `sigmutsel-setup` downloads about 300 MB of reference files and
# installs SigProfilerMatrixGenerator's GRCh38 genome (about 3 GB; on
# our connection it took 18 minutes, most of it the genome). **Do not
# skip the genome**: without it nothing fails, but every step that
# reads the genome falls back to defaults and the rates come out
# different (you would see a `RuntimeWarning: No installed GRCh38
# genome`).
#
# To run this file as a notebook, open it in Jupyter with `jupytext`
# installed (`pip install jupytext notebook`) or in VS Code; or run it
# as a plain script, `python selection_in_hnsc.py`.
#
# **What it costs.** Measured on a desktop (16 cores, of which the fit
# uses 4; 100 Mb/s connection). The outputs shown on this page come
# from that run.
#
# | step | time | peak memory | disk |
# |---|---|---|---|
# | download and select MAFs | 1 min | small | 180 MB |
# | build the dataset (QC, signatures, opportunities) | 6 min | 12 GB | 1 GB |
# | calibration steps (sample QC, spectra, site weights) | 16 min | 16-20 GB | |
# | fit the rate model (NUTS, 4 chains) | 47 min (300 components) | under 10 GB | 0.6 GB |
# | gammas (each) | 4-8 s | | |
# | selection table, every gene | 5 min (4 processes) | | |
#
# **Memory.** One step, the site weights, needs 16-20 GB whatever the
# cohort (it holds every coding site of the exome at once; 16 GB in
# the run shown here, 20 GB in a run as a script); the rest fits in
# 12 GB. On a 16 GB laptop set `SITE_WEIGHTS = False` in
# step 3: everything runs, and the rates lose the correction for how
# detection varies along each exon (fine for learning the package;
# use a larger machine for results you report). The first sample QC
# parses TCGA's copy-number file (3 minutes); later runs reuse it.

# %%
import os

# The rate model's sampler runs four chains in parallel only if JAX
# sees four devices. This must be set before anything imports JAX.
os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=4"
)

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sigmutsel
from sigmutsel import Model, MutationDataset

# The package logs every step at INFO; warnings are what to read.
logging.getLogger("sigmutsel").setLevel(logging.WARNING)

PROJECT = "HNSC"
WORK = Path("hnsc")  # everything the tutorial writes goes here
WORK.mkdir(exist_ok=True)
print("sigmutsel", sigmutsel.__version__)


def show(fig, name):
    """Save a figure as WORK/<name>.png and display it in a notebook.

    The signature step imports SigProfilerAssignment, which switches
    matplotlib to a non-interactive backend: after it, figures no longer
    appear by themselves in Jupyter, so show the saved image instead.
    """
    path = WORK / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:  # a plain script: the PNG is the figure
        return
    if get_ipython() is not None:
        display(Image(filename=path))


# %% [markdown]
# ## 1. Download and select the mutation calls
#
# The GDC (NCI's Genomic Data Commons) publishes TCGA's somatic
# mutation calls as one MAF file (a tab-separated table of mutations)
# per sequenced *aliquot*. `download_project_maf_files` finds a
# project's open-access MAFs, downloads them over HTTPS and unpacks
# them into one flat directory. It needs no account and no extra
# program; a rerun only fetches what is missing.

# %%
from sigmutsel.download_tcga_data import download_project_maf_files

raw_mafs = WORK / "all_maf_files_raw"
mafs = download_project_maf_files(PROJECT, raw_mafs)
print(len(mafs), "MAF files")

# %% [markdown]
# **Select the files before anything reads them.** One aliquot per
# file means the directory also holds recurrent and metastatic
# samples, several aliquots of the same patient and patients treated
# before the sample was taken. Counted as separate tumors, a patient
# with four aliquots makes each of their mutations look recurrent.
# `select_tcga_maf_files` keeps primary tumors (TCGA sample types 01
# and 03), the newest aliquot of each patient, and drops patients the
# GDC records as treated before collection. **Its defaults keep every
# aliquot**, so pass the options below.

# %%
from sigmutsel.tcga_sample_selection import select_tcga_maf_files

selected = WORK / "all_maf_files"
if not selected.is_dir():
    chosen = select_tcga_maf_files(
        raw_mafs,
        duplicate_policy="newest",
        exclude_prior_treatment=True,
        project=PROJECT,
    )
    selected.mkdir()
    for f in chosen:  # links, so the downloads stay where they are
        (selected / f.name).symlink_to(f.resolve())
print(len(list(selected.glob("*.maf"))), "files selected")

# %% [markdown]
# ## 2. Build the dataset
#
# A `MutationDataset` turns the MAFs into everything the model needs.
# Four things happen here, and they take about six minutes:
#
# - **Call quality control** (`qc_mode=True`): calls in repeat regions
#   (where reads map badly), duplicated calls, and multi-base events
#   (doublets, kept aside because the model is about single-base
#   substitutions) are removed, and calls on common germline alleles
#   (gnomAD frequency above `1.5e-4`) are masked, because the GDC's
#   own filter already removes most calls there.
# - **Mutational signatures**: each tumor's mutations are decomposed
#   into COSMIC signatures (SBS4 tobacco, SBS2/SBS13 APOBEC, SBS1/SBS5
#   clock-like, ...). The decomposition runs twice: first unrestricted,
#   to find sequencing-artifact mutations and treatment-exposed
#   tumors; then restricted to the signatures known in this cancer
#   type (`exclude_signature_subgroups=PROJECT`).
# - **The channel universe**: each call is placed on its gene's
#   transcript and classified as synonymous, missense, nonsense or
#   splice, and each gene's **opportunity** (how many sites could
#   mutate to each consequence, in each trinucleotide context) is
#   counted.
# - **The gene universe** (`gene_universe="wes_target"`): every gene
#   of the exome capture, mutated in this cohort or not. The default,
#   `"own_cohort"`, keeps only mutated genes and inflates the rates of
#   a small cohort; always pass `"wes_target"`.
#
# The built dataset is saved, so a rerun of this cell loads it in
# seconds.

# %%
from sigmutsel.qc import load_repeat_intervals
from sigmutsel.setup import download_repeatmasker_bed

dataset_dir = WORK / f"{PROJECT}_dataset"
if dataset_dir.is_dir():
    dataset = MutationDataset.load_dataset(dataset_dir)
else:
    dataset = MutationDataset(selected, signature_class="SBS")
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
    dataset.build_full_dataset(
        gene_universe="wes_target", regenerate_mutation_db=False
    )
    dataset.save_dataset(dataset_dir)

print(
    dataset.n_samples,
    "tumors,",
    len(dataset.model_db),
    "modelled calls",
)

# %% [markdown]
# **What to check.** The number of tumors is a little below the number
# of files: tumors whose every call was removed by the QC, or that the
# signature step flagged as treatment-exposed, are gone.
# `dataset.model_db` is the table of calls the model counts (one row
# per call, with its gene, consequence and context); `mutation_db`
# also holds the calls outside coding territory, which the model does
# not use. Always take calls from `model_db`.

# %%
calls = dataset.model_db
calls[
    [
        "Tumor_Sample_Barcode",
        "gene",
        "variant",
        "Variant_Classification",
    ]
].head()

# %% [markdown]
# ## 3. Calibration steps (once per cohort)
#
# Five more steps finish the dataset. They are separate because each
# needs data or choices of its own; all are saved with the dataset.
#
# **Sample QC.** A tumor sample with little tumor in it (low purity),
# or whose variant allele fractions do not look like a tumor's, gives
# too few calls and would look like a low-rate tumor. Such tumors are
# flagged, from TCGA's ABSOLUTE purity and copy-number calls (public,
# downloaded once), and left out of every fit below through
# `excluded_samples`. This step is TCGA-specific: for other data, use
# your own purity estimates or skip it.

# %%
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

# %% [markdown]
# **The rest**, in one cell:
#
# - repeat regions leave the *opportunity* too, not only the calls
#   (a site where no call can be made must not count as a site that
#   could mutate);
# - each genomic position is assigned to one gene, where genes
#   overlap;
# - each tumor's mutational **spectrum** (the share of each of the 96
#   trinucleotide substitution types) is estimated by shrinking its
#   own counts toward the cohort's, by an amount chosen on held-out
#   mutations (`select_kappas`); this matters most for tumors with few
#   mutations;
# - **site weights**: detection is lower near exon ends and differs
#   by DNA strand, and the weights correct the opportunity for it,
#   fitted on passenger genes;
# - homozygous deletions and coding indels are recorded, so that a
#   tumor that lost a gene, or hit it by an indel, is not counted as
#   evidence that the gene is not selected.

# %%
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
    Model(
        dataset,
        None,
        cov_effects_per_sigma=False,
        prob_g_tau_tau_independent=False,
    ).estimate_site_weights(excluded_samples=EXCLUDED)
if dataset.homdel_db is None:
    dataset.homdel_db = homozygously_deleted_genes(
        segments, load_or_build_transcript_models(), tumors
    )
if dataset.indel_db is None:
    dataset.build_indel_db()
dataset.save_dataset(dataset_dir, overwrite=True)
print("spectrum shrinkage:", dataset.spectrum_shrinkage.label())

# %% [markdown]
# ## 4. The tumors: burden and signatures
#
# Before fitting anything, look at the data. Each tumor's number of
# calls (its **mutation burden**) and the share of them its signature
# decomposition gives to APOBEC (SBS2 + SBS13) and to tobacco (SBS4):

# %%
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
show(fig, "burden_signatures")

# %% [markdown]
# **Reading this.** Each point is a tumor. Burdens span two orders of
# magnitude (from a handful of calls to over a thousand), so a single
# "cohort rate" would describe almost no tumor: the model gives each
# tumor its own rate, from its own burden and spectrum. A group of
# tumors gets most of its mutations from APOBEC (left, the points to
# the right), another from tobacco (right). APOBEC prefers a C in a
# TCW context (W = A or T), and two of the most common mutations of
# this cancer, PIK3CA E545K and E542K, are C>T changes in exactly that
# context. Keep that in mind for project (a).

# %% [markdown]
# ## 5. Fit the neutral rate model
#
# The rate of gene `g` in tumor `j` combines:
#
# - the tumor's **spectrum and burden** (from the signatures);
# - the gene's **opportunity** (its sites, by context and consequence,
#   with the site weights);
# - a **covariate** term `exp(c . x_g)`: genes that replicate late or
#   sit in closed chromatin mutate faster, and `x_g` are those
#   properties of the gene;
# - a **channel factor**: non-synonymous sites get their own
#   intercept;
# - a per-gene **rate correction** `r_g`, learned from the gene's own
#   synonymous (silent) mutations, which selection barely touches.
#
# The covariates come ready-made from Zenodo through **sigmutselcovs**:
# the principal components of about 5,000 per-gene measurements
# (expression, histone marks, chromatin accessibility, replication
# timing) for this tissue, plus 0/1 indicators for genes missing part
# of them. The production model for HNSC uses the first 300
# components (`fitted_n_components` in the record's manifest, chosen
# by held-out cross-validation); a smaller number fits faster.

# %%
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

# %% [markdown]
# Now the model. `prob_g_tau_tau_independent=False` lets the share of
# a tumor's mutations that falls in a gene depend on the tumor's
# spectrum (the recommended setting; it has no default, and `True`
# is an older, simpler model). `assign_cov_matrix` takes the design as given (no
# second PCA). `estimate_channel_rg_cov_effects` fits the covariate
# effects `c`, the channel intercept and the dispersion `theta` of
# `r_g` by Hamiltonian Monte Carlo (NUTS, 4 chains); `sample="full"`
# keeps the posterior draws, which the selection estimates below need.
# This is the slow step; the fitted model is saved.

# %%
model_dir = WORK / f"model_nc{N_COMPONENTS}"
if model_dir.is_dir():
    model = Model.load_model(model_dir)
    # A saved model loads its own copy of the dataset: hand it the
    # tables the selection estimates read.
    for table in ("multi_base_db", "homdel_db", "indel_db"):
        setattr(model.dataset, table, getattr(dataset, table))
    model.dataset.sample_qc_flags = dataset.sample_qc_flags
else:
    model = Model(
        dataset,
        None,
        cov_effects_per_sigma=False,
        prob_g_tau_tau_independent=False,
    )
    model.assign_cov_matrix(design)
    model.compute_channel_base_mus()
    model.estimate_channel_rg_cov_effects(
        excluded_samples=EXCLUDED, include_drivers=True, sample="full"
    )
    model.save_model(model_dir)
print(f"theta = {model.rg_theta:.2f}")

# %% [markdown]
# **What to check.**
#
# - **Convergence.** The posterior's R-hat: every parameter's should
#   be below 1.01 (much above means the chains disagree; rerun with
#   more draws, or check the data).
# - **`theta`**, the inverse dispersion of `r_g`: large `theta` means
#   the covariates leave little gene-to-gene variation unexplained.
# - **Fit to held-out data.** The honest check of a rate model is how
#   well it predicts mutation counts it has not seen. Here we compute
#   the R² of passenger genes' non-silent counts using `r_g` from their
#   silent counts only (`compute_r_g_for_evaluation`): the non-silent
#   counts were not used to correct the gene's rate, so the model is
#   predicting them, not fitting them.

# %%
import arviz as az

posterior = model.cov_effects_posteriors
rhat = az.rhat(posterior)
print("max R-hat:", float(max(rhat[v].max() for v in rhat.data_vars)))
r2 = model.estimate_passenger_genes_r2(
    excluded_samples=EXCLUDED,
    target="non_silent_counts",
    gene_scaling=model.compute_r_g_for_evaluation(),
)
print(f"R² of passenger genes' non-silent counts: {r2:.3f}")

# %% [markdown]
# **Common mistakes at this step.**
#
# - Using the *production* `r_g` (`compute_r_g_production`) to judge
#   the fit: it is learned from both channels, so it partly predicts
#   its own target and the R² looks better than it is.
# - Reading R² as the measure of a rate model: a few heavily mutated
#   genes carry most of its sum of squares. Compare models by held-out
#   likelihood instead (the package's `cross_validation` module).
# - Passing a different `excluded_samples` to later steps than to the
#   fit: every per-gene sum is over the kept tumors.

# %% [markdown]
# ## 6. Rates: per tumor, per gene, per variant
#
# **Per gene.** `model.mu_gs` is a genes x tumors table: the expected
# number of non-synonymous mutations of each gene in each tumor, from
# the spectrum, opportunity, covariates and channel factor. It does
# **not** include the gene's own correction `r_g`; multiply by it
# yourself (`compute_r_g_for_evaluation` is the one learned from
# silent mutations only, the one every selection estimate below uses).

# %%
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

# %% [markdown]
# **Reading this.** "Expected carriers" is how many of the kept tumors
# would carry at least one non-synonymous mutation in the gene if it
# were neutral (a tumor with rate `mu` carries one with probability
# `1 - exp(-mu)`). TTN and MUC16 are among the most mutated genes of
# every cancer, and their observed counts are near what their length
# and late replication predict: they are long, not (much) selected.
# TP53 is mutated in about 190 of the 332 kept tumors where neutral
# evolution would put a non-synonymous mutation in about one. This
# comparison of observed with expected is the whole idea; `gamma`
# makes it precise.
#
# **Per variant.** `SiteRates` gives the rate of any single-nucleotide
# variant, protein change or splice change in every tumor, whether it
# was observed or not, without building a variants x tumors table. A
# protein change's rate is the sum over every single-base route to it
# (PIK3CA E545K has one; a change with two possible codons has two).
# The *observable* rate is the one to compare with calls (a germline-
# masked allele can mutate but cannot be called).

# %%
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

# %% [markdown]
# `observable` is variants x tumors. Its sum over the kept tumors is
# the number of tumors expected to carry the variant by mutation
# alone; the ratio of observed to expected is a first, crude look at
# selection.

# %%
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

# %% [markdown]
# **Reading this.** A variant's rate is tiny (a few in a million per
# tumor), so even the most frequent variants are expected in a small
# fraction of one tumor across the whole cohort (0.003 tumors for
# E545K). The ratio is thousands: these are among the most strongly
# selected mutations in this cancer. The rates of variants of the
# same gene differ: E545K and E542K are C>T changes in a TCA context,
# APOBEC's favourite, and mutate about four times faster than H1047R
# (an A>G change). Project (a) is about exactly that.

# %% [markdown]
# ## 7. Selection coefficients for genes and variants
#
# `estimate_gamma` estimates `gamma` for a gene (all its
# non-synonymous sites together), a variant, or one consequence
# channel of a gene. It compares, tumor by tumor, the probability
# that a tumor carries the mutation given its rate and `gamma`, with
# whether it does.
#
# **The package's defaults are not the recommended settings** (they
# are kept for reproducibility of older analyses). Pass these:

# %%
model.estimate_gene_tumor_dispersion(excluded_samples=EXCLUDED)

GAMMA_SETTINGS = dict(
    excluded_samples=EXCLUDED,  # the QC-flagged tumors stay out
    use_mu_posterior=True,  # carry the rate model's uncertainty...
    mu_uncertainty="draws",  # ...exactly, over its posterior draws
    r_g_variant="evaluation",  # r_g from silent mutations only
    gene_tumor_dispersion="fitted",  # tumors differ beyond their rate
)

# %% [markdown]
# What each setting does:
#
# - `use_mu_posterior=True, mu_uncertainty="draws"`: the rates are
#   estimates, not known numbers. With these, gamma's interval
#   averages over the rate model's posterior draws (the *cut*), so it
#   is wide when the rates are uncertain. Without them the interval
#   is too narrow.
# - `r_g_variant="evaluation"`: the gene's rate correction from its
#   silent mutations. The other one (`"production"`) has seen the
#   non-silent mutations, so it would absorb part of the selection it
#   is meant to measure.
# - `gene_tumor_dispersion="fitted"`: mutations of a gene cluster in
#   some tumors more than their rates predict. The dispersion `phi`
#   is fitted on passenger genes (`estimate_gene_tumor_dispersion`,
#   run above) and switched off where it does not improve held-out
#   fit.
#
# Tumors with a homozygous deletion of the gene, a coding indel in
# it, or a doublet there are held out automatically (they met the
# gene's pressure by an event the rates do not describe). For a
# variant, tumors with *another* non-silent mutation in the same gene
# are held out too.

# %%
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

# %% [markdown]
# **Reading this.** `mean`, `lower` and `upper` are the posterior mean
# of gamma and its 89% interval; `tumors informative` are the tumors
# without the mutation that entered the fit (the held-out ones are
# not). TP53's gamma, about 600, says a TP53 non-silent mutation is
# found in tumors about 600 times more often than its rate predicts:
# it is the **strength of selection** on the gene (relative to neutral
# evolution), comparable to other genes' gammas. MUC16 sits at 1; TTN
# at about 1.5, with an interval just above 1. That is a modest
# excess, and TTN is not called in section 8: among 19,000 genes
# some intervals exclude 1 by chance, and TTN, the longest gene,
# is always among the genes one looks at first.
#
# Two cautions. A *gene* gamma averages over all the gene's
# non-synonymous sites: for an oncogene, a few selected hotspots
# among hundreds of neutral sites give a modest gene gamma (PIK3CA,
# HRAS), and its variant gammas are the effect sizes to report. And
# one interval above 1 is not a selection call: across 19,000 genes
# many neutral genes have intervals above 1 by chance. The call is
# section 8.
#
# **Per channel.** For a tumor suppressor, the predicted
# loss-of-function channel (`"plof"`: nonsense and essential splice)
# carries the selection; the synonymous channel (`"syn"`) is a
# control, expected near 1.

# %%
channel_gammas = pd.DataFrame(
    [
        gamma_row(g, level="gene", channel=ch)
        for g in ("TP53", "NOTCH1", "PIK3CA")
        for ch in ("mis", "plof", "syn")
    ]
).set_index("item")
channel_gammas.round(2)

# %% [markdown]
# **Reading this.** TP53 is selected on both channels (its hotspot
# missense mutations and its truncations). NOTCH1, a tumor suppressor
# in this cancer, is selected mostly on `plof` (about 22, against 6 on
# missense). PIK3CA, an oncogene, is selected on missense, and its
# `plof` channel has no carrier: it is reported as an **upper bound
# only** (`mean` and `lower` are NaN), how large gamma can be, not
# where it is. The synonymous channel is the control: neutral by
# design, so its gamma should cover 1. NOTCH1's and PIK3CA's do. TP53's
# does not (about 11, on two carriers): with two calls the interval is
# wide, and the usual suspects are synonymous changes that act on
# splicing, or chance; it is a flag to look at the two calls, not a
# verdict on the rates. A synonymous
# gamma well above 1 in *many* genes would mean the rates are off.
#
# **Variants.**

# %%
variant_gammas = pd.DataFrame(
    [gamma_row(v, level="variant") for v in VARIANTS]
).set_index("item")
variant_gammas.round(0)

# %%
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
show(fig, "gammas")

# %% [markdown]
# **Reading this.** The horizontal axis is logarithmic: the dashed
# line is neutral (gamma = 1). Genes sit between about 1 and several
# hundred (TP53, CDKN2A); hotspot variants at thousands, because a
# variant gamma measures selection on one site, not averaged over a
# gene's hundreds of sites. Intervals are wide for variants with few
# carriers (HRAS G12S has three).

# %% [markdown]
# ## 8. Is a gene selected, and how strongly?
#
# Two questions, two answers, in one table over every gene:
#
# - **The call.** A test of the null hypothesis that the gene is
#   neutral (`gamma = 1`): how surprising its number of carrier tumors
#   is under its rates, integrating over the uncertainty of `r_g`
#   (`p_mid`, the mid-p-value), then Benjamini-Hochberg over all genes
#   (`q`). `called` is `q < 0.1`.
# - **The strength.** Gamma's posterior under an **empirical-Bayes
#   prior**: a spike near 1 (most genes are neutral) plus a slab for
#   the selected ones, both fitted on this cohort's genes. This
#   shrinks the gammas of genes with little information toward 1, so
#   they can be compared across genes (`eb_median`, `eb_lower`,
#   `eb_upper`); `eb_p_above` is the posterior probability that gamma
#   exceeds 2. The flat-prior interval (`flat_*`) is kept beside it,
#   as an estimate only, never as a call.
#
# It takes a few minutes (it runs every gene; `n_jobs` processes).

# %%
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

# %% [markdown]
# **Reading this table.** `n_tumors_with` are the carriers and
# `expected_with` the carriers expected under neutrality. The genes at
# the top are the known drivers of head and neck cancer (TP53, CDKN2A,
# PIK3CA, CASP8, HRAS, NFE2L2, NOTCH1, FBXW7, NSD1, FAT1, HLA-A/B,
# ...). Compare
# `eb_median` with `flat_median`: for a strongly selected gene with
# many carriers they agree; for a gene with few carriers the EB
# estimate is pulled toward 1. That is intended: across thousands of
# genes, the flat estimates of genes with little data scatter widely
# by chance, and the prior, learned from all genes, says how much.
#
# **What to check.** Known passengers (TTN, MUC16) should not be
# called, and are not. Near the bottom of the list there are genes
# you will not find in a cancer gene list: OR11H1 (an olfactory
# receptor), and past the 20 rows shown (`called.tail(3)`) NEK5 and
# CGB8 (one of a cluster of near-identical copies). OR11H1 and CGB8
# belong to multi-copy families, where reads map to the wrong copy
# and calls pile up. A called gene you do not recognise is a gene to examine
# (its calls, its rates, its paralogs), not yet a discovery.

# %%
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
show(fig, "selection_called")

# %% [markdown]
# **Reading this.** One point per gene. The dashed line is "observed =
# expected": a neutral gene scatters around it, more widely at small
# counts. Called genes sit far above it. Long genes (to the right) can
# have many carriers and still sit on the line.

# %% [markdown]
# ## 9. Project (a): frequency is not selection
#
# Suppose a published paper ranked the mutations of head and neck
# cancer by how often they occur, and called the most frequent the
# most important. PIK3CA has three classic hotspots: E545K and E542K
# in the helical domain, H1047R in the kinase domain. Which is the
# most frequent, which mutates fastest, and which is the most strongly
# selected?

# %%
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

# %%
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
show(fig, "pik3ca_hotspots")

# %% [markdown]
# **Reading this.** The two helical-domain hotspots are the frequent
# ones (12 and 13 carriers), and the kinase-domain H1047R has half as
# many (6). But E545K and E542K mutate about four times faster: they
# are C>T changes in APOBEC's TCA context, and APOBEC is active in
# many of these tumors (section 4). Once each is divided by its own
# neutral rate, H1047R comes out the *most* strongly selected, about
# twice the helical ones. Its interval overlaps theirs, so how sure is
# that? From the posterior draws:

# %%
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

# %% [markdown]
# The posterior favours stronger selection on H1047R, but with six
# carriers it does not settle it: a frequency ranking puts the three
# in the opposite order to the selection ranking, and the honest
# statement is "H1047R is at least as strongly selected as the
# helical hotspots despite being half as frequent". (The draws of two
# gammas are paired at random: they share the rate model's
# uncertainty, which this ignores, so the interval is if anything too
# wide.) That frequency and strength of selection part ways in head
# and neck cancer, and that APOBEC deaminates the E545K and E542K
# sites, was shown with cancereffectsizeR's method by Cannataro et al.
# (2019, *Oncogene* 38:3475-3487, doi:10.1038/s41388-018-0657-6): read
# it beside this project.
#
# **How to turn this into a letter.** State the claim you test (the
# paper's ranking by frequency), show frequency, neutral expectation
# and gamma side by side as here, for the paper's own variants, and
# report gamma with its interval. Check that your conclusion does not
# depend on one setting: refit with fewer covariate components, and
# with `gene_tumor_dispersion=None`; a robust difference survives
# both. And say what gamma is not: it is selection relative to the
# variant's own neutral rate, under this rate model; a variant gamma
# is conditional on the site-level rate model (sites recur somewhat
# more often than it predicts), so read a difference of less than
# about a factor of two between two variants with care.

# %% [markdown]
# ## 10. Project (b): selection in two groups of tumors
#
# Does selection on a gene differ between smokers and never-smokers?
# A paper that compares *mutation frequencies* between the groups
# cannot tell: smokers' tumors carry more mutations everywhere, so a
# gene is mutated more often in them even with the same selection.
# Gamma removes that, because each tumor's rate includes its own
# burden and spectrum (tobacco's SBS4 included).
#
# **Step 1: the groups**, from the GDC's clinical data (public, no
# account): `tobacco_smoking_status` of each patient.

# %%
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

# %% [markdown]
# **Step 2: one rate model, two gammas.** The rate model above was
# fitted on all tumors; it already gives every tumor its own rates.
# A group's gamma is estimated on that group's tumors alone, by
# excluding everyone else. **Do not fit a separate rate model per
# group**: the covariate effects and `r_g` are properties of genes,
# best estimated on all tumors, and a small group's own fit would
# make its rates noisier, not more specific.


# %%
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
    row["smoker / never"] = float(np.exp(np.median(log_ratio)))
    row["ratio 89%"] = f"[{lo:.2f}, {hi:.2f}]"
    row["P(smoker > never)"] = float((log_ratio > 0).mean())
    rows.append(row)
group_table = pd.DataFrame(rows).set_index("item")
group_table

# %% [markdown]
# **Reading this.** No ratio's interval excludes 1. TP53's gamma is
# somewhat higher in smokers and CDKN2A's in never-smokers, but the
# intervals run from about half to two or three times: with 74
# never-smokers (and, for E545K, two carriers among them), this cohort
# cannot tell these genes' selection apart between the groups. That
# is a result worth reporting as it is ("no evidence of a difference,
# and differences up to threefold cannot be excluded"), and it is
# what a frequency comparison hides: TP53 is mutated in 56% of the
# smokers' tumors and 55% of the never-smokers', and the gamma
# comparison says that similar frequency is consistent with similar
# selection, given each tumor's rates.
#
# **How the uncertainty is computed, and what it leaves out.** The
# ratio's interval comes from pairing independent draws of the two
# groups' posteriors. Both gammas share the same rate model, and its
# uncertainty (the covariate effects, `r_g`) moves both the same way;
# pairing independent draws ignores that shared part, which makes
# the ratio's interval somewhat *wider* than it should be. So a ratio
# whose interval excludes 1 here is a conservative finding. A
# `P(smoker > never)` near 0.5 means no evidence of a difference,
# not evidence of no difference: check the interval's width before
# writing "the same".
#
# **Common mistakes in group comparisons.**
#
# - Comparing frequencies (or the fraction of tumors mutated) between
#   groups whose burdens differ.
# - Comparing two groups' gammas from two separately fitted rate
#   models: their difference then mixes selection with the two fits'
#   differences.
# - Forgetting confounders. In head and neck cancer, never-smokers
#   are enriched for HPV-positive tumors, whose biology differs
#   (HPV's E6 protein degrades p53, so TP53 mutations are rarely
#   needed). A smoker/never-smoker difference in TP53 selection may be
#   an HPV difference. Compare within HPV-negative tumors before
#   attributing it to tobacco.
# - Testing many genes and reporting the few with intervals excluding
#   1: correct for the number of genes compared, or decide the genes
#   before looking.

# %% [markdown]
# ## 11. Reading the output correctly: a checklist
#
# - **Gamma is the strength of selection**: how many times more
#   often a mutation is found in tumors than its neutral rate
#   predicts. It is not a probability of being a driver, not the
#   mutation's fitness effect in absolute units, and not its
#   "impact" on the patient.
# - **Frequency is rate times selection.** A frequent mutation can be
#   a fast-mutating site under modest selection; a rare one can be a
#   slow site under strong selection. Always show the rate.
# - **Compare genes with the EB estimates** (`eb_*`), not the flat
#   ones; decide "selected or not" with `called` (`q`), not with an
#   interval.
# - **Gene gamma vs variant gamma.** For oncogenes, read variant (or
#   missense-channel) gammas; for tumor suppressors, the `plof`
#   channel.
# - **A variant seen in one tumor** has a gamma set almost entirely
#   by its rate (about 1/rate): ranking singletons by gamma ranks
#   them by how slowly they mutate.
# - **Near-neutral variant gammas are biased upward** by the
#   same-gene hold-out (tumors with another hit in the gene leave the
#   variant's absent set, and those are high-burden tumors): read a
#   variant gamma of a few as possibly neutral.
# - **mu_gs does not include r_g**; multiply by
#   `compute_r_g_for_evaluation()` yourself.
# - **QC consistency**: pass the same `excluded_samples` everywhere.
#
# ## 12. What next: project ideas
#
# Each of these is a summer project with this tutorial's tools.
#
# 1. **A letter on a frequency-based claim.** Find a published paper
#    that ranks or compares mutations by frequency (or calls the most
#    frequent mutation the main driver) and redo its comparison with
#    rates and gammas, as in project (a).
# 2. **Selection by group.** Smokers and never-smokers in lung
#    adenocarcinoma (`LUAD`; a larger cohort: run it on a server),
#    young and old patients (`diagnoses.age_at_diagnosis` in the same
#    GDC query), sexes, HPV status in head and neck cancer, molecular
#    subtypes. Which genes' gammas differ, and which differences
#    survive a confounder?
# 3. **APOBEC hotspots across cancers.** PIK3CA E545K/E542K in
#    bladder (`BLCA`), cervical (`CESC`), breast (`BRCA`) and head and
#    neck cancer: does the helical/kinase difference hold everywhere?
# 4. **The synonymous control.** Estimate the `syn` channel gamma for
#    the 50 most mutated genes of a cohort. Which are above 1, and
#    why (splicing, mapping, a rate error)?
# 5. **A cohort outside TCGA.** Any GRCh38 MAF works; skip the
#    TCGA-specific sample QC, and give covariates with sigmutselcovs
#    (`project_genes` places your genes in a TCGA cohort's
#    components).
#
# ## Troubleshooting
#
# - `RuntimeWarning: No installed GRCh38 genome`: run
#   `sigmutsel-setup` (or `python -c "from sigmutsel.setup import
#   install_reference_genome; install_reference_genome()"`), then
#   rebuild the dataset: one built without the genome has other
#   opportunities.
# - `FileNotFoundError: ... not found ... python -m sigmutsel setup`:
#   reference data missing; `sigmutsel-check-data` lists what is.
# - The fit runs on one core: `XLA_FLAGS` was set after JAX was
#   imported; restart the kernel and run the first cell first.
# - Killed / out of memory during the dataset build: the build peaks
#   at about 12 GB for this cohort; use a smaller cohort, or a
#   machine with more memory.
# - A gamma raises `ValueError`: the item is not in the dataset (check
#   the spelling, `"GENE p.X123Y"`, and that it has a call in a kept
#   tumor) or the gene has no rate.
# - `estimate_gamma` returns quickly with a huge gamma and
#   `likelihood_saturated` in its attributes: the data cannot bound
#   it from above (all carriers, almost no rate); report it as a lower
#   bound.
