"""Tests for the consequence-split (syn/non-syn) channel machinery.

Covers the three pieces stage 2 of the channel-split model adds:
``compute_mu_g_channel_per_tumor`` (the split baseline rates),
``MutationDataset``'s silent presence matrix, and ``Model``'s
two-channel shared-``c`` fit plus the ``target="non_silent"``
evaluation.

The load-bearing property is that the two channels sum back to the
merged model exactly, for *both* ``p_gτ`` variants -- the τ-dependent
and τ-independent paths normalise differently (one counts positions,
the other opportunities), and getting either denominator wrong is
silent, not an error.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.constants import canonical_types_order
from sigmutsel.estimate_mus import (
    compute_mu_g_channel_per_tumor,
    compute_mu_g_per_tumor,
)
from sigmutsel.estimate_presence import compute_genes_present
from sigmutsel.models import Model, MutationDataset

_GENES = ["ENSG_A", "ENSG_B", "ENSG_C"]
_TUMORS = ["T1", "T2", "T3"]


def _synthetic_opportunities(seed=0):
    """A (contexts, syn, nonsyn) triple satisfying the identity.

    Built the way the real tables are: pick a full context table,
    then split each context's count into the 3 types sharing it, so
    ``syn + nonsyn`` reproduces the broadcast context counts exactly.
    """
    from sigmutsel.constants import extract_context

    rng = np.random.default_rng(seed)
    contexts_order = sorted(
        {extract_context(t) for t in canonical_types_order}
    )
    contexts = pd.DataFrame(
        rng.integers(
            20, 200, size=(len(_GENES), len(contexts_order))
        ),
        index=_GENES,
        columns=contexts_order,
    )

    syn = pd.DataFrame(
        0, index=_GENES, columns=canonical_types_order, dtype=int
    )
    nonsyn = pd.DataFrame(
        0, index=_GENES, columns=canonical_types_order, dtype=int
    )
    for sbs_type in canonical_types_order:
        total = contexts[extract_context(sbs_type)]
        # an arbitrary but reproducible synonymous share per gene/type
        share = rng.integers(0, 4, size=len(_GENES))
        syn[sbs_type] = np.minimum(total.values, share)
        nonsyn[sbs_type] = total.values - syn[sbs_type].values

    return contexts, syn, nonsyn


def _synthetic_mu_taus(seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.random((len(_TUMORS), 96)) * 0.01,
        index=_TUMORS,
        columns=canonical_types_order,
    )


@pytest.mark.parametrize("tau_independent", [False, True])
def test_channels_sum_to_merged_mu_g(tau_independent):
    """μ_g^(syn) + μ_g^(nonsyn) == μ_g, for both p_gτ variants."""
    contexts, syn, nonsyn = _synthetic_opportunities()
    mu_taus = _synthetic_mu_taus()

    merged = compute_mu_g_per_tumor(
        mu_taus=mu_taus,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_independent,
    )
    channels = [
        compute_mu_g_channel_per_tumor(
            mu_taus=mu_taus,
            channel_contexts_by_gene=channel,
            contexts_by_gene=contexts,
            prob_g_tau_tau_independent=tau_independent,
        )
        for channel in (syn, nonsyn)
    ]

    pd.testing.assert_frame_equal(
        channels[0] + channels[1], merged, check_exact=False
    )


def test_channel_rates_are_a_small_fraction_for_syn():
    """Sanity check on the denominator: the synonymous channel must
    carry roughly its opportunity share, not ~1/2 or ~4× of it.

    A synonymous *denominator* (rather than the full τ-site count)
    would be the natural mistake here, and would inflate this ratio
    by about 4×.
    """
    contexts, syn, nonsyn = _synthetic_opportunities()
    mu_taus = _synthetic_mu_taus()

    syn_rates = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=syn,
        contexts_by_gene=contexts,
    )
    merged = compute_mu_g_per_tumor(
        mu_taus=mu_taus, contexts_by_gene=contexts
    )

    opportunity_share = syn.values.sum() / (
        syn.values.sum() + nonsyn.values.sum()
    )
    rate_share = syn_rates.values.sum() / merged.values.sum()
    assert np.isclose(rate_share, opportunity_share, atol=0.05)


def test_channel_mismatched_gene_universe_raises():
    """The denominator is a sum over genes, so a mismatched gene set
    silently rescales every rate -- must raise instead."""
    contexts, syn, _ = _synthetic_opportunities()
    mu_taus = _synthetic_mu_taus()
    with pytest.raises(ValueError, match="same genes"):
        compute_mu_g_channel_per_tumor(
            mu_taus=mu_taus,
            channel_contexts_by_gene=syn.iloc[:2],
            contexts_by_gene=contexts,
        )


@pytest.mark.parametrize("tau_independent", [False, True])
def test_channel_separate_per_tau_sums_to_aggregate(tau_independent):
    """separate_per_tau=True must decompose the same total, as it
    does for the merged function."""
    contexts, syn, _ = _synthetic_opportunities()
    mu_taus = _synthetic_mu_taus()

    aggregate = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=syn,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_independent,
    )
    per_tau = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=syn,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_independent,
        separate_per_tau=True,
    )
    pd.testing.assert_frame_equal(
        sum(per_tau.values()), aggregate, check_exact=False
    )


def _mutation_db():
    """Three genes × three tumors, with both silent and non-silent."""
    rows = [
        ("ENSG_A", "T1", "Silent"),
        ("ENSG_A", "T1", "Missense_Mutation"),
        ("ENSG_A", "T2", "Silent"),
        ("ENSG_B", "T2", "Missense_Mutation"),
        ("ENSG_B", "T3", "Nonsense_Mutation"),
        ("ENSG_C", "T3", "Silent"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "ensembl_gene_id",
            "Tumor_Sample_Barcode",
            "Variant_Classification",
        ],
    ).assign(variant="X p.A1B")


def test_silent_and_non_silent_presence_partition_any():
    """A gene/sample is present overall iff it is present in at least
    one channel -- the OR the split model replaces."""
    db = _mutation_db()
    any_present = compute_genes_present(db)
    silent = compute_genes_present(db, scope="silent")
    non_silent = compute_genes_present(db, scope="non-silent")

    silent = silent.reindex(
        index=any_present.index,
        columns=any_present.columns,
        fill_value=0,
    )
    non_silent = non_silent.reindex(
        index=any_present.index,
        columns=any_present.columns,
        fill_value=0,
    )
    pd.testing.assert_frame_equal(
        ((silent + non_silent) > 0).astype(int),
        any_present,
        check_dtype=False,
    )


def test_compute_gene_presence_silent(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = _mutation_db()
    assert not dataset.has_silent_presence()

    dataset.compute_gene_presence_silent()
    assert dataset.has_silent_presence()
    silent = dataset.genes_present_silent
    assert silent.loc["ENSG_A", "T1"] == 1
    # ENSG_B's only mutations are non-silent
    assert (
        "ENSG_B" not in silent.index
        or silent.loc["ENSG_B"].sum() == 0
    )


def test_genes_present_silent_raises_before_computing(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    with pytest.raises(ValueError, match="Silent gene presence"):
        _ = dataset.genes_present_silent


def test_silent_presence_survives_save_load(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path / "src")
    dataset._mutation_db = _mutation_db()
    dataset.compute_gene_presence_silent()

    save_dir = tmp_path / "saved"
    dataset.save_dataset(save_dir)
    loaded = MutationDataset.load_dataset(save_dir)

    pd.testing.assert_frame_equal(
        loaded.genes_present_silent,
        dataset.genes_present_silent,
        check_dtype=False,
    )


def _model_with_channels(tmp_path, tau_independent=False):
    """A minimal Model wired up for the two-channel fit."""
    contexts, syn, nonsyn = _synthetic_opportunities()
    db = _mutation_db()

    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = db
    dataset._contexts_by_gene = contexts
    dataset._contexts_by_gene_syn = syn
    dataset._contexts_by_gene_nonsyn = nonsyn
    dataset.compute_gene_presence()
    dataset.compute_gene_presence_non_silent()
    dataset.compute_gene_presence_silent()

    cov_matrix = pd.DataFrame(
        {"cov1": [0.5, -0.5, 0.1]}, index=_GENES
    )
    model = Model(dataset, cov_matrix)
    model._mu_taus = _synthetic_mu_taus()
    model.compute_base_mus(prob_g_tau_tau_independent=tau_independent)
    model.compute_channel_base_mus()
    return model


def test_compute_channel_base_mus_sums_to_base_mus(tmp_path):
    model = _model_with_channels(tmp_path)
    pd.testing.assert_frame_equal(
        model.base_mus_syn + model.base_mus_nonsyn,
        model.base_mus,
        check_exact=False,
    )
    assert model.has_channel_base_mus()


def test_compute_channel_base_mus_inherits_tau_variant(tmp_path):
    """Left at None, the channels must use whatever compute_base_mus
    used -- otherwise they would not sum back to it."""
    model = _model_with_channels(tmp_path, tau_independent=True)
    pd.testing.assert_frame_equal(
        model.base_mus_syn + model.base_mus_nonsyn,
        model.base_mus,
        check_exact=False,
    )


def test_compute_channel_base_mus_requires_split_tables(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = _mutation_db()
    contexts, _, _ = _synthetic_opportunities()
    dataset._contexts_by_gene = contexts

    model = Model(dataset)
    model._mu_taus = _synthetic_mu_taus()
    model.compute_base_mus()
    with pytest.raises(ValueError, match="Consequence-split"):
        model.compute_channel_base_mus()


def test_channel_base_mus_properties_raise_before_computing(
    tmp_path,
):
    dataset = MutationDataset(location_maf_files=tmp_path)
    model = Model(dataset)
    assert not model.has_channel_base_mus()
    with pytest.raises(ValueError, match="Synonymous-channel"):
        _ = model.base_mus_syn
    with pytest.raises(ValueError, match="Non-synonymous-channel"):
        _ = model.base_mus_nonsyn


def test_compute_channel_mu_gs_rejects_unknown_channel(tmp_path):
    model = _model_with_channels(tmp_path)
    with pytest.raises(ValueError, match="Unknown channel"):
        model.compute_channel_mu_gs("silent")


def test_estimate_channel_cov_effects_map(tmp_path):
    """The joint fit runs and returns one shared coefficient vector
    (intercept + 1 covariate), not one per channel."""
    model = _model_with_channels(tmp_path)
    result = model.estimate_channel_cov_effects(sample="MAP")
    assert result.shape == (2,)
    assert np.all(np.isfinite(result))
    # downstream recomputation happened
    assert model._mu_gs is not None
    assert model.passenger_genes_r2 is not None


def test_estimate_channel_cov_effects_drivers_toggle(tmp_path):
    """include_drivers only changes the silent channel's gene set;
    both settings must fit and report the passenger-channel count."""
    model_on = _model_with_channels(tmp_path)
    model_on.estimate_channel_cov_effects(
        sample="MAP", include_drivers=True
    )
    model_off = _model_with_channels(tmp_path)
    model_off.estimate_channel_cov_effects(
        sample="MAP", include_drivers=False
    )
    assert (
        model_on.n_in_cov_effects_estimation
        == model_off.n_in_cov_effects_estimation
    )


def test_estimate_channel_cov_effects_requires_channel_base_mus(
    tmp_path,
):
    contexts, syn, nonsyn = _synthetic_opportunities()
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = _mutation_db()
    dataset._contexts_by_gene = contexts
    dataset._contexts_by_gene_syn = syn
    dataset._contexts_by_gene_nonsyn = nonsyn
    dataset.compute_gene_presence()

    model = Model(
        dataset, pd.DataFrame({"cov1": [0.1] * 3}, index=_GENES)
    )
    model._mu_taus = _synthetic_mu_taus()
    model.compute_base_mus()
    with pytest.raises(ValueError, match="Channel baseline rates"):
        model.estimate_channel_cov_effects(sample="MAP")


def test_estimate_channel_cov_effects_requires_silent_presence(
    tmp_path,
):
    model = _model_with_channels(tmp_path)
    model.dataset._genes_present_silent = None
    with pytest.raises(ValueError, match="Silent gene presence"):
        model.estimate_channel_cov_effects(sample="MAP")


def test_passenger_genes_r2_non_silent_target(tmp_path):
    """The non-silent target must score against the non-silent
    channel's own rates and be stored separately from the 'any' one,
    so no caller can read one as the other."""
    model = _model_with_channels(tmp_path)
    model.estimate_channel_cov_effects(sample="MAP")

    r2_any = model.estimate_passenger_genes_r2(target="any")
    r2_non_silent = model.estimate_passenger_genes_r2(
        target="non_silent"
    )

    assert np.isfinite(r2_non_silent)
    assert model.passenger_genes_r2 == r2_any
    assert model.passenger_genes_r2_non_silent == r2_non_silent
    # the two attributes are independent storage, not one overwritten
    assert (
        model.passenger_genes_r2
        != model.passenger_genes_r2_non_silent
    )


def test_passenger_genes_r2_default_target_unchanged(tmp_path):
    """Regression guard: adding `target` must not change what a
    no-argument call does."""
    model = _model_with_channels(tmp_path)
    model.estimate_channel_cov_effects(sample="MAP")
    assert model.estimate_passenger_genes_r2() == (
        model.estimate_passenger_genes_r2(target="any")
    )


def test_passenger_genes_r2_rejects_unknown_target(tmp_path):
    model = _model_with_channels(tmp_path)
    model.estimate_channel_cov_effects(sample="MAP")
    with pytest.raises(ValueError, match="Unknown target"):
        model.estimate_passenger_genes_r2(target="silent")


def test_passenger_genes_r2_rejects_nan_rates(tmp_path):
    """A NaN rate fails silently under pandas' NaN-skipping .sum()
    (the gene is scored as expecting 0), so it must raise instead --
    this is what a cov_matrix with NaN rows for genes lacking
    covariate data produces, and it cost 0.08 of R² on COAD with no
    error anywhere."""
    model = _model_with_channels(tmp_path)
    model.estimate_channel_cov_effects(sample="MAP")
    model._mu_gs.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN rate"):
        model.estimate_passenger_genes_r2()
