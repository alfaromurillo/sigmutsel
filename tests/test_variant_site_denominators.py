"""A variant's rate is its gene's rate spread over the right sites.

``compute_mu_m_per_tumor`` divides a gene's type-tau rate by the
number of type-tau sites that rate was built from. For a consequence
channel (the non-synonymous rate a variant is scored against) that is
the channel's own opportunity count, not every position with tau's
context: otherwise each site is scaled by its gene's non-synonymous
fraction for tau, and a site's rate depends on how many *other* sites
in the gene happen to be synonymous.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.constants import (
    canonical_contexts_order,
    canonical_types_order,
    extract_context,
)
from sigmutsel.estimate_mus import (
    compute_mu_g_channel_per_tumor,
    compute_mu_g_per_tumor,
    compute_mu_m_per_tumor,
    variant_site_denominators,
)

TAU = canonical_types_order[0]
CTX = extract_context(TAU)


def _tables():
    """Two genes with identical contexts but different non-syn splits."""
    rng = np.random.default_rng(0)
    genes = ["G1", "G2", "G3"]
    contexts = pd.DataFrame(
        rng.integers(20, 60, size=(3, 32)),
        index=genes,
        columns=canonical_contexts_order,
    )
    contexts.loc["G2"] = contexts.loc["G1"]
    # Non-synonymous share of each type's positions: G1 all of them,
    # G2 half, G3 random.
    frac = pd.DataFrame(
        rng.uniform(0.3, 1.0, size=(3, 96)),
        index=genes,
        columns=canonical_types_order,
    )
    frac.loc["G1"] = 1.0
    frac.loc["G2"] = 0.5
    full = pd.DataFrame(
        {
            t: contexts[extract_context(t)]
            for t in canonical_types_order
        }
    )
    nonsyn = (full * frac).round().astype(int)
    mu_taus = pd.DataFrame(
        rng.uniform(0.1, 2.0, size=(4, 96)),
        index=[f"T{i}" for i in range(4)],
        columns=canonical_types_order,
    )
    return contexts, nonsyn, mu_taus


def _variants(genes):
    return pd.DataFrame(
        {"ensembl_gene_id": genes, "mut_types": [TAU] * len(genes)},
        index=[f"V_{g}" for g in genes],
    )


@pytest.mark.parametrize("tau_indep", [False, True])
def test_site_rate_does_not_depend_on_the_genes_synonymous_sites(
    tau_indep,
):
    contexts, nonsyn, mu_taus = _tables()
    mu_nonsyn = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=nonsyn,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_indep,
        separate_per_tau=True,
    )
    mu_m = compute_mu_m_per_tumor(
        _variants(["G1", "G2"]),
        mu_nonsyn,
        contexts,
        prob_g_tau_tau_independent=tau_indep,
        opportunity_by_type=nonsyn,
    )
    # G1 and G2 share every context count; only their synonymous share
    # differs. A site's rate must not see that.
    np.testing.assert_allclose(
        mu_m.loc["V_G1"].to_numpy(), mu_m.loc["V_G2"].to_numpy()
    )


@pytest.mark.parametrize("tau_indep", [False, True])
def test_channel_site_rate_equals_merged_site_rate(tau_indep):
    """A non-syn site and any site have the same per-site rate (before
    delta and covariates): one opportunity is one opportunity."""
    contexts, nonsyn, mu_taus = _tables()
    mu_nonsyn = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=nonsyn,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_indep,
        separate_per_tau=True,
    )
    mu_merged = compute_mu_g_per_tumor(
        mu_taus=mu_taus,
        contexts_by_gene=contexts,
        prob_g_tau_tau_independent=tau_indep,
        separate_per_tau=True,
    )
    variants = _variants(["G1", "G2", "G3"])
    channel = compute_mu_m_per_tumor(
        variants,
        mu_nonsyn,
        contexts,
        prob_g_tau_tau_independent=tau_indep,
        opportunity_by_type=nonsyn,
    )
    merged = compute_mu_m_per_tumor(
        variants,
        mu_merged,
        contexts,
        prob_g_tau_tau_independent=tau_indep,
    )
    np.testing.assert_allclose(channel.to_numpy(), merged.to_numpy())


def test_channel_sites_sum_back_to_the_channel_gene_rate():
    contexts, nonsyn, mu_taus = _tables()
    mu_nonsyn = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=nonsyn,
        contexts_by_gene=contexts,
        separate_per_tau=True,
    )
    mu_m = compute_mu_m_per_tumor(
        _variants(["G3"]),
        mu_nonsyn,
        contexts,
        opportunity_by_type=nonsyn,
    )
    np.testing.assert_allclose(
        mu_m.loc["V_G3"].to_numpy() * nonsyn.at["G3", TAU],
        mu_nonsyn[TAU].loc["G3"].to_numpy(),
    )


def test_the_old_divisor_scaled_sites_by_the_nonsyn_fraction():
    """Regression check on the bug: channel rate / n_{g,c(tau)}."""
    contexts, nonsyn, mu_taus = _tables()
    mu_nonsyn = compute_mu_g_channel_per_tumor(
        mu_taus=mu_taus,
        channel_contexts_by_gene=nonsyn,
        contexts_by_gene=contexts,
        separate_per_tau=True,
    )
    variants = _variants(["G2"])
    fixed = compute_mu_m_per_tumor(
        variants, mu_nonsyn, contexts, opportunity_by_type=nonsyn
    )
    old = compute_mu_m_per_tumor(variants, mu_nonsyn, contexts)
    frac = nonsyn.at["G2", TAU] / contexts.at["G2", CTX]
    np.testing.assert_allclose(
        old.to_numpy(), fixed.to_numpy() * frac
    )


def test_denominators_default_is_the_context_count():
    contexts, _, _ = _tables()
    d = variant_site_denominators(contexts)
    assert list(d.columns) == canonical_types_order
    assert d.at["G3", TAU] == contexts.at["G3", CTX]


def test_tau_independent_channel_denominator_reduces_to_merged():
    """With every opportunity in the channel, the channel form must
    equal the merged form (3 opportunities per position)."""
    contexts, _, _ = _tables()
    full = pd.DataFrame(
        {
            t: contexts[extract_context(t)]
            for t in canonical_types_order
        }
    )
    np.testing.assert_allclose(
        variant_site_denominators(
            contexts, full, prob_g_tau_tau_independent=True
        ).to_numpy(),
        variant_site_denominators(
            contexts, prob_g_tau_tau_independent=True
        ).to_numpy(),
    )
