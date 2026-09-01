"""Regression tests for Model's ``include_other`` signature-grouping
handling.

Bug this covers (found while building a signature-stratified
covariate-effect pilot on SKCM, see the `mutation_rates` project's
TODO.md): getting a genuine Nth "rest" group required calling
``model.aggregate_signatures(selection, include_other=True)``
manually, since ``Model.__init__`` had no ``include_other`` parameter
at all. But that manual call left ``self._auto_signature_selection``
unset, and the later automatic step ``estimate_cov_effects()`` ->
``compute_mu_ms()`` -> ``_compute_mu_g_taus()`` reconstructs a
per-type signature grouping from ``self._mu_taus`` to match the
already-aggregated ``self._base_mus`` -- that reconstruction never
passed ``include_other=True`` through, so it silently dropped the
"other" group and crashed downstream with a coefficient/group-count
shape mismatch.

Fixed by: (1) adding an ``include_other`` parameter to
``Model.__init__``/``_apply_auto_configuration`` so the constructor
path can request a genuine "other" bucket directly, and (2) making
``_compute_mu_g_taus``'s reconstruction infer ``include_other``
correctly in both the constructor-recorded case
(``self._auto_include_other``) and the fallback case (an ``"other"``
key present in ``self._base_mus``, e.g. after a manual
``aggregate_signatures`` call or a ``load_model()`` round-trip).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd

from sigmutsel.models import Model

SIGNATURES = ("SBS7a", "SBS7b", "SBS7c", "SBS5", "SBS1")
TYPES = ["A[C>A]A", "A[C>A]C"]
TUMORS = ["T1", "T2"]


def _mu_taus_fixture():
    rng = np.random.RandomState(0)
    return {
        sig: pd.DataFrame(
            rng.rand(len(TYPES), len(TUMORS)),
            index=TYPES,
            columns=TUMORS,
        )
        for sig in SIGNATURES
    }


def _fake_compute_mu_g_per_tumor(mu_taus, **kwargs):
    """Stand-in for the real gene-level expansion. Doesn't compute
    anything meaningful -- just echoes back a dict shaped by
    mu_taus' own keys, so the caller can assert on which signature
    groups actually reached this point."""
    tau = kwargs["separate_per_tau"][0]
    genes = ["ENSG_A", "ENSG_B"]
    return {
        sigma: {tau: pd.DataFrame(0.0, index=genes, columns=TUMORS)}
        for sigma in mu_taus
    }


def _minimal_model():
    model = Model.__new__(Model)
    model._mu_taus = _mu_taus_fixture()
    model._prob_g_tau_tau_independent = True
    model.dataset = type("DS", (), {"contexts_by_gene": None})()
    model.cov_effects = None
    model.cov_matrix = None
    return model


def _reconstructed_signature_keys(model):
    """Call the private reconstruction path with use_cov_effects=False
    (skips the covariate-scaling branch entirely, so no cov_effects/
    cov_matrix needed) and return the signature keys the mocked
    compute_mu_g_per_tumor was actually called with."""
    with patch(
        "sigmutsel.estimate_mus.compute_mu_g_per_tumor",
        side_effect=_fake_compute_mu_g_per_tumor,
    ) as mocked:
        model._compute_mu_g_taus(use_cov_effects=False)
    return set(mocked.call_args.kwargs["mu_taus"].keys())


def test_manual_aggregate_signatures_include_other_reconstructs_correctly():
    """The original crash scenario: aggregate_signatures(...,
    include_other=True) called manually (not via the constructor)."""
    model = _minimal_model()
    model._base_mus = dict(model._mu_taus)  # pre-aggregation state
    model.aggregate_signatures(
        ["SBS7a", "SBS7b", "SBS7c"], include_other=True
    )
    assert set(model._base_mus.keys()) == {
        "SBS7a",
        "SBS7b",
        "SBS7c",
        "other",
    }

    reconstructed = _reconstructed_signature_keys(model)
    assert reconstructed == set(model._base_mus.keys())


def test_constructor_recorded_include_other_reconstructs_correctly():
    """The constructor path: _auto_signature_selection/
    _auto_include_other set as Model.__init__ now does, base_mus
    already aggregated (as _apply_auto_configuration would leave
    it)."""
    model = _minimal_model()
    model._auto_signature_selection = ["SBS7a", "SBS7b", "SBS7c"]
    model._auto_include_other = True
    model._base_mus = {
        "SBS7a": model._mu_taus["SBS7a"],
        "SBS7b": model._mu_taus["SBS7b"],
        "SBS7c": model._mu_taus["SBS7c"],
        "other": model._mu_taus["SBS5"] + model._mu_taus["SBS1"],
    }

    reconstructed = _reconstructed_signature_keys(model)
    assert reconstructed == {"SBS7a", "SBS7b", "SBS7c", "other"}


def test_no_include_other_still_drops_unlisted_signatures():
    """No-regression check: without include_other, aggregate_signatures
    still drops unlisted signatures entirely (documented, unaffected
    by this fix) -- reconstruction should match that, not silently
    add an "other" group nobody asked for."""
    model = _minimal_model()
    model._base_mus = dict(model._mu_taus)
    model.aggregate_signatures(["SBS7a", "SBS7b", "SBS7c"])
    assert set(model._base_mus.keys()) == {"SBS7a", "SBS7b", "SBS7c"}

    reconstructed = _reconstructed_signature_keys(model)
    assert reconstructed == {"SBS7a", "SBS7b", "SBS7c"}


def _dummy_dataset():
    """A non-str/Path stand-in so __post_init__'s
    MutationDataset.load_dataset(str) branch is skipped."""
    return object()


def test_init_stores_include_other():
    """__init__ unconditionally calls _apply_auto_configuration() at
    the end, so its compute_mu_taus/compute_base_mus/
    aggregate_signatures calls need stubbing here too -- this test
    only cares that the two _auto_* fields get stored correctly."""
    with (
        patch.object(
            Model, "compute_mu_taus", lambda self, **kw: None
        ),
        patch.object(
            Model, "compute_base_mus", lambda self, **kw: None
        ),
        patch.object(
            Model, "aggregate_signatures", lambda self, *a, **kw: None
        ),
    ):
        model = Model.__new__(Model)
        model.__init__(
            dataset=_dummy_dataset(),
            signature_selection=["SBS7a"],
            include_other=True,
        )
    assert model._auto_signature_selection == ["SBS7a"]
    assert model._auto_include_other is True


def test_init_defaults_include_other_false():
    model = Model.__new__(Model)
    model.__init__(dataset=_dummy_dataset())
    assert model._auto_include_other is False


def test_apply_auto_configuration_forwards_include_other():
    """Bypass __init__ entirely (it would run the real
    compute_mu_taus/compute_base_mus pipeline) -- set the recorded
    _auto_* state directly, matching what __init__ would have stored,
    and call _apply_auto_configuration() in isolation."""
    model = Model.__new__(Model)
    model._auto_mu_taus_kwargs = {}
    model._auto_cov_effects_per_sigma = None
    model._auto_prob_g_tau_tau_independent = None
    model._auto_signature_selection = ["SBS7a"]
    model._auto_include_other = True
    model._mu_taus = "already computed"  # skip compute_mu_taus
    model._base_mus = {"placeholder": None}  # skip compute_base_mus
    captured = {}
    model.aggregate_signatures = lambda *a, **kw: captured.update(
        selection=a[0], include_other=kw.get("include_other")
    )

    model._apply_auto_configuration()

    assert captured == {"selection": ["SBS7a"], "include_other": True}
