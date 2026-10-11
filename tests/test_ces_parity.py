"""cancereffectsizeR parity: the indel hold-out and the exact likelihood.

With dispersion, the mu cut and the absence floor off, and the clip
switched off, the gamma likelihood is cancereffectsizeR's
``sswm_lik`` (``single_variant_si_objectives.R``), whose MLE
``ces_variant`` finds with L-BFGS-B on [1e-3, 1e9]. The same-gene
hold-out reads every call in the gene there, indels included; here
indels come from :attr:`MutationDataset.indel_db`.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from sigmutsel.estimate_gammas import (
    estimate_gamma_from_mus,
    log_absence_probability,
)
from sigmutsel.load_maf_files import read_coding_indels
from sigmutsel.models import MutationDataset
from tests.test_multi_base import _capture, _variant_model

# --- reading the indels --------------------------------------------------

_MAF_HEADER = [
    "Hugo_Symbol",
    "Gene",
    "Chromosome",
    "Start_Position",
    "Variant_Classification",
    "Variant_Type",
    "Tumor_Sample_Barcode",
]


def _write_maf(path, rows):
    pd.DataFrame(rows, columns=_MAF_HEADER).to_csv(
        path, sep="\t", index=False
    )


def test_only_non_silent_coding_indels_are_read(tmp_path):
    _write_maf(
        tmp_path / "a.maf",
        [
            ["EGFR", "ENSG1", "chr7", 1, "In_Frame_Del", "DEL", "T1"],
            [
                "TP53",
                "ENSG2",
                "chr17",
                2,
                "Frame_Shift_Ins",
                "INS",
                "T2",
            ],
            [
                "TP53",
                "ENSG2",
                "chr17",
                3,
                "Missense_Mutation",
                "SNP",
                "T3",
            ],
            ["TP53", "ENSG2", "chr17", 4, "3'UTR", "DEL", "T4"],
            [
                "Unknown",
                "ENSG3",
                "chr1",
                5,
                "Frame_Shift_Del",
                "DEL",
                "T5",
            ],
            ["KRAS", "ENSG4", "chr12", 6, "Splice_Site", "DEL", "T6"],
        ],
    )
    db = read_coding_indels(tmp_path, location_gene_set=None)
    assert list(db["Tumor_Sample_Barcode"]) == ["T1", "T2", "T6"]
    assert list(db["ensembl_gene_id"]) == ["ENSG1", "ENSG2", "ENSG4"]


def test_build_keeps_only_the_datasets_tumors(tmp_path):
    _write_maf(
        tmp_path / "a.maf",
        [
            ["EGFR", "G1", "chr7", 1, "In_Frame_Del", "DEL", "T2"],
            ["EGFR", "G1", "chr7", 1, "In_Frame_Del", "DEL", "T9"],
        ],
    )
    ds = MutationDataset(location_maf_files=str(tmp_path))
    ds._mutation_db = pd.DataFrame(
        {"Tumor_Sample_Barcode": ["T1", "T2"]}
    )
    ds.build_indel_db(location_gene_set=None)
    assert list(ds.indel_hits("G1")) == ["T2"]
    assert len(ds.indel_hits("G2")) == 0


def test_indel_table_survives_save_and_load(tmp_path):
    ds = MutationDataset(location_maf_files=str(tmp_path / "mafs"))
    ds._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["T1"],
            "ensembl_gene_id": ["G1"],
            "gene": ["A"],
            "Variant_Classification": ["Missense_Mutation"],
        }
    )
    ds.indel_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["T3"],
            "ensembl_gene_id": ["G1"],
            "Variant_Classification": ["Frame_Shift_Del"],
        }
    )
    ds.save_dataset(tmp_path / "ds", overwrite=True)
    back = MutationDataset.load_dataset(tmp_path / "ds")
    assert list(back.indel_hits("G1")) == ["T3"]


# --- the hold-out ---------------------------------------------------------


def _with_indels(model, hits):
    model.dataset.indel_hits = lambda gene_id: pd.Index(
        hits.get(gene_id, [])
    )
    return model


def test_variant_absent_set_loses_the_indel_tumors(monkeypatch):
    captured = _capture(monkeypatch)
    model = _with_indels(_variant_model({}), {"G1": ["T2"]})
    model._estimate_gamma_variant("VAR1", store=False)
    attrs = captured["result"].posterior.attrs
    assert captured["yes"] == ["T1"]
    assert captured["no"] == ["T3", "T4"]
    assert attrs["n_tumors_held_out_indel"] == 1
    assert attrs["n_tumors_held_out"] == 1


def test_a_tumor_held_out_twice_counts_once(monkeypatch):
    captured = _capture(monkeypatch)
    model = _with_indels(
        _variant_model({"G1": ["T3"]}), {"G1": ["T3"]}
    )
    model._estimate_gamma_variant("VAR1", store=False)
    attrs = captured["result"].posterior.attrs
    assert captured["no"] == ["T2", "T4"]
    assert attrs["n_tumors_held_out_multi_base"] == 1
    assert attrs["n_tumors_held_out_indel"] == 0


def test_the_indel_hold_out_can_be_switched_off(monkeypatch):
    captured = _capture(monkeypatch)
    model = _with_indels(_variant_model({}), {"G1": ["T2"]})
    model._estimate_gamma_variant(
        "VAR1", store=False, hold_out_indel=False
    )
    assert captured["no"] == ["T2", "T3", "T4"]


# --- the exact likelihood -------------------------------------------------


def test_log_absence_is_the_link_on_the_log_scale():
    mu = np.array([1e-4, 0.5, 50.0])
    np.testing.assert_allclose(
        log_absence_probability(10.0, mu).eval(), -10.0 * mu
    )
    u = 1e-3
    np.testing.assert_allclose(
        log_absence_probability(10.0, mu, absence_floor=u).eval(),
        np.log(u + (1 - u) * np.exp(-10.0 * mu)),
    )


def _sswm_mle(rates_with, rates_without):
    """cancereffectsizeR's estimate: ``sswm_lik`` maximized as
    ``ces_variant`` does (L-BFGS-B from 1 on [1e-3, 1e9])."""

    def neg_loglik(g):
        g = g[0]
        return -(
            -np.sum(g * rates_without)
            + np.sum(np.log(1 - np.exp(-g * rates_with)))
        )

    fit = minimize(
        neg_loglik,
        x0=[1.0],
        method="L-BFGS-B",
        bounds=[(1e-3, 1e9)],
    )
    return float(fit.x[0])


def _map(mus_yes, mus_no, **kwargs):
    out = estimate_gamma_from_mus(
        mus_yes,
        mus_no,
        draws=1,
        upper_bound_prior=1e6,
        absence_floor=0.0,
        **kwargs,
    )
    return float(out["gamma"])


_RNG = np.random.default_rng(7)
_YES = _RNG.lognormal(np.log(1e-3), 1.0, 12)
_NO = _RNG.lognormal(np.log(1e-3), 1.0, 300)


def test_everything_off_reproduces_the_ces_estimate():
    ces = _sswm_mle(_YES, _NO)
    ours = _map(_YES, _NO, presence_clip=False)
    assert ours == pytest.approx(ces, rel=1e-3)
    # Away from the clip, the clip changes nothing.
    assert _map(_YES, _NO) == pytest.approx(ces, rel=1e-3)


def test_the_clip_is_what_separates_a_high_rate_absent_tumor():
    """A common hotspot and a hypermutator lacking it: at the CES
    estimate its gamma * mu passes 27.6, so the clip caps its
    evidence and gamma rises; with the clip off the estimate is
    CES's again."""
    rng = np.random.default_rng(7)
    yes = rng.lognormal(np.log(1e-3), 1.0, 150)
    no = np.append(rng.lognormal(np.log(1e-3), 1.0, 150), 0.3)
    ces = _sswm_mle(yes, no)
    assert ces * 0.3 > 27.6
    assert _map(yes, no, presence_clip=False) == pytest.approx(
        ces, rel=1e-3
    )
    assert _map(yes, no) > 1.5 * ces


def test_no_clip_with_a_custom_link_is_refused():
    with pytest.raises(ValueError, match="presence_model"):
        estimate_gamma_from_mus(
            _YES,
            _NO,
            draws=1,
            presence_clip=False,
            presence_model=lambda g, m, **k: 1 - np.exp(-g * m),
        )


def test_the_exact_likelihood_samples_and_says_so():
    out = estimate_gamma_from_mus(
        _YES,
        _NO,
        draws=800,
        chains=2,
        burn=300,
        absence_floor=0.0,
        presence_clip=False,
    )
    assert out.posterior.attrs["presence_clip"] == 0
    mean = float(out.posterior["gamma"].mean())
    assert mean == pytest.approx(_sswm_mle(_YES, _NO), rel=0.3)


def test_build_keeps_the_mafs_gene_ids_by_default(
    tmp_path, monkeypatch
):
    # The hold-out matches indels to genes by id, so the indel table
    # must carry the ids mutation_db does; generate_mutation_db keeps
    # the MAF's by default, and so must build_indel_db.
    import sigmutsel.load_maf_files as lmf

    seen = {}

    def fake(location, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            {
                "Tumor_Sample_Barcode": ["T1"],
                "ensembl_gene_id": ["G1"],
                "Variant_Classification": ["In_Frame_Del"],
            }
        )

    monkeypatch.setattr(lmf, "read_coding_indels", fake)
    ds = MutationDataset(location_maf_files=str(tmp_path))
    ds.build_indel_db()
    assert seen == {"location_gene_set": None}
    seen.clear()
    ds.build_indel_db(location_gene_set="default")
    assert seen == {}  # the reader's own default, the HGNC table
