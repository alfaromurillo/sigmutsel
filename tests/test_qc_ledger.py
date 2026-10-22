"""The QC ledger: what each quality-control step removes, per tumor."""

import json

import pandas as pd

from sigmutsel.load_maf_files import process_single_maf
from sigmutsel.models import MutationDataset
from sigmutsel.qc_ledger import (
    LEDGER_COLUMNS,
    STEP_ORDER,
    empty_ledger,
    totals,
)


def _snv(tumor, pos, **overrides):
    row = {
        "Tumor_Sample_Barcode": tumor,
        "Chromosome": "chr1",
        "Variant_Classification": "Missense_Mutation",
        "Variant_Type": "SNP",
        "Start_Position": pos,
        "Hugo_Symbol": "TP53",
        "Gene": "ENSG00000141510",
        "HGVSp_Short": "p.R175H",
        "CONTEXT": "AAAAACAAAAA",
        "Reference_Allele": "C",
        "Tumor_Seq_Allele2": "T",
        "t_depth": 80,
        "t_ref_count": 60,
        "t_alt_count": 20,
    }
    row.update(overrides)
    return row


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def _maf_rows(tumor):
    """One tumor's MAF rows, one or more per step."""
    return [
        _snv(tumor, 1000),  # clean
        _snv(tumor, 9000),  # clean
        _snv(tumor, 2000, Variant_Type="DEL", Reference_Allele="CA"),
        _snv(tumor, 3000),  # exact duplicate pair: one kept
        _snv(tumor, 3000),
        _snv(tumor, 4000),  # doublet: set aside
        _snv(tumor, 4001),
        _snv(tumor, 5000, Chromosome="chrUn_KI270742v1"),
        _snv(tumor, 6000, Hugo_Symbol="Unknown"),  # compaction drops
    ]


def _by_step(ledger, tumor):
    sub = ledger[ledger["tumor"] == tumor]
    return dict(zip(sub["step"], sub["n"]))


def test_process_single_maf_returns_the_files_ledger(tmp_path):
    maf = _write(tmp_path / "a.maf", _maf_rows("T1"))

    out, ledger = process_single_maf(
        maf, qc_mode=True, keep_multi_base=True, return_ledger=True
    )

    assert list(ledger.columns) == LEDGER_COLUMNS
    steps = _by_step(ledger, "T1")
    assert steps["maf_rows"] == 9
    assert steps["not_snv"] == 1
    assert steps["duplicate_record"] == 1
    assert steps["merged_into_dbs_variant"] == 2
    assert steps["invalid_chromosome"] == 1
    assert steps["compact_failed"] == 1
    clean = int(out["multi_base"].isna().sum())
    assert clean == 3  # 1000, 9000 and the kept duplicate
    # maf_rows is the total; the other steps and the clean calls
    # account for every row.
    assert steps["maf_rows"] == clean + sum(
        n for s, n in steps.items() if s != "maf_rows"
    )
    assert (ledger["level"] == "calls").all()
    assert ledger["order"].tolist() == sorted(ledger["order"])


def test_return_ledger_needs_qc_mode(tmp_path):
    maf = _write(tmp_path / "a.maf", _maf_rows("T1"))
    try:
        process_single_maf(maf, return_ledger=True)
    except ValueError as exc:
        assert "qc_mode" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_generate_mutation_db_keeps_a_ledger_that_round_trips(
    tmp_path,
):
    maf_dir = tmp_path / "mafs"
    maf_dir.mkdir()
    _write(maf_dir / "a.maf", _maf_rows("T1"))
    _write(maf_dir / "b.maf", _maf_rows("T2")[:3])

    ds = MutationDataset(
        location_maf_files=maf_dir, signature_class="SBS"
    )
    ds.generate_mutation_db(qc_mode=True, channel_universe=False)

    ledger = ds.qc_ledger
    assert set(ledger["tumor"]) == {"T1", "T2"}
    t2 = _by_step(ledger, "T2")
    assert t2 == {"maf_rows": 3, "not_snv": 1}
    # The ledger accounts for every call kept in mutation_db.
    removed = ledger[ledger["step"] != "maf_rows"]["n"].sum()
    rows = ledger.loc[ledger["step"] == "maf_rows", "n"].sum()
    aside = len(ds.multi_base_db)
    assert rows - removed == len(ds.mutation_db)
    assert (
        ledger.loc[
            ledger["step"] == "merged_into_dbs_variant", "n"
        ].sum()
        == aside
    )

    ds.save_dataset(tmp_path / "ds", overwrite=True)
    manifest = json.loads(
        (tmp_path / "ds" / "dataset_manifest.json").read_text()
    )
    assert (
        manifest["files"]["qc_ledger"]["filename"]
        == "qc_ledger.parquet"
    )
    back = MutationDataset.load_dataset(tmp_path / "ds")
    pd.testing.assert_frame_equal(
        back.qc_ledger.reset_index(drop=True),
        ledger.reset_index(drop=True),
    )


def test_without_qc_mode_there_is_no_ledger(tmp_path):
    maf_dir = tmp_path / "mafs"
    maf_dir.mkdir()
    _write(maf_dir / "a.maf", _maf_rows("T1")[:2])

    ds = MutationDataset(
        location_maf_files=maf_dir, signature_class="SBS"
    )
    ds.generate_mutation_db(channel_universe=False)

    assert ds.qc_ledger is None
    ds.save_dataset(tmp_path / "ds", overwrite=True)
    back = MutationDataset.load_dataset(tmp_path / "ds")
    assert back.qc_ledger is None


def test_record_qc_steps_replaces_a_rerun_step():
    ds = MutationDataset(
        location_maf_files=".", signature_class="SBS"
    )
    first = pd.DataFrame(
        {
            "tumor": ["T1", "T2"],
            "level": "tumors",
            "step": "sample_type",
            "order": 20,
            "n": 1,
        }
    )
    ds.record_qc_steps(first)
    ds.record_qc_steps(first.iloc[:1])

    assert ds.qc_ledger["tumor"].tolist() == ["T1"]
    assert totals(ds.qc_ledger)["n"].tolist() == [1]


def test_two_pass_records_its_tumors_and_calls(tmp_path, monkeypatch):
    ds = MutationDataset(
        location_maf_files=tmp_path, signature_class="SBS"
    )
    ds._mutation_db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["S1"] * 3 + ["S2"] * 9,
            "type": ["A[C>A]A"] * 3 + ["A[C>A]A"] * 8 + ["A[C>A]C"],
            "Variant_Classification": ["Missense_Mutation"] * 12,
        }
    )
    ds._qc_ledger = empty_ledger()
    monkeypatch.setattr(
        "sigmutsel.constants.ARTIFACT_SIGNATURES", ["SIG_ARTIFACT"]
    )
    monkeypatch.setattr(
        "sigmutsel.constants.TREATMENT_ASSOCIATED_SIGNATURES",
        ["SIG_TREAT"],
    )
    sig_matrix = pd.DataFrame(
        {
            "MutationType": ["A[C>A]A", "A[C>A]C"],
            "SIG_A": [1.0, 0.0],
            "SIG_TREAT": [1.0, 0.0],
            "SIG_ARTIFACT": [0.0, 1.0],
        }
    ).set_index("MutationType")
    # S1: all treatment load (dropped whole); S2: one artifact call.
    pass_a = pd.DataFrame(
        {
            "SIG_A": [0, 8],
            "SIG_TREAT": [6, 0],
            "SIG_ARTIFACT": [0, 1],
        },
        index=["S1", "S2"],
    )

    def fake_pass_a(self, *args, **kwargs):
        self._sig_assignments = pass_a
        self._signature_matrix = sig_matrix
        return pass_a

    monkeypatch.setattr(
        MutationDataset, "run_signature_decomposition", fake_pass_a
    )
    monkeypatch.setattr(
        "sigmutsel.signature_decomposition.signature_decomposition",
        lambda **kwargs: pd.DataFrame({"SIG_A": [8]}, index=["S2"]),
    )

    ds.run_two_pass_signature_decomposition(
        treatment_load_threshold=0.2, min_burden_for_diagnostics=5
    )
    # A rerun on the cleaned set adds only what it removes itself.
    ds.run_two_pass_signature_decomposition(
        treatment_load_threshold=0.2, min_burden_for_diagnostics=5
    )

    ledger = ds.qc_ledger
    assert _by_step(ledger, "S1") == {
        "treatment_signature_load": 1,
        "treatment_signature_sample": 3,
    }
    assert (
        ledger.loc[
            (ledger["tumor"] == "S1")
            & (ledger["step"] == "treatment_signature_load"),
            "level",
        ].item()
        == "tumors"
    )
    assert _by_step(ledger, "S2") == {
        "artifact_signature_mutation": 1
    }
    assert set(ledger["order"]) <= set(STEP_ORDER.values())
