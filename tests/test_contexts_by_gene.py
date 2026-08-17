"""Tests for contexts_by_gene.py and MutationDataset's gene_universe."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sigmutsel.contexts_by_gene import compute_contexts_by_gene
from sigmutsel.models import Model, MutationDataset

# A tiny synthetic Ensembl-style FASTA: two genes, each with two
# transcripts of different lengths (the longer one should win).
_FASTA_CONTENT = """\
>ENST001 gene:ENSG00000000001.5 gene_biotype:protein_coding
ACGTACGTACGT
>ENST002 gene:ENSG00000000001.5 gene_biotype:protein_coding
ACGTACGTACGTACGTACGT
>ENST003 gene:ENSG00000000002.3 gene_biotype:protein_coding
TTTTAAAACCCCGGGG
"""


@pytest.fixture
def fasta_path(tmp_path):
    p = tmp_path / "test.fa"
    p.write_text(_FASTA_CONTENT)
    return p


def test_compute_contexts_by_gene_no_restriction(fasta_path):
    df = compute_contexts_by_gene(fasta_path, restrict_to_db=None)
    assert set(df.index) == {"ENSG00000000001", "ENSG00000000002"}
    # 32 pyrimidine-centered contexts
    assert df.shape[1] == 32


def test_compute_contexts_by_gene_picks_longest_transcript(
    fasta_path,
):
    df = compute_contexts_by_gene(fasta_path, restrict_to_db=None)
    # ENSG1's longer transcript (20bp) should be used, not the 12bp one:
    # total context counts (row sum) reflect len(seq) - 2 valid triplets.
    total_g1 = df.loc["ENSG00000000001"].sum()
    assert total_g1 == 18  # 20 - 2


def test_compute_contexts_by_gene_iterable_restriction(fasta_path):
    df = compute_contexts_by_gene(
        fasta_path, restrict_to_db=["ENSG00000000002"]
    )
    assert set(df.index) == {"ENSG00000000002"}


def test_compute_contexts_by_gene_dataframe_restriction(fasta_path):
    mutation_db = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001", "ENSG00000000002"],
            "variant": ["TP53 p.R175H", None],
        }
    )
    df = compute_contexts_by_gene(
        fasta_path, restrict_to_db=mutation_db
    )
    # only the row with a non-null variant survives the DataFrame-mode mask
    assert set(df.index) == {"ENSG00000000001"}


def _dataset_with_mutation_db(tmp_path, mutation_db):
    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._mutation_db = mutation_db
    return dataset


def test_generate_contexts_by_gene_own_cohort_matches_direct_call(
    tmp_path, fasta_path
):
    mutation_db = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001"],
            "variant": ["GENE1 p.X1Y"],
        }
    )
    dataset = _dataset_with_mutation_db(tmp_path, mutation_db)
    result = dataset.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )
    direct = compute_contexts_by_gene(
        fasta_path, restrict_to_db=mutation_db
    )
    pd.testing.assert_frame_equal(result, direct)
    assert dataset._contexts_by_gene_gene_universe == "own_cohort"


def test_generate_contexts_by_gene_wes_target_is_superset(
    tmp_path, fasta_path
):
    mutation_db = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001"],
            "variant": ["GENE1 p.X1Y"],
        }
    )
    dataset_own = _dataset_with_mutation_db(tmp_path, mutation_db)
    own_result = dataset_own.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )

    dataset_wes = _dataset_with_mutation_db(tmp_path, mutation_db)
    with patch(
        "sigmutsel.wes_target.get_wes_target_gene_ids",
        return_value={"ENSG00000000002"},
    ):
        wes_result = dataset_wes.generate_contexts_by_gene(
            fastas=fasta_path, gene_universe="wes_target"
        )

    assert set(own_result.index) <= set(wes_result.index)
    assert set(wes_result.index) == {
        "ENSG00000000001",
        "ENSG00000000002",
    }
    assert dataset_wes._contexts_by_gene_gene_universe == "wes_target"


def test_generate_contexts_by_gene_wes_target_keeps_genes_absent_from_wes_set(
    tmp_path, fasta_path
):
    """The core regression case: a gene with real mutation evidence
    but absent from the (mocked) WES-target set must still survive,
    the same way ENSG00000275395 (a real GENCODE-v19-vs-v38
    annotation gap, not a capture gap) did for TGCT/COAD/etc."""
    mutation_db = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001"],
            "variant": ["GENE1 p.X1Y"],
        }
    )
    dataset = _dataset_with_mutation_db(tmp_path, mutation_db)
    with patch(
        "sigmutsel.wes_target.get_wes_target_gene_ids",
        return_value=set(),  # ENSG00000000001 absent from WES-target
    ):
        result = dataset.generate_contexts_by_gene(
            fastas=fasta_path, gene_universe="wes_target"
        )
    assert "ENSG00000000001" in result.index


def test_generate_contexts_by_gene_invalid_universe_raises(
    tmp_path, fasta_path
):
    mutation_db = pd.DataFrame(
        {"ensembl_gene_id": ["ENSG00000000001"], "variant": ["x"]}
    )
    dataset = _dataset_with_mutation_db(tmp_path, mutation_db)
    with pytest.raises(ValueError, match="gene_universe"):
        dataset.generate_contexts_by_gene(
            fastas=fasta_path, gene_universe="bogus"
        )


def test_save_load_dataset_records_gene_universe(
    tmp_path, fasta_path
):
    mutation_db = pd.DataFrame(
        {"ensembl_gene_id": ["ENSG00000000001"], "variant": ["x"]}
    )
    dataset = _dataset_with_mutation_db(tmp_path / "src", mutation_db)
    dataset.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )

    save_dir = tmp_path / "saved"
    dataset.save_dataset(save_dir)

    loaded = MutationDataset.load_dataset(save_dir)
    assert loaded._contexts_by_gene_gene_universe == "own_cohort"


def test_load_dataset_defaults_gene_universe_for_old_manifests(
    tmp_path, fasta_path
):
    """A manifest saved before this field existed must default to
    'own_cohort' -- the behavior it actually used, not an unknown
    state."""
    mutation_db = pd.DataFrame(
        {"ensembl_gene_id": ["ENSG00000000001"], "variant": ["x"]}
    )
    dataset = _dataset_with_mutation_db(tmp_path / "src", mutation_db)
    dataset.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )
    save_dir = tmp_path / "saved"
    dataset.save_dataset(save_dir)

    # simulate a pre-existing manifest that predates this field
    import json

    manifest_path = save_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["contexts_by_gene_gene_universe"]
    manifest_path.write_text(json.dumps(manifest))

    loaded = MutationDataset.load_dataset(save_dir)
    assert loaded._contexts_by_gene_gene_universe == "own_cohort"


def test_passenger_genes_r2_handles_genes_absent_from_genes_present(
    tmp_path,
):
    """Regression test: under gene_universe="wes_target", mu_gs can
    include genes never observed as mutated in this cohort -- absent
    from genes_present's crosstab by construction, not an error
    state. estimate_passenger_genes_r2() must treat that as 0
    observed presence, not raise KeyError (caught during COAD's
    rollout, 2026-08-17)."""
    tumors = ["T1", "T2", "T3"]
    # ENSG_OBSERVED has real mutation evidence; ENSG_WES_ONLY was
    # added by the WES-target union but never mutated in this cohort.
    genes_present = pd.DataFrame(
        [[1, 0, 1]], index=["ENSG_OBSERVED"], columns=tumors
    )

    dataset = MutationDataset(location_maf_files=tmp_path)
    dataset._genes_present = genes_present

    mu_gs = pd.DataFrame(
        [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        index=["ENSG_OBSERVED", "ENSG_WES_ONLY"],
        columns=tumors,
    )

    model = Model.__new__(Model)
    model.dataset = dataset
    model._mu_gs = mu_gs
    model._passenger_genes_r2 = None

    r2 = model.estimate_passenger_genes_r2()
    assert np.isfinite(r2)
    assert model.passenger_genes_r2 == r2
