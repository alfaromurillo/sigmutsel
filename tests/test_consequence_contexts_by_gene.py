"""Tests for consequence_contexts_by_gene.py and its call site.

The load-bearing property here is the sum identity

    syn[τ] + nonsyn[τ] == contexts_by_gene[extract_context(τ)]

for each of the 3 SBS types τ sharing a context, since that is what
makes ``p_gτ^(syn) + p_gτ^(nonsyn) == p_gτ`` exact downstream. It is
checked against an independent run of ``compute_contexts_by_gene``
(not against a re-derivation of the same walk) plus a hand-computed
split on a sequence small enough to enumerate by eye.
"""

import pandas as pd
import pytest

from sigmutsel.consequence_contexts_by_gene import (
    _SYNONYMOUS_ALTS,
    compute_consequence_contexts_by_gene,
)
from sigmutsel.constants import (
    canonical_types_order,
    extract_context,
)
from sigmutsel.contexts_by_gene import compute_contexts_by_gene
from sigmutsel.models import MutationDataset

# Two genes, one with a longer second transcript (which must win, the
# same way compute_contexts_by_gene picks it), plus deliberately
# awkward material: a length that is not a multiple of 3 (truncated
# final codon) and an N (ambiguous context *and* ambiguous codon).
_FASTA_CONTENT = """\
>ENST001 gene:ENSG00000000001.5 gene_biotype:protein_coding
ATGACGTCATTAGGGCCC
>ENST002 gene:ENSG00000000001.5 gene_biotype:protein_coding
ATGACGTCATTAGGGCCCAAATTTCCCGGGTAATT
>ENST003 gene:ENSG00000000002.3 gene_biotype:protein_coding
ATGCCNTTAGGGAAACCC
"""

# A single 6-base gene, short enough to enumerate the whole split by
# hand (see test_hand_computed_split).
_TINY_FASTA_CONTENT = """\
>ENST100 gene:ENSG00000000100.1 gene_biotype:protein_coding
ATGTTA
"""


@pytest.fixture
def fasta_path(tmp_path):
    p = tmp_path / "test.fa"
    p.write_text(_FASTA_CONTENT)
    return p


@pytest.fixture
def tiny_fasta_path(tmp_path):
    p = tmp_path / "tiny.fa"
    p.write_text(_TINY_FASTA_CONTENT)
    return p


def test_shape_and_columns(fasta_path):
    syn, nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=None
    )
    for df in (syn, nonsyn):
        assert list(df.columns) == canonical_types_order
        assert df.shape == (2, 96)
        assert set(df.index) == {
            "ENSG00000000001",
            "ENSG00000000002",
        }
        assert (df.values >= 0).all()


def test_sum_reproduces_contexts_by_gene(fasta_path):
    """The identity the whole mechanism rests on.

    Each of the 3 types sharing a context must split exactly that
    context's count, as computed independently by
    compute_contexts_by_gene.
    """
    syn, nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=None
    )
    contexts = compute_contexts_by_gene(
        fasta_path, restrict_to_db=None
    )

    total = syn + nonsyn
    broadcast = contexts[
        [extract_context(t) for t in canonical_types_order]
    ]
    broadcast.columns = canonical_types_order

    pd.testing.assert_frame_equal(total, broadcast, check_dtype=False)


def test_sum_reproduces_contexts_by_gene_under_restriction(
    fasta_path,
):
    """Restriction must not desynchronise the two tables."""
    restrict = ["ENSG00000000002"]
    syn, nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=restrict
    )
    contexts = compute_contexts_by_gene(
        fasta_path, restrict_to_db=restrict
    )
    assert list(syn.index) == list(contexts.index)

    total = syn + nonsyn
    for sbs_type in canonical_types_order:
        assert (
            total[sbs_type] == contexts[extract_context(sbs_type)]
        ).all()


def test_picks_longest_transcript(fasta_path):
    """Same transcript choice as compute_contexts_by_gene: the 35bp
    ENST002, not the 18bp ENST001 (35 - 2 = 33 counted positions, each
    contributing one opportunity to each of 3 types)."""
    syn, nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=None
    )
    total = (
        syn.loc["ENSG00000000001"].sum()
        + nonsyn.loc["ENSG00000000001"].sum()
    )
    assert total == 33 * 3


def test_ambiguous_context_positions_are_skipped(fasta_path):
    """ENSG2's N kills 3 context windows, exactly as it does for
    compute_contexts_by_gene -- those positions belong to neither
    channel rather than defaulting into one."""
    syn, nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=None
    )
    total = (
        syn.loc["ENSG00000000002"].sum()
        + nonsyn.loc["ENSG00000000002"].sum()
    )
    # 18bp sequence -> 16 candidate centres, 3 of which have the N in
    # their window (centres 4, 5 and 6, 0-based).
    assert total == (16 - 3) * 3


def test_hand_computed_split(tiny_fasta_path):
    """ATG TTA (Met-Leu), enumerated by hand.

    Counted centres are 1..4 (0 and 5 have no full context window):

    * centre 1 (T of ATG, codon ATG position 1): ATG codes Met, and
      no substitution at any position of ATG is synonymous.
    * centre 2 (G of ATG, position 2): likewise none synonymous.
    * centre 3 (T of TTA, codon TTA position 0): TTA is Leu, and
      CTA is Leu too -> T>C is synonymous, T>A (ATA, Ile) and T>G
      (GTA, Val) are not.
    * centre 4 (T of TTA, position 1): TAA/TCA/TGA are stop/Ser/stop,
      none Leu -> no synonymous alternate.

    So exactly one of the 12 opportunities is synonymous: T>C at
    centre 3. Its *context* is the sequence window seq[2:5] = 'GTT'
    (the flanking bases, which do not have to lie in the same codon),
    and the centre is a pyrimidine, so the canonical type is read off
    directly: G[T>C]T.
    """
    syn, nonsyn = compute_consequence_contexts_by_gene(
        tiny_fasta_path, restrict_to_db=None
    )
    gene = "ENSG00000000100"

    assert syn.loc[gene].sum() == 1
    assert syn.loc[gene, "G[T>C]T"] == 1
    assert nonsyn.loc[gene].sum() == 11


def test_purine_centred_positions_are_strand_collapsed(
    tiny_fasta_path,
):
    """Centre 2 of ATGTTA is a G, so its opportunities must be
    recorded on the reverse-complement (pyrimidine-centred) type,
    exactly as compute_contexts_by_gene collapses the context: the
    window is 'TGT', whose reverse complement is 'ACA', and the
    coding-strand G>A becomes C>T there."""
    syn, nonsyn = compute_consequence_contexts_by_gene(
        tiny_fasta_path, restrict_to_db=None
    )
    gene = "ENSG00000000100"
    total = syn + nonsyn
    # window seq[1:4] == 'TGT'; reverse complement 'ACA'
    assert total.loc[gene, "A[C>T]A"] == 1
    assert total.loc[gene, "A[C>A]A"] == 1
    assert total.loc[gene, "A[C>G]A"] == 1


def test_synonymous_alts_table_matches_genetic_code():
    """Spot-check the precomputed codon table against known facts of
    the standard genetic code."""
    # Leucine's 4-fold degenerate third position (CTN all Leu).
    assert _SYNONYMOUS_ALTS["CTA"][2] == frozenset("CGT")
    # Methionine and tryptophan have no synonymous alternates at all.
    for position in range(3):
        assert _SYNONYMOUS_ALTS["ATG"][position] == frozenset()
        assert _SYNONYMOUS_ALTS["TGG"][position] == frozenset()
    # Stop-to-stop counts as synonymous: TAA -> TAG and TGA.
    assert "G" in _SYNONYMOUS_ALTS["TAA"][2]
    assert "G" in _SYNONYMOUS_ALTS["TAA"][1]


def _dataset_with_mutation_db(tmp_path, mutation_db, **kwargs):
    dataset = MutationDataset(location_maf_files=tmp_path, **kwargs)
    dataset._mutation_db = mutation_db
    return dataset


@pytest.fixture
def mutation_db():
    return pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSG00000000001"],
            "variant": ["GENE1 p.X1Y"],
        }
    )


def test_generate_matches_direct_call(
    tmp_path, fasta_path, mutation_db
):
    dataset = _dataset_with_mutation_db(tmp_path, mutation_db)
    syn, nonsyn = dataset.generate_consequence_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )
    direct_syn, direct_nonsyn = compute_consequence_contexts_by_gene(
        fasta_path, restrict_to_db=mutation_db
    )
    pd.testing.assert_frame_equal(syn, direct_syn)
    pd.testing.assert_frame_equal(nonsyn, direct_nonsyn)
    pd.testing.assert_frame_equal(
        dataset.contexts_by_gene_syn, direct_syn
    )
    pd.testing.assert_frame_equal(
        dataset.contexts_by_gene_nonsyn, direct_nonsyn
    )
    assert dataset.has_consequence_contexts_by_gene()


def test_generate_rejects_non_sbs_signature_class(
    tmp_path, fasta_path, mutation_db
):
    """The SBS gate: 'synonymous' has no meaning for ID/DBS/CN/SV, so
    the mechanism must refuse rather than quietly produce a table."""
    dataset = _dataset_with_mutation_db(
        tmp_path, mutation_db, signature_class="ID"
    )
    with pytest.raises(ValueError, match="SBS-only"):
        dataset.generate_consequence_contexts_by_gene(
            fastas=fasta_path
        )


def test_generate_requires_mutation_db(tmp_path, fasta_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    with pytest.raises(ValueError, match="Mutation database"):
        dataset.generate_consequence_contexts_by_gene(
            fastas=fasta_path
        )


def test_generate_rejects_unknown_gene_universe(
    tmp_path, fasta_path, mutation_db
):
    dataset = _dataset_with_mutation_db(tmp_path, mutation_db)
    with pytest.raises(ValueError, match="gene_universe"):
        dataset.generate_consequence_contexts_by_gene(
            fastas=fasta_path, gene_universe="bogus"
        )


def test_properties_raise_before_generation(tmp_path):
    dataset = MutationDataset(location_maf_files=tmp_path)
    assert not dataset.has_consequence_contexts_by_gene()
    with pytest.raises(ValueError, match="Synonymous"):
        _ = dataset.contexts_by_gene_syn
    with pytest.raises(ValueError, match="Non-synonymous"):
        _ = dataset.contexts_by_gene_nonsyn


def test_save_load_round_trip(tmp_path, fasta_path, mutation_db):
    dataset = _dataset_with_mutation_db(tmp_path / "src", mutation_db)
    dataset.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )
    dataset.generate_consequence_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )

    save_dir = tmp_path / "saved"
    dataset.save_dataset(save_dir)
    loaded = MutationDataset.load_dataset(save_dir)

    pd.testing.assert_frame_equal(
        loaded.contexts_by_gene_syn,
        dataset.contexts_by_gene_syn,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        loaded.contexts_by_gene_nonsyn,
        dataset.contexts_by_gene_nonsyn,
        check_dtype=False,
    )


def test_save_load_without_split_tables_still_works(
    tmp_path, fasta_path, mutation_db
):
    """The split tables are optional: a dataset that never generated
    them must save and load exactly as before."""
    dataset = _dataset_with_mutation_db(tmp_path / "src", mutation_db)
    dataset.generate_contexts_by_gene(
        fastas=fasta_path, gene_universe="own_cohort"
    )
    save_dir = tmp_path / "saved"
    dataset.save_dataset(save_dir)

    loaded = MutationDataset.load_dataset(save_dir)
    assert not loaded.has_consequence_contexts_by_gene()
    assert not (save_dir / "contexts_by_gene_syn.csv").exists()
