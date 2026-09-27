"""Tests for the channel universe (sigmutsel.channel_universe).

A synthetic two-gene annotation -- one gene per strand, each with one
coding intron and a separate stop_codon feature -- is enough to check
the spec's invariants exactly:

- the four channels partition every gene's sites;
- a site's rate is the same in every channel (apart from delta);
- every single-nucleotide route to a protein change is enumerated;
- calls are labelled on the chosen transcript, on either strand;
- the dataset round trip keeps the channel columns.
"""

import numpy as np
import pandas as pd
import pytest

from sigmutsel.channel_universe import (
    CHANNELS,
    TranscriptModels,
    _routes,
    classify_calls,
    compute_channel_opportunity,
    model_calls,
    read_cds_fasta,
    read_coding_transcripts,
    select_transcripts,
)
from sigmutsel.constants import canonical_types_order, extract_context
from sigmutsel.estimate_mus import (
    compute_mu_g_channel_per_tumor,
    variant_site_denominators,
)
from sigmutsel.liftover import (
    in_intervals,
    lift_positions,
    merged_bed_intervals,
    read_chain,
)

# Gene A, + strand, chr1: CDS 101-109 | intron 110-199 | CDS 200-208,
# stop 209-211. Codons: ATG TTT CGA | GGG TTT GCA TAA.
SEQ_A = "ATGTTTCGA" + "GGGTTTGCA" + "TAA"
# Gene B, - strand, chr2: in transcript order CDS 1100-1108 (reverse),
# intron 1012-1099, CDS 1003-1011, stop 1000-1002.
# Codons: ATG CCC TGG | AAA CGC TTT TAG.
SEQ_B = "ATGCCCTGG" + "AAACGCTTT" + "TAG"

GTF_LINES = [
    # gene A: MANE transcript plus a longer, untagged one
    (
        "chr1",
        "transcript",
        101,
        211,
        "+",
        ".",
        (
            'gene_id "ENSGA.1"; transcript_id "ENSTA1.1"; gene_type '
            '"protein_coding"; gene_name "GA"; tag "MANE_Select"; '
            'tag "Ensembl_canonical";'
        ),
    ),
    ("chr1", "CDS", 101, 109, "+", "0", 'transcript_id "ENSTA1.1";'),
    ("chr1", "CDS", 200, 208, "+", "0", 'transcript_id "ENSTA1.1";'),
    (
        "chr1",
        "stop_codon",
        209,
        211,
        "+",
        "0",
        'transcript_id "ENSTA1.1";',
    ),
    (
        "chr1",
        "transcript",
        101,
        300,
        "+",
        ".",
        (
            'gene_id "ENSGA.1"; transcript_id "ENSTA2.1"; gene_type '
            '"protein_coding"; gene_name "GA";'
        ),
    ),
    ("chr1", "CDS", 101, 130, "+", "0", 'transcript_id "ENSTA2.1";'),
    # gene B: canonical only, minus strand
    (
        "chr2",
        "transcript",
        1000,
        1108,
        "-",
        ".",
        (
            'gene_id "ENSGB.1"; transcript_id "ENSTB1.1"; gene_type '
            '"protein_coding"; gene_name "GB"; tag "Ensembl_canonical";'
        ),
    ),
    (
        "chr2",
        "CDS",
        1100,
        1108,
        "-",
        "0",
        'transcript_id "ENSTB1.1";',
    ),
    (
        "chr2",
        "CDS",
        1003,
        1011,
        "-",
        "0",
        'transcript_id "ENSTB1.1";',
    ),
    (
        "chr2",
        "stop_codon",
        1000,
        1002,
        "-",
        "0",
        'transcript_id "ENSTB1.1";',
    ),
]


@pytest.fixture
def models(tmp_path):
    gtf = tmp_path / "toy.gtf"
    gtf.write_text(
        "".join(
            "\t".join(
                [
                    c,
                    "TEST",
                    kind,
                    str(s),
                    str(e),
                    ".",
                    strand,
                    phase,
                    attrs,
                ]
            )
            + "\n"
            for c, kind, s, e, strand, phase, attrs in GTF_LINES
        )
    )
    fasta = tmp_path / "toy.fa"
    # ENSTA2 has a sequence of the right length too, so only the rule
    # order can keep it from being chosen.
    fasta.write_text(
        f">ENSTA1.1 cds\n{SEQ_A}\n"
        f">ENSTA2.1 cds\n{'ATG' + 'GCT' * 9}\n"
        f">ENSTB1.1 cds\n{SEQ_B}\n"
    )
    transcripts, features = read_coding_transcripts(gtf)
    records = read_cds_fasta(fasta)
    selection = select_transcripts(transcripts, features, records)
    return TranscriptModels.build(selection, features, records)


def test_rule_order_and_record(models):
    sel = models.selection
    assert sel.loc["ENSGA", "transcript_id"] == "ENSTA1.1"
    assert sel.loc["ENSGA", "rule"] == "mane_select"
    assert sel.loc["ENSGB", "rule"] == "ensembl_canonical"
    assert sel["fasta_version_match"].all()


def test_genomic_map_and_splice_sites(models):
    g = models.gene_index["ENSGB"]
    block = slice(
        models.offsets[g], models.offsets[g] + models.lengths[g]
    )
    gpos = models.gpos[block]
    # transcript order on the minus strand walks down the genome
    assert gpos[0] == 1108 and gpos[8] == 1100
    assert gpos[9] == 1011 and gpos[-1] == 1000
    sp = models.splice.set_index(["gene", "kind"])["gpos"]
    assert sp[(g, "d1")] == 1099 and sp[(g, "d2")] == 1098
    assert sp[(g, "a2")] == 1013 and sp[(g, "a1")] == 1012
    a = models.gene_index["ENSGA"]
    assert sp[(a, "d1")] == 110 and sp[(a, "a1")] == 199


def test_channels_partition_every_site(models):
    tables = compute_channel_opportunity(models)
    total = sum(tables[c] for c in CHANNELS)
    for tau in canonical_types_order:
        np.testing.assert_allclose(
            total[tau], tables["contexts"][extract_context(tau)]
        )
    # Coding positions counted: centres 1..len-2 of each block; each
    # offers 3 opportunities. Splice: 4 sites per intron, 3 each.
    coding = sum(len(s) - 2 for s in (SEQ_A, SEQ_B)) * 3
    splice = 2 * 4 * 3
    grand = sum(tables[c].to_numpy().sum() for c in CHANNELS)
    assert grand == pytest.approx(coding + splice)
    assert tables["spl"].to_numpy().sum() == pytest.approx(splice)


def test_known_consequences(models):
    tables = compute_channel_opportunity(models)
    # GA's TAA stop: TAA>TAG and TAA>TGA stay stops -> syn;
    # the start codon never becomes a stop.
    assert tables["non"].loc["ENSGA"].sum() > 0
    # CGA (Arg) -> TGA is nonsense: C>T at context GCG on the coding
    # strand (GA ... TTT CGA|GGG: left T? codon starts after TTT).
    assert tables["non"].loc["ENSGA", "T[C>T]G"] >= 1


def test_site_rate_is_the_same_in_every_channel(models):
    tables = compute_channel_opportunity(models)
    contexts = tables["contexts"]
    rng = np.random.default_rng(0)
    mu_taus = pd.DataFrame(
        rng.gamma(2.0, 1.0, size=(3, 96)),
        index=["t1", "t2", "t3"],
        columns=canonical_types_order,
    )
    site_rates = {}
    for channel in CHANNELS:
        rates = compute_mu_g_channel_per_tumor(
            mu_taus, tables[channel], contexts, separate_per_tau=True
        )
        n = variant_site_denominators(
            contexts, opportunity_by_type=tables[channel]
        )
        site_rates[channel] = {
            tau: rates[tau].div(n[tau], axis=0) for tau in rates
        }
    denominators = contexts.sum(axis=0)
    for tau in canonical_types_order:
        expected = mu_taus[tau] / denominators[extract_context(tau)]
        for channel in CHANNELS:
            frame = site_rates[channel][tau]
            for gene in frame.index:
                if tables[channel].at[gene, tau] > 0:
                    np.testing.assert_allclose(
                        frame.loc[gene].to_numpy(),
                        expected.to_numpy(),
                    )


def test_every_route_is_enumerated(models):
    a = models.gene_index["ENSGA"]
    start = models.offsets[a]
    # Codon 2 of GA is TTT (Phe). Leu is reached three ways: CTT at
    # position 1, TTA and TTG at position 3.
    routes = _routes(models, start + 3, "L", "mis", None)
    assert sorted(routes) == sorted(["G[T>C]T", "T[T>A]C", "T[T>G]C"])
    # Codon 4 is GGG (Gly), whose third position is four-fold
    # degenerate: three synonymous routes, all at that position.
    routes = _routes(models, start + 9, None, "syn", None)
    assert len(routes) == 3
    # Codon 3 is CGA (Arg): TGA is the only route to a stop.
    assert _routes(models, start + 6, "*", "non", None) == ["T[C>T]G"]


def _calls(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "ensembl_gene_id",
            "gene",
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
            "type",
        ],
    )


def test_classify_calls_both_strands(models):
    db = _calls(
        [
            # GA codon 2 TTT, position 2: T>C gives TCT (Ser)
            ("ENSGA", "GA", "chr1", 105, "T", "C", "T[T>C]T"),
            # GA donor +1 (G on the + strand)
            ("ENSGA", "GA", "chr1", 110, "G", "A", "A[C>T]C"),
            # GA intron, well inside: out of the universe
            ("ENSGA", "GA", "chr1", 150, "A", "G", "T[T>C]A"),
            # GB start codon A (genomic T on the minus strand) -> G: Val
            ("ENSGB", "GB", "chr2", 1108, "T", "C", "C[T>C]G"),
            # GB acceptor -1 (G of AG on the coding strand = C genomic)
            ("ENSGB", "GB", "chr2", 1012, "C", "T", "T[C>T]T"),
            # wrong reference allele
            ("ENSGA", "GA", "chr1", 105, "G", "C", "T[T>C]T"),
            # gene with no transcript
            ("ENSGZ", "GZ", "chr3", 5, "A", "G", "T[T>C]A"),
        ]
    )
    out = classify_calls(db, models)
    assert out["channel"].tolist()[:5] == [
        "mis",
        "spl",
        None,
        "mis",
        "spl",
    ]
    assert out["variant_label"].iloc[0] == "GA p.F2S"
    assert out["variant_label"].iloc[1] == "GA c.9+1G>A"
    assert out["variant_label"].iloc[3] == "GB p.M1V"
    assert out["variant_label"].iloc[4] == "GB c.10-1G>A"
    assert out["universe_reason"].tolist()[2:] == [
        "noncoding",
        "coding",
        "splice",
        "ref_mismatch",
        "no_transcript",
    ]
    assert out["in_universe"].tolist() == [
        True,
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_model_calls_scopes():
    db = pd.DataFrame(
        {
            "in_universe": [True, True, True, False],
            "channel": ["syn", "mis", "spl", "syn"],
            "Variant_Classification": ["Silent"] * 4,
        }
    )
    assert len(model_calls(db)) == 3
    assert model_calls(db, "silent")["channel"].tolist() == ["syn"]
    assert model_calls(db, "non-silent")["channel"].tolist() == [
        "mis",
        "spl",
    ]
    legacy = db.drop(columns=["in_universe", "channel"])
    assert len(model_calls(legacy, "silent")) == 4


def test_dataset_round_trip_keeps_channel(tmp_path):
    from sigmutsel.models import MutationDataset

    db = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": ["t1", "t2", "t2"],
            "gene": ["GA", "GA", "GB"],
            "ensembl_gene_id": ["ENSGA", "ENSGA", "ENSGB"],
            "variant": ["GA p.F2S", "GA c.9+1G>A", "GB intron"],
            "type": ["T[T>C]T", "A[C>T]C", "T[T>C]A"],
            "Variant_Classification": [
                "Missense_Mutation",
                "Splice_Site",
                "Intron",
            ],
            "channel": ["mis", "spl", None],
            "in_universe": [True, True, False],
            "universe_reason": ["coding", "splice", "noncoding"],
            "routes": ["T[T>C]T", "A[C>T]C", None],
        }
    )
    dataset = MutationDataset(str(tmp_path / "mafs"))
    dataset.mutation_db = db
    dataset._channel_universe = {
        "territory": "mc3",
        "splice_padding": 0,
    }
    dataset.save_dataset(tmp_path / "ds", overwrite=True)
    loaded = MutationDataset.load_dataset(tmp_path / "ds")
    pd.testing.assert_series_equal(
        loaded.mutation_db["channel"], db["channel"]
    )
    assert loaded.mutation_db["in_universe"].dtype == bool
    assert loaded.has_channel_universe()
    assert loaded.channel_universe["territory"] == "mc3"
    assert len(loaded.model_db) == 2


def test_point_liftover_and_membership(tmp_path):
    chain = tmp_path / "toy.chain"
    # chrA 0-100 maps to chrB 1000-1100 on +, then a 10-base gap in
    # the source; chrA 110-160 maps to chrC on the - strand.
    chain.write_text(
        "chain 100 chrA 1000 + 0 100 chrB 5000 + 1000 1100 1\n"
        "100\n\n"
        "chain 50 chrA 1000 + 110 160 chrC 500 - 0 50 2\n"
        "50\n\n"
    )
    blocks = read_chain(chain)
    chrom, pos = lift_positions(
        blocks, ["chrA", "chrA", "chrA", "chrA"], [1, 100, 105, 111]
    )
    assert chrom.tolist() == ["chrB", "chrB", None, "chrC"]
    # 1-based 111 is 0-based 110, the first base of the - block:
    # query 0 on the - strand, i.e. forward 500 - 1 - 0 = 499.
    assert pos.tolist() == [1001, 1100, -1, 500]
    bed = pd.DataFrame(
        {"chrom": ["chrB"], "start": [1000], "end": [1050]}
    )
    inside = in_intervals(merged_bed_intervals(bed), chrom, pos)
    assert inside.tolist() == [True, False, False, False]
    padded = in_intervals(
        merged_bed_intervals(bed, padding=50), chrom, pos
    )
    assert padded.tolist() == [True, True, False, False]


def test_route_rates_sum_and_compounds_add(models):
    """Item 15 and compound variants together.

    A variant's rate is the sum over its routes of the channel's site
    rate, a repeated type counting once per site; a compound's rate is
    the sum of its members'. Checked against hand arithmetic on the
    synthetic tables.
    """
    from sigmutsel.compound_variants import compound_rates
    from sigmutsel.estimate_mus import compute_mu_m_per_tumor

    tables = compute_channel_opportunity(models)
    contexts = tables["contexts"]
    nonsyn = tables["mis"] + tables["non"] + tables["spl"]
    rng = np.random.default_rng(1)
    mu_taus = pd.DataFrame(
        rng.gamma(2.0, 1.0, size=(2, 96)),
        index=["t1", "t2"],
        columns=canonical_types_order,
    )
    gene_rates = compute_mu_g_channel_per_tumor(
        mu_taus, nonsyn, contexts, separate_per_tau=True
    )
    variants = pd.DataFrame(
        {
            "ensembl_gene_id": ["ENSGA", "ENSGA"],
            "gene": ["GA", "GA"],
            # GA's donor +1 window is AGT on the coding strand, so
            # G>A there is A[C>T]T.
            "mut_types": [
                ["G[T>C]T", "T[T>A]C", "T[T>A]C"],
                "A[C>T]T",
            ],
        },
        index=["GA p.F2L", "GA c.9+1G>A"],
    )
    mu_ms = compute_mu_m_per_tumor(
        variants, gene_rates, contexts, opportunity_by_type=nonsyn
    )
    denominators = contexts.sum(axis=0)

    def site(tau):
        return mu_taus[tau] / denominators[extract_context(tau)]

    expected_mis = site("G[T>C]T") + 2 * site("T[T>A]C")
    np.testing.assert_allclose(mu_ms.loc["GA p.F2L"], expected_mis)
    np.testing.assert_allclose(
        mu_ms.loc["GA c.9+1G>A"], site("A[C>T]T")
    )
    np.testing.assert_allclose(
        compound_rates(mu_ms, list(variants.index)),
        expected_mis + site("A[C>T]T"),
    )
