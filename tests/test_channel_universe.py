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
        "germline_mask_af": 0.001,
    }
    dataset.save_dataset(tmp_path / "ds", overwrite=True)
    loaded = MutationDataset.load_dataset(tmp_path / "ds")
    pd.testing.assert_series_equal(
        loaded.mutation_db["channel"], db["channel"]
    )
    assert loaded.mutation_db["in_universe"].dtype == bool
    assert loaded.has_channel_universe()
    assert loaded.channel_universe["territory"] == "mc3"
    assert loaded.germline_mask_af == 0.001
    assert len(loaded.model_db) == 2


def test_type_opportunity_only_under_a_mask():
    from sigmutsel.models import MutationDataset

    dataset = MutationDataset("unused")
    syn = pd.DataFrame(
        1.0, index=["g1", "g2"], columns=canonical_types_order
    )
    dataset._contexts_by_gene_syn = syn
    dataset._contexts_by_gene_nonsyn = 2 * syn
    dataset._channel_universe = {"territory": "mc3"}
    assert dataset.type_opportunity is None
    dataset._channel_universe["germline_mask_af"] = 0.001
    np.testing.assert_allclose(
        dataset.type_opportunity.to_numpy(), 3.0
    )


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


def test_splice_flank_composition_reads_both_strands(
    models, tmp_path
):
    """Synthetic genome in SigProfiler's tsb format (byte % 4 = base).

    GA (+ strand) has intron 110-199: G T A ... C A G at 110-112 and
    197-199. GB (- strand) has intron 1012-1099; on the coding strand
    its donor is GTG and its acceptor TAG, i.e. genomic C A C at
    1099-1097 and A T C at 1014-1012.
    """
    from sigmutsel.channel_universe import splice_flank_composition

    code = {"A": 0, "C": 1, "G": 2, "T": 3}

    def write(chrom, length, bases):
        genome = np.full(length, 16, dtype=np.uint8)  # N
        for pos, b in bases.items():
            genome[pos - 1] = code[b]
        genome.tofile(tmp_path / f"{chrom}.txt")

    write(
        "1",
        300,
        {110: "G", 111: "T", 112: "A", 197: "C", 198: "A", 199: "G"},
    )
    write(
        "2",
        1200,
        {
            1099: "C",
            1098: "A",
            1097: "C",
            1014: "A",
            1013: "T",
            1012: "C",
        },
    )
    out = splice_flank_composition(
        models, genome_dir=tmp_path, rules=None
    )
    assert out["introns"] == 2 and out["canonical_introns"] == 2
    assert out["donor_plus3"] == {
        "A": 0.5,
        "C": 0.0,
        "G": 0.5,
        "T": 0.0,
    }
    assert out["acceptor_minus3"] == {
        "A": 0.0,
        "C": 0.5,
        "G": 0.0,
        "T": 0.5,
    }


# ---------------------------------------------------------------------
# Germline mask
# ---------------------------------------------------------------------


def _masked_keys(entries):
    from sigmutsel.germline_mask import allele_keys

    chrom, pos, alt = zip(*entries)
    codes = ["ACGT".index(a) for a in alt]
    return np.unique(allele_keys(list(chrom), list(pos), codes))


# GA (+ strand) position 105 is the middle T of TTT: genomic T>C is
# coding T>C (Phe -> Ser). GB (- strand) position 1104 is the middle C
# of CCC on the coding strand, G on the genome: genomic G>A is coding
# C>T (Pro -> Leu). GA's donor +1 at 110 is a coding G; genomic G>A
# there is coding G>A after the exon's last base A.
MASK = [("chr1", 105, "C"), ("chr2", 1104, "A"), ("chr1", 110, "A")]


def test_germline_mask_removes_exactly_the_masked_alleles(models):
    plain = compute_channel_opportunity(models)
    masked = compute_channel_opportunity(
        models, masked_keys=_masked_keys(MASK)
    )
    assert masked["masked"] == {
        "syn": 0.0,
        "mis": 2.0,
        "non": 0.0,
        "spl": 1.0,
    }
    expected = {
        ("mis", "ENSGA", "T[T>C]T"): 1,
        ("mis", "ENSGB", "C[C>T]C"): 1,
        ("spl", "ENSGA", "A[C>T]T"): 1,
    }
    for channel in CHANNELS:
        diff = plain[channel] - masked[channel]
        for (c, gene, tau), n in expected.items():
            if c == channel:
                assert diff.at[gene, tau] == pytest.approx(n)
                diff.at[gene, tau] = 0.0
        np.testing.assert_allclose(diff.to_numpy(), 0.0)
    # Positions are untouched: contexts count sites, not alleles.
    pd.testing.assert_frame_equal(
        plain["contexts"], masked["contexts"]
    )


def test_germline_mask_keeps_one_site_rate_per_type(models):
    tables = compute_channel_opportunity(
        models, masked_keys=_masked_keys(MASK)
    )
    contexts = tables["contexts"]
    type_opportunity = sum(tables[c] for c in CHANNELS)
    rng = np.random.default_rng(1)
    mu_taus = pd.DataFrame(
        rng.gamma(2.0, 1.0, size=(2, 96)),
        index=["t1", "t2"],
        columns=canonical_types_order,
    )
    denominators = type_opportunity.sum(axis=0)
    for channel in CHANNELS:
        rates = compute_mu_g_channel_per_tumor(
            mu_taus,
            tables[channel],
            contexts,
            separate_per_tau=True,
            type_opportunity=type_opportunity,
        )
        n = variant_site_denominators(
            contexts, opportunity_by_type=tables[channel]
        )
        for tau in canonical_types_order:
            expected = mu_taus[tau] / denominators[tau]
            for gene in rates[tau].index:
                if tables[channel].at[gene, tau] > 0:
                    np.testing.assert_allclose(
                        (
                            rates[tau].loc[gene] / n.at[gene, tau]
                        ).to_numpy(),
                        expected.to_numpy(),
                    )


def test_type_opportunity_without_a_mask_changes_nothing(models):
    from sigmutsel.estimate_mus import compute_mu_g_per_tumor

    tables = compute_channel_opportunity(models)
    contexts = tables["contexts"]
    type_opportunity = sum(tables[c] for c in CHANNELS)
    mu_taus = pd.DataFrame(
        np.random.default_rng(2).gamma(2.0, 1.0, size=(2, 96)),
        index=["t1", "t2"],
        columns=canonical_types_order,
    )
    for independent in (False, True):
        a = compute_mu_g_per_tumor(
            mu_taus, contexts, prob_g_tau_tau_independent=independent
        )
        b = compute_mu_g_per_tumor(
            mu_taus,
            contexts,
            prob_g_tau_tau_independent=independent,
            type_opportunity=type_opportunity,
        )
        pd.testing.assert_frame_equal(a, b, check_exact=False)
        for channel in ("syn", "mis"):
            a = compute_mu_g_channel_per_tumor(
                mu_taus,
                tables[channel],
                contexts,
                prob_g_tau_tau_independent=independent,
            )
            b = compute_mu_g_channel_per_tumor(
                mu_taus,
                tables[channel],
                contexts,
                prob_g_tau_tau_independent=independent,
                type_opportunity=type_opportunity,
            )
            pd.testing.assert_frame_equal(a, b, check_exact=False)


def test_germline_mask_table_and_calls(tmp_path):
    import gzip

    from sigmutsel.germline_mask import (
        _parse_af,
        calls_on_masked_alleles,
        load_germline_mask,
    )

    assert _parse_af(b"AC=3;AN=10;AF=1.5e-03;AF_afr=0") == 1.5e-3
    assert _parse_af(b"AF=0.25;AC=1") == 0.25
    assert _parse_af(b"AC=0;AN=0") is None

    table = tmp_path / "snv.tsv.gz"
    with gzip.open(table, "wt") as out:
        out.write("chrom\tpos\tref\talt\taf\tfilter\n")
        out.write("chr1\t105\tT\tC\t0.01\tPASS\n")
        out.write("chr1\t106\tT\tG\t0.0005\tPASS\n")
        out.write("chrX\t7\tA\tG\t0.2\tRF\n")
    keys = load_germline_mask(0.001, path=table)
    assert len(keys) == 2
    assert len(load_germline_mask(0.0, path=table)) == 3

    db = pd.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr1", "chrX"],
            "Start_Position": [105, 105, 106, 7],
            "Tumor_Seq_Allele2": ["C", "A", "G", "G"],
        }
    )
    assert calls_on_masked_alleles(db, keys).tolist() == [
        True,
        False,
        False,
        True,
    ]


def test_germline_table_resolution(tmp_path, monkeypatch):
    """Full table first; else the distributed one down to its floor;
    below the floor only an explicit full build."""
    import gzip

    from sigmutsel import germline_mask as gm

    full = tmp_path / "full.tsv.gz"
    dist = tmp_path / gm.DISTRIBUTED_NAME
    monkeypatch.setattr(gm, "default_table_path", lambda: full)
    monkeypatch.setattr(gm, "distributed_table_path", lambda: dist)

    rows = "chrom\tpos\tref\talt\taf\tfilter\n" + "".join(
        f"chr1\t{100 + i}\tA\tG\t{af}\tPASS\n"
        for i, af in enumerate([1e-6, 2e-5, 6e-5, 2e-4, 0.01])
    )

    def fake_download(force=False, url=None):
        with gzip.GzipFile(dist, "wb", mtime=0) as out:
            out.write(
                "".join(
                    line + "\n"
                    for line in rows.splitlines()
                    if line.startswith("chrom")
                    or float(line.split("\t")[4])
                    > gm.DISTRIBUTED_FLOOR
                ).encode()
            )
        return dist

    monkeypatch.setattr(
        gm, "download_germline_mask_table", fake_download
    )
    assert gm.resolve_table(1.5e-4) == dist
    assert len(gm.load_germline_mask(1.5e-4)) == 2
    with pytest.raises(FileNotFoundError):
        gm.resolve_table(1e-5)
    with gzip.open(full, "wt") as out:
        out.write(rows)
    assert gm.resolve_table(1e-5) == full
    assert len(gm.load_germline_mask(1e-5)) == 4
    # The distributed cut of the full table is what was downloaded.
    cut = gm.write_distributed_table(full, tmp_path / "cut.tsv.gz")
    with gzip.open(cut) as a, gzip.open(dist) as b:
        assert a.read() == b.read()


def test_distributed_download_checks_its_checksum(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from sigmutsel import germline_mask as gm

    monkeypatch.setattr(
        gm, "distributed_table_path", lambda: tmp_path / "t.tsv.gz"
    )
    monkeypatch.setattr(
        gm.urllib.request,
        "urlretrieve",
        lambda url, dest: Path(dest).write_bytes(b"not the table"),
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        gm.download_germline_mask_table()
    assert not (tmp_path / "t.tsv.gz").exists()


def test_junction_neighbours_come_from_the_genome(models, tmp_path):
    """Across a junction the CDS gives the next exon's base; a genome
    gives the intron's. Bases chosen to differ from the CDS ones."""
    genome = tmp_path / "tsb"
    genome.mkdir()
    for chrom, size in (("1", 400), ("2", 1300)):
        (genome / f"{chrom}.txt").write_bytes(bytes(size))  # all A
    g1 = bytearray((genome / "1.txt").read_bytes())
    g1[110 - 1] = 3  # T after GA's first exon (donor +1)
    g1[199 - 1] = 1  # C before GA's second exon (acceptor -1)
    (genome / "1.txt").write_bytes(bytes(g1))
    g2 = bytearray((genome / "2.txt").read_bytes())
    g2[1099 - 1] = 0  # A below GB's first exon: coding-strand T
    (genome / "2.txt").write_bytes(bytes(g2))

    a, b = models.gene_index["ENSGA"], models.gene_index["ENSGB"]
    oa, ob = models.offsets[a], models.offsets[b]
    left_c, right_c, src_c = models.neighbours(genome_dir=None)
    left_g, right_g, src_g = models.neighbours(genome_dir=genome)
    assert (src_c, src_g) == ("cds", "genome")
    # GA: position 109 (flat oa+8) and 200 (oa+9)
    assert right_c[oa + 8] == 2 and right_g[oa + 8] == 3
    assert left_c[oa + 9] == 0 and left_g[oa + 9] == 1
    # GB (minus strand): 1100 (ob+8) is followed by intron base 1099
    assert right_g[ob + 8] == 3
    # Away from junctions nothing changes.
    inner = np.ones(len(models.codes), dtype=bool)
    inner[[oa + 8, oa + 9, ob + 8, ob + 9]] = False
    np.testing.assert_array_equal(left_c[inner], left_g[inner])
    np.testing.assert_array_equal(right_c[inner], right_g[inner])


# ---------------------------------------------------------------------
# Site weights
# ---------------------------------------------------------------------


def _planted():
    from sigmutsel.site_weights import SiteWeights

    return SiteWeights(
        distance=np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
        strand=np.array(
            [
                1.2,
                0.8,
                1.1,
                0.9,
                1.3,
                0.7,
                1.0,
                1.0,
                0.6,
                1.4,
                1.05,
                0.95,
            ]
        ),
    )


def test_unit_site_weights_change_nothing(models):
    from sigmutsel.site_weights import SiteWeights

    plain = compute_channel_opportunity(models)
    unit = compute_channel_opportunity(
        models, site_weights=SiteWeights()
    )
    for ch in CHANNELS:
        pd.testing.assert_frame_equal(plain[ch], unit[ch])
    pd.testing.assert_frame_equal(plain["contexts"], unit["contexts"])


def test_weighted_opportunity_is_the_sum_of_site_weights(models):
    from sigmutsel.channel_universe import _coding_site_table
    from sigmutsel.site_weights import position_bins, weights_at

    sw = _planted()
    weighted = compute_channel_opportunity(models, site_weights=sw)
    plain = compute_channel_opportunity(models)
    _gene, types, cons, idx, _ = _coding_site_table(models)
    dbin, orient = position_bins(models)
    w = weights_at(dbin[idx], orient[idx], types, sw)
    for h, ch in enumerate(("syn", "mis", "non")):
        expected = w[cons == h].sum()
        assert weighted[ch].to_numpy().sum() == pytest.approx(
            expected
        )
    # splice sites are not weighted; positions are never weighted
    pd.testing.assert_frame_equal(weighted["spl"], plain["spl"])
    pd.testing.assert_frame_equal(
        weighted["contexts"], plain["contexts"]
    )


def test_fit_recovers_planted_site_weights():
    from sigmutsel.site_weights import CLASS_OF, fit_site_weights

    rng = np.random.default_rng(3)
    n = 300000
    dbin = rng.integers(0, 8, n)
    orient = rng.integers(0, 2, n)
    types = rng.integers(0, 96, (n, 3))
    expected = np.full((n, 3), 0.2)
    true = _planted()
    # normalise the planted factors the way the fit does
    cls_or = CLASS_OF[types] * 2 + orient[:, None]
    w = true.distance[dbin][:, None] * true.strand[cls_or]
    counts = rng.poisson(expected * w)
    rows, alts = np.nonzero(counts)
    rows = np.repeat(rows, counts[rows, alts])
    alts = np.repeat(alts, counts[np.nonzero(counts)])
    fit = fit_site_weights(
        dbin, orient, types, expected, rows, alts, pseudo_count=0.0
    )
    ratio_u = fit.distance / true.distance
    np.testing.assert_allclose(
        ratio_u / ratio_u.mean(), 1.0, atol=0.03
    )
    for k in range(6):
        pair = [2 * k, 2 * k + 1]
        got = fit.strand[pair[0]] / fit.strand[pair[1]]
        want = true.strand[pair[0]] / true.strand[pair[1]]
        assert got == pytest.approx(want, rel=0.05)


def test_route_weights_line_up_with_routes(models):
    from sigmutsel.channel_universe import _routes
    from sigmutsel.site_weights import (
        position_bins,
        variant_route_weights,
        weights_at,
    )

    sw = _planted()
    db = pd.DataFrame(
        {
            "in_universe": [True],
            "channel": ["mis"],
            "variant": ["GA p.F2L"],
            "ensembl_gene_id": ["ENSGA"],
            "Start_Position": [104],
            "Tumor_Seq_Allele2": ["C"],
        }
    )
    got = variant_route_weights(db, models, None, sw)["GA p.F2L"]
    start = models.offsets[models.gene_index["ENSGA"]] + 3
    types, sites = _routes(
        models, start, "L", "mis", None, with_sites=True
    )
    assert len(got) == len(types) == 3
    dbin, orient = position_bins(models)
    t_idx = np.array([canonical_types_order.index(t) for t in types])
    s_idx = np.array([i for i, _ in sites])
    want = weights_at(dbin[s_idx], orient[s_idx], t_idx[:, None], sw)[
        :, 0
    ]
    np.testing.assert_allclose(got, want)


def test_route_weights_scale_variant_rates(models):
    from sigmutsel.estimate_mus import compute_mu_m_per_tumor

    tables = compute_channel_opportunity(models)
    contexts = tables["contexts"]
    nonsyn = tables["mis"] + tables["non"] + tables["spl"]
    mu_taus = pd.DataFrame(
        np.random.default_rng(4).gamma(2.0, 1.0, size=(2, 96)),
        index=["t1", "t2"],
        columns=canonical_types_order,
    )
    gene_rates = compute_mu_g_channel_per_tumor(
        mu_taus, nonsyn, contexts, separate_per_tau=True
    )
    routes = ["G[T>C]T", "T[T>A]C", "T[T>A]C"]

    def rate(weights):
        variants = pd.DataFrame(
            {
                "ensembl_gene_id": ["ENSGA"],
                "gene": ["GA"],
                "mut_types": [routes],
                **({"route_weights": [weights]} if weights else {}),
            },
            index=["GA p.F2L"],
        )
        return compute_mu_m_per_tumor(
            variants, gene_rates, contexts, opportunity_by_type=nonsyn
        ).loc["GA p.F2L"]

    base = rate(None)
    np.testing.assert_allclose(rate([1.0, 1.0, 1.0]), base)
    single = rate([1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        rate([2.0, 0.5, 0.5]), 2 * single + 0.5 * (base - single)
    )
