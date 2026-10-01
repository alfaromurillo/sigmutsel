"""The channel universe: one territory for calls and opportunity.

A call is modelled if and only if it falls in one of four
**channels** of one chosen transcript per gene, inside the capture
territory:

========  ============================================================
channel   sites
========  ============================================================
``syn``   coding substitutions that keep the amino acid (a stop that
          stays a stop included)
``mis``   coding substitutions that change the amino acid; start-lost
          and stop-lost go here, as in dNdScv
``non``   coding substitutions that create a stop codon
``spl``   essential splice sites: the two intronic bases at each end
          of each intron of the coding region (donor +1/+2, acceptor
          -2/-1)
========  ============================================================

Everything else -- intron, UTR, flank, non-coding RNA genes, splice
region 3-8, and any call outside the territory -- is out of the
universe: kept in the stored mutation table, dropped from every
model input. Exonic splice-region calls are coding changes and take
their coding channel.

Both halves of the model read the same objects built here, so they
describe the same territory by construction:

* **The opportunity** (:func:`channel_opportunity`): per gene and SBS
  type, the number of sites of each channel. For every gene *g* and
  type *tau*, ``sum_h n^h[g, tau] == contexts[g, context(tau)]``, the
  site count behind the common denominator ``sum_g' n_{g', c(tau)}``.
* **The calls** (:func:`classify_calls`): each call is placed on the
  same transcript, at its genomic position, and labelled with its
  channel, whether it is in the universe, a protein-level variant
  label computed on that transcript, and every single-nucleotide
  route to that change (``routes``), each at its own context.

Transcript choice (:func:`select_transcripts`)
----------------------------------------------
One transcript per gene, for labels and opportunity alike: GENCODE
v38's ``MANE_Select``, else ``Ensembl_canonical``, else the longest
CDS. A candidate is usable only if the Ensembl CDS FASTA holds the
same stable transcript id with a sequence whose length matches the
GTF's CDS plus stop codon (after the FASTA's leading ``N`` padding),
so that sequence and exon structure agree; otherwise the next rule
applies. The rule used, and whether the FASTA carried the same
transcript version, are recorded per gene.

No genome (first version)
-------------------------
Sequence comes from the CDS FASTA, exon structure from the GTF.
Codons, and so every coding consequence, are exact. Two things need
intronic bases and are approximated:

* **Essential splice contexts.** Donor +1 (``xGT``) and acceptor -1
  (``AGx``) are exact from the adjacent coding base and the canonical
  GT/AG. Donor +2 (``GTb``) and acceptor -2 (``bAG``) each lack one
  intronic base; each such site is spread over the four contexts with
  weights :data:`DONOR_PLUS3_COMPOSITION` and
  :data:`ACCEPTOR_MINUS3_COMPOSITION`. The per-gene ``spl`` total is
  exact (4 sites per intron, times 3 alternate bases). Non-canonical
  introns (GC-AG, AT-AC; 1.1% in the sample below) are treated as
  canonical.
* **Junction bases of coding positions.** The first and last base of
  each exon keep the neighbour from the adjacent exon, as the CDS
  FASTA gives them (about 1% of positions). Call labels and variant
  routes use the same neighbours, so a site's context agrees between
  opportunity and routes.

The splice compositions were **measured** over all 159,892 coding
introns of the GENCODE v38 MANE Select transcripts, on GRCh38 as
SigProfilerMatrixGenerator installs it, by
:func:`splice_flank_composition` (which reproduces them). 99.1% are
GT-AG; the 158,454 canonical ones give the constants below. The
classic table (Shapiro & Senapathy 1987, Nucleic Acids Res.
15:7155, Table 1, 542 primate sites) agrees in pattern: donor +3
A 57, C 2, G 39, T 2%; acceptor -3 A 3, C 74, G 1, T 22%. The
census is used because it counts the very introns modelled here.

Territory
---------
A site counts, and a call is kept, only if it lies inside the MC3
capture BED (hg19). Positions are lifted GRCh38 -> hg19 one at a time
(:mod:`sigmutsel.liftover`); a position the chain does not map is
outside. Splice sites may be tested against a padded BED
(``splice_padding``); coding positions never are.
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import canonical_types_order

logger = logging.getLogger(__name__)

CHANNELS = ("syn", "mis", "non", "spl")
NONSYN_CHANNELS = ("mis", "non", "spl")
TRANSCRIPT_RULES = ("mane_select", "ensembl_canonical", "longest_cds")

# Measured frequencies at the intronic base the donor +2 / acceptor -2
# contexts lack (see the module docstring for how).
DONOR_PLUS3_COMPOSITION = {
    "A": 0.6096,
    "C": 0.0252,
    "G": 0.3381,
    "T": 0.0271,
}
ACCEPTOR_MINUS3_COMPOSITION = {
    "A": 0.0564,
    "C": 0.6457,
    "G": 0.0017,
    "T": 0.2961,
}

# Bumped whenever a change here alters the derived tables, so a cache
# written by older code is never served.
_BUILD_VERSION = 5

_BASES = "ACGT"
_CODE = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(_BASES):
    _CODE[ord(_b)] = _i
_COMPLEMENT_CODE = np.array([3, 2, 1, 0, 4], dtype=np.int8)
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_TYPE_INDEX = {t: i for i, t in enumerate(canonical_types_order)}
_SPLICE_OFFSETS = {"d1": "+1", "d2": "+2", "a2": "-2", "a1": "-1"}


def _codon_table():
    """The standard genetic code, 64 codons, via Biopython."""
    from .consequence_contexts_by_gene import _build_codon_table

    return _build_codon_table()


_CODONS = _codon_table()


def _build_type_table():
    """``table[l, r, x, a]``: canonical type index of a coding-strand
    substitution ``l r x -> l a x`` (-1 when ``a == r``)."""
    table = np.full((4, 4, 4, 4), -1, dtype=np.int16)
    for li, left in enumerate(_BASES):
        for ri, ref in enumerate(_BASES):
            for xi, right in enumerate(_BASES):
                for ai, alt in enumerate(_BASES):
                    if alt == ref:
                        continue
                    if ref in "CT":
                        label = f"{left}[{ref}>{alt}]{right}"
                    else:
                        tri = (left + ref + right).translate(
                            _COMPLEMENT
                        )[::-1]
                        label = (
                            f"{tri[0]}[{tri[1]}>"
                            f"{alt.translate(_COMPLEMENT)}]{tri[2]}"
                        )
                    table[li, ri, xi, ai] = _TYPE_INDEX[label]
    return table


def _build_consequence_table():
    """``table[c0, c1, c2, q, a]``: 0 syn, 1 mis, 2 non, -1 no change.

    Stop to stop is synonymous, stop to amino acid (stop-lost) is
    missense, amino acid to stop is nonsense.
    """
    table = np.full((4, 4, 4, 3, 4), -1, dtype=np.int8)
    for codon, residue in _CODONS.items():
        c = [_BASES.index(b) for b in codon]
        for q in range(3):
            for ai, alt in enumerate(_BASES):
                if alt == codon[q]:
                    continue
                new = _CODONS[codon[:q] + alt + codon[q + 1 :]]
                if new == residue:
                    table[c[0], c[1], c[2], q, ai] = 0
                elif new == "*":
                    table[c[0], c[1], c[2], q, ai] = 2
                else:
                    table[c[0], c[1], c[2], q, ai] = 1
    return table


_TYPE_TABLE = _build_type_table()
_CONSEQUENCE_TABLE = _build_consequence_table()


# ---------------------------------------------------------------------
# Reading the annotation
# ---------------------------------------------------------------------

_ATTR_RE = re.compile(r'(\S+) "([^"]*)"')


def read_coding_transcripts(gtf_path):
    """Coding transcripts and their CDS/stop features from a GTF.

    Skips chrY PAR copies (``_PAR_Y``; the chrX copy is kept) and
    chrM, whose genetic code differs.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``transcripts`` indexed by versioned transcript id, with
        ``gene_id`` (stable), ``gene_name``, ``gene_type``, ``chrom``,
        ``strand``, ``mane_select``, ``ensembl_canonical``; and
        ``features`` with ``transcript_id``, ``start``, ``end``
        (1-based inclusive), ``feature`` and ``phase``.
    """
    import gzip

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    transcripts = []
    features = []
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            kind = fields[2]
            if kind not in ("transcript", "CDS", "stop_codon"):
                continue
            if fields[0] == "chrM":
                continue
            attrs = fields[8]
            transcript_id = re.search(
                r'transcript_id "([^"]+)"', attrs
            ).group(1)
            if transcript_id.endswith("_PAR_Y"):
                continue
            if kind == "transcript":
                pairs = _ATTR_RE.findall(attrs)
                values = dict(pairs)
                tags = {v for k, v in pairs if k == "tag"}
                transcripts.append(
                    (
                        transcript_id,
                        values["gene_id"].split(".")[0],
                        values.get("gene_name"),
                        values.get("gene_type"),
                        fields[0],
                        fields[6],
                        "MANE_Select" in tags,
                        "Ensembl_canonical" in tags,
                    )
                )
            else:
                features.append(
                    (
                        transcript_id,
                        int(fields[3]),
                        int(fields[4]),
                        kind,
                        int(fields[7]) if fields[7] != "." else 0,
                    )
                )

    transcripts = pd.DataFrame(
        transcripts,
        columns=[
            "transcript_id",
            "gene_id",
            "gene_name",
            "gene_type",
            "chrom",
            "strand",
            "mane_select",
            "ensembl_canonical",
        ],
    ).set_index("transcript_id")
    features = pd.DataFrame(
        features,
        columns=["transcript_id", "start", "end", "feature", "phase"],
    )
    coding = features["transcript_id"].unique()
    return transcripts.loc[transcripts.index.isin(coding)], features


def read_cds_fasta(fasta_paths):
    """Map stable transcript id -> (versioned id, upper-case sequence)."""
    from Bio import SeqIO

    from .contexts_by_gene import normalize_fasta_paths

    records = {}
    for path in normalize_fasta_paths(fasta_paths):
        for rec in SeqIO.parse(path, "fasta"):
            records[rec.id.split(".")[0]] = (
                rec.id,
                str(rec.seq).upper(),
            )
    return records


def _segments(features):
    """Merge each transcript's CDS + stop features into genomic segments.

    Adjacent features (a stop codon right after the last CDS) are
    merged; a gap between two segments is an intron of the coding
    region. Returns a dict transcript -> sorted (start, end) array.
    """
    out = {}
    for transcript_id, group in features.sort_values("start").groupby(
        "transcript_id", sort=False
    ):
        merged = []
        for start, end in zip(group["start"], group["end"]):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        out[transcript_id] = np.array(merged, dtype=np.int64)
    return out


def select_transcripts(transcripts, features, fasta, keep_ids=None):
    """Choose one transcript per gene.

    Rules, in order: GENCODE ``MANE_Select``, ``Ensembl_canonical``,
    longest CDS (ties broken by transcript id). A candidate is usable
    only when ``fasta`` holds the same stable transcript id with a
    sequence that, minus the leading ``N`` padding Ensembl adds to
    5'-incomplete CDSs, is exactly as long as the GTF's CDS plus stop
    codon. The version may differ (the FASTA is a later Ensembl
    release than GENCODE v38, and most version bumps change only the
    UTRs); whether it matched is recorded in
    ``fasta_version_match``.

    Parameters
    ----------
    transcripts, features
        From :func:`read_coding_transcripts`.
    fasta : dict
        From :func:`read_cds_fasta`.
    keep_ids : set of str, optional
        Restrict to these stable gene ids.

    Returns
    -------
    pandas.DataFrame
        Indexed by stable gene id: ``transcript_id``, ``rule`` (one of
        :data:`TRANSCRIPT_RULES`), ``gene_name``, ``gene_type``,
        ``chrom``, ``strand``, ``cds_length``, ``n_pad``,
        ``has_mane`` (whether the gene has a MANE Select transcript
        at all, usable or not), ``n_candidates``.
    """
    lengths = (
        (features["end"] - features["start"] + 1)
        .groupby(features["transcript_id"])
        .sum()
    )
    tx = transcripts.copy()
    if keep_ids is not None:
        tx = tx[tx["gene_id"].isin(keep_ids)]
    tx["cds_length"] = lengths.reindex(tx.index).astype(int)

    def usable(transcript_id):
        stable = transcript_id.split(".")[0]
        hit = fasta.get(stable)
        if hit is None:
            return False, 0, False
        seq = hit[1]
        n_pad = len(seq) - len(seq.lstrip("N"))
        same_length = (
            len(seq) - n_pad == tx.at[transcript_id, "cds_length"]
        )
        return same_length, n_pad, hit[0] == transcript_id

    checks = {t: usable(t) for t in tx.index}
    tx["usable"] = [checks[t][0] for t in tx.index]
    tx["n_pad"] = [checks[t][1] for t in tx.index]
    tx["fasta_version_match"] = [checks[t][2] for t in tx.index]

    rows = {}
    for gene_id, group in tx.groupby("gene_id"):
        ok = group[group["usable"]]
        choice = rule = None
        for rule_name, mask in (
            ("mane_select", ok["mane_select"]),
            ("ensembl_canonical", ok["ensembl_canonical"]),
        ):
            if mask.any():
                choice, rule = min(ok.index[mask]), rule_name
                break
        if choice is None and len(ok):
            longest = ok.sort_values(
                ["cds_length"], ascending=False, kind="stable"
            )
            top = longest["cds_length"].iloc[0]
            choice = min(longest.index[longest["cds_length"] == top])
            rule = "longest_cds"
        if choice is None:
            continue
        row = tx.loc[choice]
        rows[gene_id] = {
            "transcript_id": choice,
            "rule": rule,
            "gene_name": row["gene_name"],
            "gene_type": row["gene_type"],
            "chrom": row["chrom"],
            "strand": row["strand"],
            "cds_length": int(row["cds_length"]),
            "n_pad": int(row["n_pad"]),
            "fasta_version_match": bool(row["fasta_version_match"]),
            "has_mane": bool(group["mane_select"].any()),
            "n_candidates": len(group),
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "gene_id"
    return out.sort_index()


# ---------------------------------------------------------------------
# Transcript models
# ---------------------------------------------------------------------


class TranscriptModels:
    """Chosen transcripts as flat arrays, one block per gene.

    Attributes
    ----------
    selection : pandas.DataFrame
        :func:`select_transcripts`' output; its row order is the gene
        index used by every array below.
    offsets, lengths : numpy.ndarray
        Start and length of each gene's block in the flat arrays.
    codes : numpy.ndarray (int8)
        Padded CDS sequence, coding strand, A/C/G/T = 0..3, N = 4.
        Index ``i`` of a block is in frame: codon ``i // 3``.
    gpos : numpy.ndarray (int64)
        Genomic position (1-based, GRCh38) of each block index; -1 for
        the leading ``N`` padding.
    splice : pandas.DataFrame
        One row per essential splice site: ``gene`` (index into
        ``selection``), ``gpos``, ``kind`` (``d1``, ``d2``, ``a2``,
        ``a1``), ``anchor`` (1-based unpadded CDS coordinate of the
        coding base the site hangs off, for ``c.`` labels) and
        ``exon_base`` (that base's code).
    """

    def __init__(
        self, selection, offsets, lengths, codes, gpos, splice
    ):
        self.selection = selection
        self.offsets = offsets
        self.lengths = lengths
        self.codes = codes
        self.gpos = gpos
        self.splice = splice
        self.directory = None
        self.gene_ids = selection.index.to_numpy()
        self.gene_index = {g: i for i, g in enumerate(self.gene_ids)}
        chrom = selection["chrom"].to_numpy()
        self.gene_of = np.repeat(np.arange(len(selection)), lengths)
        self.chrom_of = chrom[self.gene_of]
        self.local = np.arange(len(codes)) - offsets[self.gene_of]

    @classmethod
    def build(cls, selection, features, fasta):
        """Assemble models from a selection, GTF features and FASTA."""
        segments = _segments(
            features[
                features["transcript_id"].isin(
                    selection["transcript_id"]
                )
            ]
        )
        blocks_codes = []
        blocks_gpos = []
        splice_rows = []
        lengths = np.zeros(len(selection), dtype=np.int64)
        for g, (gene_id, row) in enumerate(selection.iterrows()):
            seq = fasta[row["transcript_id"].split(".")[0]][1]
            segs = segments[row["transcript_id"]]
            minus = row["strand"] == "-"
            positions = np.concatenate(
                [np.arange(s, e + 1) for s, e in segs]
            )
            if minus:
                positions = positions[::-1]
            n_pad = row["n_pad"]
            gpos = np.concatenate(
                [np.full(n_pad, -1, dtype=np.int64), positions]
            )
            codes = _CODE[np.frombuffer(seq.encode(), dtype=np.uint8)]
            blocks_codes.append(codes)
            blocks_gpos.append(gpos)
            lengths[g] = len(codes)

            # Introns between consecutive coding segments, in
            # transcript order: (last coding base before, first after).
            order = segs[::-1] if minus else segs
            cds_coord = 0
            for k in range(len(order) - 1):
                cds_coord += order[k][1] - order[k][0] + 1
                donor_anchor = cds_coord
                acceptor_anchor = cds_coord + 1
                last_base = codes[n_pad + donor_anchor - 1]
                first_base = codes[n_pad + acceptor_anchor - 1]
                if minus:
                    donor = order[k][0]  # genomic start of segment
                    acceptor = order[k + 1][1]
                    sites = (
                        (donor - 1, "d1"),
                        (donor - 2, "d2"),
                        (acceptor + 2, "a2"),
                        (acceptor + 1, "a1"),
                    )
                else:
                    donor = order[k][1]
                    acceptor = order[k + 1][0]
                    sites = (
                        (donor + 1, "d1"),
                        (donor + 2, "d2"),
                        (acceptor - 2, "a2"),
                        (acceptor - 1, "a1"),
                    )
                for pos, kind in sites:
                    donor_side = kind[0] == "d"
                    splice_rows.append(
                        (
                            g,
                            pos,
                            kind,
                            (
                                donor_anchor
                                if donor_side
                                else acceptor_anchor
                            ),
                            last_base if donor_side else first_base,
                        )
                    )

        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        splice = pd.DataFrame(
            splice_rows,
            columns=["gene", "gpos", "kind", "anchor", "exon_base"],
        )
        return cls(
            selection,
            offsets.astype(np.int64),
            lengths,
            np.concatenate(blocks_codes).astype(np.int8),
            np.concatenate(blocks_gpos).astype(np.int64),
            splice,
        )

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # models.npz goes last and atomically: a reader that finds it
        # (the cache test in load_or_build_transcript_models) finds the
        # other two complete, even with several processes building.
        self.selection.to_csv(directory / "transcripts.csv")
        self.splice.to_parquet(directory / "splice_sites.parquet")
        tmp = directory / f".models.{os.getpid()}.npz"
        np.savez_compressed(
            tmp,
            offsets=self.offsets,
            lengths=self.lengths,
            codes=self.codes,
            gpos=self.gpos,
        )
        os.replace(tmp, directory / "models.npz")
        self.directory = directory

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        selection = pd.read_csv(
            directory / "transcripts.csv", index_col=0
        )
        arrays = np.load(directory / "models.npz")
        splice = pd.read_parquet(directory / "splice_sites.parquet")
        models = cls(
            selection,
            arrays["offsets"],
            arrays["lengths"],
            arrays["codes"],
            arrays["gpos"],
            splice,
        )
        models.directory = directory
        return models

    def neighbours(self, genome_dir="default"):
        """Coding-strand left and right neighbour of every position.

        Within an exon the neighbour is the adjacent base of the CDS.
        Across an exon-exon junction the CDS joins exons, so its
        neighbour is the next exon's base; the genome's is the
        intron's (the donor's G, the acceptor's G). With a genome in
        SigProfilerMatrixGenerator's ``tsb`` format (default: the
        installed GRCh38, :func:`sigprofiler_genome_dir`), the two
        positions next to each junction get the intronic base, so
        their trinucleotide -- and the type of every substitution
        there -- is the one a call's genomic context gives. Without a
        genome the CDS neighbours are kept.

        Returns ``(left, right, source)``, ``source`` being
        ``"genome"`` or ``"cds"``; cached per genome.
        """
        if genome_dir == "default":
            genome_dir = sigprofiler_genome_dir()
        key = str(genome_dir)
        cache = getattr(self, "_neighbour_cache", {})
        if key in cache:
            return cache[key]
        codes = self.codes
        n = len(codes)
        left = np.full(n, 4, dtype=np.int8)
        right = np.full(n, 4, dtype=np.int8)
        left[1:] = codes[:-1]
        right[:-1] = codes[1:]
        source = "cds"
        if genome_dir is not None:
            gp, g_of = self.gpos, self.gene_of
            junc = np.flatnonzero(
                (g_of[1:] == g_of[:-1])
                & (gp[1:] >= 0)
                & (gp[:-1] >= 0)
                & (np.abs(gp[1:] - gp[:-1]) != 1)
            )
            minus_gene = self.selection["strand"].to_numpy() == "-"
            for side, at_flat in (
                ("before", junc),
                ("after", junc + 1),
            ):
                g_minus = minus_gene[g_of[at_flat]]
                step = np.where(g_minus, -1, 1) * (
                    1 if side == "before" else -1
                )
                at = gp[at_flat] + step
                base = np.full(len(at_flat), 4, dtype=np.int8)
                chroms = self.chrom_of[at_flat]
                for chrom in pd.unique(chroms):
                    path = Path(genome_dir) / (
                        f"{str(chrom).removeprefix('chr')}.txt"
                    )
                    if not path.exists():
                        continue
                    genome = np.memmap(path, dtype=np.uint8, mode="r")
                    sel = np.flatnonzero(chroms == chrom)
                    a = at[sel]
                    ok = (a >= 1) & (a <= len(genome))
                    raw = np.full(len(sel), 16, dtype=np.uint8)
                    raw[ok] = genome[a[ok] - 1]
                    base[sel] = np.where(raw < 16, raw % 4, 4)
                base = np.where(
                    g_minus & (base < 4), _COMPLEMENT_CODE[base], base
                )
                good = base < 4
                if side == "before":
                    right[at_flat[good]] = base[good]
                else:
                    left[at_flat[good]] = base[good]
            source = "genome"
        cache[key] = (left, right, source)
        self._neighbour_cache = cache
        return cache[key]


def _fingerprint(*paths, extra=""):
    """Short hash of input file names and sizes, plus the build version."""
    h = hashlib.sha1()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    h.update(f"{_BUILD_VERSION}{extra}".encode())
    return h.hexdigest()[:12]


def load_or_build_transcript_models(
    gtf_path=None, fasta_paths=None, cache_dir=None, force=False
):
    """Transcript models for every coding gene, cached on disk.

    The cache lives in ``cache_dir`` (default
    :data:`locations.location_channel_universe_dir`) under a
    sub-directory named by a fingerprint of the GTF and FASTA and of
    this module's build version, so a change to any of them rebuilds
    rather than serving a stale table.
    """
    from .locations import (
        location_cds_fasta,
        location_channel_universe_dir,
        location_gencode38_annotation,
    )

    gtf_path = Path(gtf_path or location_gencode38_annotation)
    fasta_paths = fasta_paths or location_cds_fasta
    from .contexts_by_gene import normalize_fasta_paths

    fasta_list = normalize_fasta_paths(fasta_paths)
    if not gtf_path.exists():
        from .setup import download_gencode_gtf

        download_gencode_gtf(version="38")
    if not all(p.exists() for p in fasta_list):
        from .setup import download_cds_fasta

        download_cds_fasta()
    cache_dir = Path(cache_dir or location_channel_universe_dir)
    key = _fingerprint(gtf_path, *fasta_list)
    directory = cache_dir / f"models_{key}"
    if (directory / "models.npz").exists() and not force:
        logger.info(f"Loading transcript models from {directory}")
        return TranscriptModels.load(directory)

    logger.info("Building transcript models (MANE Select first)...")
    transcripts, features = read_coding_transcripts(gtf_path)
    fasta = read_cds_fasta(fasta_list)
    selection = select_transcripts(transcripts, features, fasta)
    models = TranscriptModels.build(selection, features, fasta)
    models.save(directory)
    counts = selection["rule"].value_counts().to_dict()
    logger.info(
        f"... {len(selection)} genes: {counts}; cached at {directory}"
    )
    return models


# ---------------------------------------------------------------------
# Territory
# ---------------------------------------------------------------------


def territory_masks(
    models, splice_padding=0, bed_path=None, chain_path=None
):
    """Which coding positions and splice sites lie in the MC3 territory.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Boolean masks over ``models.codes`` (padding is outside) and
        over ``models.splice`` rows.
    """
    from . import setup
    from .liftover import (
        in_intervals,
        lift_positions,
        merged_bed_intervals,
        read_chain,
    )
    from .locations import (
        location_liftover_chain_hg38_to_hg19,
        location_wes_target_bed,
    )
    from .wes_target import _parse_bed

    bed_path = Path(bed_path or location_wes_target_bed)
    chain_path = Path(
        chain_path or location_liftover_chain_hg38_to_hg19
    )
    if not bed_path.exists():
        setup.download_wes_target_bed()
    if not chain_path.exists():
        setup.download_liftover_chain()

    cache = None
    if getattr(models, "directory", None) is not None:
        key = _fingerprint(
            bed_path, chain_path, extra=str(splice_padding)
        )
        cache = Path(models.directory) / f"territory_{key}.npz"
        if cache.exists():
            arrays = np.load(cache)
            return arrays["coding_in"], arrays["splice_in"]

    blocks = read_chain(chain_path)
    bed = _parse_bed(bed_path)

    coding = models.gpos >= 0
    chrom19, pos19 = lift_positions(
        blocks, models.chrom_of[coding], models.gpos[coding]
    )
    coding_in = np.zeros(len(models.codes), dtype=bool)
    coding_in[coding] = in_intervals(
        merged_bed_intervals(bed, 0), chrom19, pos19
    )

    splice_chrom = models.selection["chrom"].to_numpy()[
        models.splice["gene"].to_numpy()
    ]
    chrom19, pos19 = lift_positions(
        blocks, splice_chrom, models.splice["gpos"].to_numpy()
    )
    splice_in = in_intervals(
        merged_bed_intervals(bed, splice_padding), chrom19, pos19
    )
    if cache is not None:
        tmp = cache.with_name(f".{cache.stem}.{os.getpid()}.npz")
        np.savez_compressed(
            tmp, coding_in=coding_in, splice_in=splice_in
        )
        os.replace(tmp, cache)
    return coding_in, splice_in


# ---------------------------------------------------------------------
# Opportunity
# ---------------------------------------------------------------------


def _coding_site_table(models, coding_in=None):
    """Per coding position: gene, type of each alt, channel of each alt.

    Counted positions are exactly those
    :func:`contexts_by_gene.compute_contexts_by_gene` counts -- every
    centre position from 1 to ``len - 2`` of the (padded) sequence
    whose trinucleotide window is pure ACGT -- restricted to the
    territory when ``coding_in`` is given.

    Returns ``(gene, types, channels)`` where ``types`` and
    ``channels`` are ``(n, 3)`` arrays over the three alternate bases.
    A position whose codon is truncated or contains N is ``mis``.
    """
    codes = models.codes
    n = len(codes)
    local = models.local
    length = models.lengths[models.gene_of]
    centre = (local >= 1) & (local <= length - 2)
    idx = np.flatnonzero(centre)
    nb_left, nb_right, _ = models.neighbours()
    left = nb_left[idx]
    ref = codes[idx]
    right = nb_right[idx]
    ok = (left < 4) & (ref < 4) & (right < 4)
    if coding_in is not None:
        ok &= coding_in[idx]
    idx, left, ref, right = idx[ok], left[ok], ref[ok], right[ok]

    q = local[idx] % 3
    codon_start = idx - q
    c0 = codes[codon_start]
    c1 = codes[np.minimum(codon_start + 1, n - 1)]
    c2 = codes[np.minimum(codon_start + 2, n - 1)]
    complete = (local[idx] - q + 2 <= length[idx] - 1) & (
        (c0 < 4) & (c1 < 4) & (c2 < 4)
    )
    alts = np.stack(
        [(ref + k) % 4 for k in (1, 2, 3)], axis=1
    )  # three alternate bases
    types = _TYPE_TABLE[
        left[:, None], ref[:, None], right[:, None], alts
    ]
    cons = np.full(alts.shape, 1, dtype=np.int8)
    cc = complete
    cons[cc] = _CONSEQUENCE_TABLE[
        c0[cc, None],
        c1[cc, None],
        c2[cc, None],
        q[cc, None],
        alts[cc],
    ]
    return models.gene_of[idx], types, cons, idx, (~complete).sum()


def _normalised(composition):
    """Weights over A, C, G, T summing to exactly 1.

    The measured frequencies are rounded, so they are renormalised:
    a site's three opportunities must total exactly 3 however they
    are spread over contexts.
    """
    total = sum(composition.values())
    return [composition[b] / total for b in _BASES]


def _splice_site_weights(models, splice_in=None, masked_keys=None):
    """Per splice site: (gene, type index, weight) triples, flat arrays.

    Each site contributes weight 1 per alternate base (3 per site),
    spread over contexts for donor +2 and acceptor -2. With
    ``masked_keys`` (:func:`germline_mask.allele_keys`), only the
    masked alternate bases keep their weight: the result is then the
    masked part of the splice opportunity.
    """
    sp = models.splice
    if splice_in is not None:
        sp = sp[splice_in]
    A, G, T = 0, 2, 3
    # Coding-strand reference base of each kind of essential site.
    kind_ref = {"d1": G, "d2": T, "a2": A, "a1": G}
    minus_gene = models.selection["strand"].to_numpy() == "-"
    genes, types, weights = [], [], []
    for kind, group in sp.groupby("kind"):
        g = group["gene"].to_numpy()
        exon = group["exon_base"].to_numpy()
        if masked_keys is not None:
            from .germline_mask import allele_keys

            site = allele_keys(
                models.selection["chrom"].to_numpy()[g],
                group["gpos"].to_numpy(),
                0,
            )
            minus = minus_gene[g]
        if kind in ("d1", "a1"):
            contexts = [
                (
                    (
                        exon,
                        np.full_like(exon, G),
                        np.full_like(exon, T),
                    )
                    if kind == "d1"
                    else (
                        np.full_like(exon, A),
                        np.full_like(exon, G),
                        exon,
                    )
                ),
            ]
            ctx_weights = [1.0]
        elif kind == "d2":
            contexts = [
                (
                    np.full_like(exon, G),
                    np.full_like(exon, T),
                    np.full_like(exon, b),
                )
                for b in range(4)
            ]
            ctx_weights = _normalised(DONOR_PLUS3_COMPOSITION)
        else:  # a2
            contexts = [
                (
                    np.full_like(exon, b),
                    np.full_like(exon, A),
                    np.full_like(exon, G),
                )
                for b in range(4)
            ]
            ctx_weights = _normalised(ACCEPTOR_MINUS3_COMPOSITION)
        for k in (1, 2, 3):
            hit = np.ones(len(g))
            if masked_keys is not None:
                alt_code = (kind_ref[kind] + k) % 4
                genomic_alt = np.where(minus, 3 - alt_code, alt_code)
                hit = np.isin(site + genomic_alt, masked_keys).astype(
                    float
                )
            for (left, ref, right), w in zip(contexts, ctx_weights):
                alt = (ref + k) % 4
                genes.append(g)
                types.append(_TYPE_TABLE[left, ref, right, alt])
                weights.append(w * hit)
    if not genes:
        return (
            np.array([], int),
            np.array([], int),
            np.array([], float),
        )
    return (
        np.concatenate(genes),
        np.concatenate(types),
        np.concatenate(weights),
    )


def compute_channel_opportunity(
    models,
    coding_in=None,
    splice_in=None,
    masked_keys=None,
    site_weights=None,
):
    """Per-gene, per-type site counts of the four channels.

    Parameters
    ----------
    models : TranscriptModels
    coding_in, splice_in : numpy.ndarray of bool, optional
        Territory masks from :func:`territory_masks`; ``None`` counts
        every site.
    masked_keys : numpy.ndarray of int64, optional
        Germline-masked alleles (:func:`germline_mask.load_germline_mask`).
        Each masked (site, alternate base) is removed from its channel:
        a call there was filtered out, so it is not an opportunity.

    site_weights : site_weights.SiteWeights, optional
        Weight every coding (site, alternate base) by its distance and
        strand factors (:mod:`site_weights`); splice sites keep weight
        1. The ``"contexts"`` position counts are never weighted.

    Returns
    -------
    dict
        ``"syn"``, ``"mis"``, ``"non"``, ``"spl"``: genes x 96 types
        (float; splice sites carry fractional contexts), after the
        mask and the weights; ``"contexts"``: genes x 32 contexts, the site count
        (positions, before any mask); ``"n_undetermined"``: coding
        positions whose codon could not be read (counted as ``mis``);
        ``"masked"``: opportunity removed per channel (0 without a
        mask).

    Notes
    -----
    Once a mask removes one alternate base of a site but not the
    others, a type's opportunity is no longer its context's position
    count, so rate denominators must then come from the channel
    tables (:attr:`models.MutationDataset.type_opportunity`), never
    from ``"contexts"``.
    """
    n_genes = len(models.selection)
    size = n_genes * 4 * 96
    gene, types, cons, idx, n_undetermined = _coding_site_table(
        models, coding_in
    )
    flat = (
        gene[:, None].astype(np.int64) * 4 * 96
        + cons.astype(np.int64) * 96
        + types.astype(np.int64)
    ).ravel()
    counts = np.bincount(flat, minlength=size).astype(float)
    alt_w = None
    if site_weights is not None:
        from .site_weights import position_bins, weights_at

        dbin, orient = position_bins(models)
        alt_w = weights_at(
            dbin[idx], orient[idx], types, site_weights
        )
        counts_w = np.bincount(
            flat, weights=alt_w.ravel(), minlength=size
        ).astype(float)

    sg, st, sw = _splice_site_weights(models, splice_in)
    splice_flat = (
        sg.astype(np.int64) * 4 * 96 + 3 * 96 + st.astype(np.int64)
    )
    if len(sg):
        counts += np.bincount(splice_flat, weights=sw, minlength=size)

    masked = np.zeros(size)
    if masked_keys is not None:
        from .germline_mask import allele_keys

        ref = models.codes[idx].astype(np.int64)
        minus = (models.selection["strand"].to_numpy() == "-")[gene]
        alts = np.stack([(ref + k) % 4 for k in (1, 2, 3)], axis=1)
        genomic_alt = np.where(minus[:, None], 3 - alts, alts)
        site = allele_keys(models.chrom_of[idx], models.gpos[idx], 0)
        hit = np.isin(site[:, None] + genomic_alt, masked_keys)
        hit_w = hit.astype(float) if alt_w is None else hit * alt_w
        masked += np.bincount(
            flat, weights=hit_w.ravel(), minlength=size
        )
        mg, mt, mw = _splice_site_weights(
            models, splice_in, masked_keys=masked_keys
        )
        if len(mg):
            masked += np.bincount(
                mg.astype(np.int64) * 4 * 96
                + 3 * 96
                + mt.astype(np.int64),
                weights=mw,
                minlength=size,
            )

    def frames(array):
        array = array.reshape(n_genes, 4, 96)
        out = {}
        for h, channel in enumerate(CHANNELS):
            frame = pd.DataFrame(
                array[:, h, :],
                index=models.gene_ids,
                columns=canonical_types_order,
            )
            frame.index.name = "ensembl_gene_id"
            out[channel] = frame
        return out

    out = frames(counts)
    out["contexts"] = contexts_from_channels(out)
    if alt_w is not None:
        # Splice sites (not weighted) carry over from the unweighted
        # counts; coding sites take their weighted counts.
        splice_part = np.zeros(size)
        if len(sg):
            splice_part = np.bincount(
                splice_flat, weights=sw, minlength=size
            )
        counts = counts_w + splice_part
    out.update(frames(counts - masked))
    masked = masked.reshape(n_genes, 4, 96)
    out["masked"] = {
        channel: float(masked[:, h, :].sum())
        for h, channel in enumerate(CHANNELS)
    }
    out["n_undetermined"] = int(n_undetermined)
    return out


def contexts_from_channels(tables):
    """Genes x 32 contexts from the four channel tables.

    Each site offers one opportunity to each of the 3 types sharing
    its context, so the context count is the channel total of any of
    them; checked across all three.
    """
    from .constants import extract_context

    total = sum(tables[c] for c in CHANNELS)
    contexts = sorted(
        {extract_context(t) for t in canonical_types_order}
    )
    out = {}
    for ctx in contexts:
        cols = [
            t
            for t in canonical_types_order
            if extract_context(t) == ctx
        ]
        block = total[cols].to_numpy()
        if not np.allclose(block, block[:, :1]):
            raise AssertionError(
                f"Channel tables break the site identity at {ctx}."
            )
        out[ctx] = block[:, 0]
    frame = pd.DataFrame(out, index=total.index)
    return frame


def channel_opportunity(
    keep_ids=None,
    territory="mc3",
    splice_padding=0,
    cache_dir=None,
    force=False,
    models=None,
    germline_mask_af=None,
    site_weights=None,
):
    """Cached channel opportunity tables, optionally subset to genes.

    Parameters
    ----------
    keep_ids : iterable of str, optional
        Stable gene ids to return (the gene universe). Genes without a
        usable transcript, and genes with no site in the territory,
        are absent from the result.
    territory : {"mc3", None}
        ``"mc3"`` restricts sites to the MC3 capture territory;
        ``None`` counts every site (used to check that MANE genes
        reproduce the old longest-CDS table).
    splice_padding : int
        Bases of BED padding for splice sites.
    cache_dir, force
        Cache location and a switch to rebuild.
    models : TranscriptModels, optional
        Reuse already-loaded models.
    germline_mask_af : float or None
        Remove from the opportunity every allele of gnomAD v2.1.1
        exomes with overall allele frequency above this value
        (:mod:`germline_mask`), for calls filtered on germline
        alleles (GDC's masked MAFs). None applies no mask.
    site_weights : site_weights.SiteWeights, optional
        Per-cohort distance and strand factors of the opportunity.

    Returns
    -------
    dict
        The tables of :func:`compute_channel_opportunity`, plus
        ``"transcripts"`` (the per-gene selection record) and
        ``"report"`` (site counts inside/outside the territory, and
        the masked opportunity per channel).
    """
    from .locations import location_channel_universe_dir

    if territory not in ("mc3", None):
        raise ValueError(
            f"territory must be 'mc3' or None, got {territory!r}"
        )
    if models is None:
        models = load_or_build_transcript_models(cache_dir=cache_dir)
    cache_dir = Path(cache_dir or location_channel_universe_dir)
    tag = (
        f"{territory or 'all'}_pad{splice_padding}_"
        f"{len(models.selection)}_{_BUILD_VERSION}_"
        f"{int(models.lengths.sum())}"
    )
    tag += f"_nb-{models.neighbours()[2]}"
    if germline_mask_af is not None:
        from .germline_mask import mask_label

        tag += f"_{mask_label(germline_mask_af)}"
    if site_weights is not None:
        tag += f"_sw-{site_weights.label()}"
    directory = cache_dir / f"opportunity_{tag}"
    names = list(CHANNELS) + ["contexts"]
    if (directory / "report.json").exists() and not force:
        tables = {
            name: pd.read_csv(directory / f"{name}.csv", index_col=0)
            for name in names
        }
        report = json.loads((directory / "report.json").read_text())
    else:
        if territory == "mc3":
            coding_in, splice_in = territory_masks(
                models, splice_padding=splice_padding
            )
        else:
            coding_in = splice_in = None
        masked_keys = None
        if germline_mask_af is not None:
            from .germline_mask import load_germline_mask

            masked_keys = load_germline_mask(germline_mask_af)
        built = compute_channel_opportunity(
            models,
            coding_in,
            splice_in,
            masked_keys=masked_keys,
            site_weights=site_weights,
        )
        tables = {name: built[name] for name in names}
        report = {
            "n_undetermined": built["n_undetermined"],
            "n_genes": len(models.selection),
            "germline_mask_af": germline_mask_af,
            "masked_opportunity": built["masked"],
        }
        if territory == "mc3":
            coding = models.gpos >= 0
            report.update(
                {
                    "coding_positions": int(coding.sum()),
                    "coding_positions_in_territory": int(
                        coding_in.sum()
                    ),
                    "splice_sites": len(splice_in),
                    "splice_sites_in_territory": int(splice_in.sum()),
                }
            )
            if splice_padding:
                _, unpadded = territory_masks(
                    models, splice_padding=0
                )
                report["splice_sites_in_unpadded_territory"] = int(
                    unpadded.sum()
                )
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            tables[name].to_csv(directory / f"{name}.csv")
        (directory / "report.json").write_text(
            json.dumps(report, indent=2)
        )

    # Genes with no site at all have no opportunity to model.
    present = tables["contexts"].sum(axis=1) > 0
    genes = tables["contexts"].index[present]
    if keep_ids is not None:
        genes = genes[genes.isin(set(map(str, keep_ids)))]
    out = {name: tables[name].loc[genes] for name in names}
    out["transcripts"] = models.selection.loc[
        models.selection.index.isin(genes)
    ]
    out["report"] = report
    return out


# ---------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------

_AA_TO_STR = {"*": "*"}


def _lookup(keys_sorted, order, query):
    """Index into the unsorted source array of each query key, or -1."""
    j = np.searchsorted(keys_sorted, query)
    j = np.clip(j, 0, len(keys_sorted) - 1)
    hit = keys_sorted[j] == query
    return np.where(hit, order[j], -1)


def _routes(
    models, codon_start, target, channel, coding_in, with_sites=False
):
    """Every single-nucleotide route to a protein change, as types.

    ``codon_start`` is the flat index of the codon's first base.
    ``target`` is the alternate residue (``None`` for synonymous,
    which takes every synonymous substitution of the codon). A route
    whose site has an N in its window, sits at a block's first or
    last base, or lies outside the territory is skipped -- it has no
    opportunity either.
    """
    codes = models.codes
    g = models.gene_of[codon_start]
    lo = models.offsets[g]
    hi = lo + models.lengths[g]
    codon = "".join(
        _BASES[c] if c < 4 else "N"
        for c in codes[codon_start : codon_start + 3]
    )
    residue = _CODONS.get(codon)
    nb_left, nb_right, _ = models.neighbours()
    routes = []
    sites = []
    for q in range(3):
        i = codon_start + q
        if i - 1 < lo or i + 1 >= hi:
            continue
        left, ref, right = nb_left[i], codes[i], nb_right[i]
        if max(left, ref, right) >= 4:
            continue
        if coding_in is not None and not coding_in[i]:
            continue
        for alt in range(4):
            if alt == ref:
                continue
            new = _CODONS[codon[:q] + _BASES[alt] + codon[q + 1 :]]
            if channel == "syn":
                hit = new == residue
            else:
                hit = new == target
            if hit:
                routes.append(
                    canonical_types_order[
                        _TYPE_TABLE[left, ref, right, alt]
                    ]
                )
                sites.append((i, alt))
    return (routes, sites) if with_sites else routes


def classify_calls(db, models, coding_in=None, splice_in=None):
    """Place each SNV call on its gene's chosen transcript.

    Parameters
    ----------
    db : pandas.DataFrame
        Mutation table with ``ensembl_gene_id``, ``Chromosome``,
        ``Start_Position``, ``Reference_Allele``, ``Tumor_Seq_Allele2``,
        ``gene`` and ``type``.
    models : TranscriptModels
    coding_in, splice_in : numpy.ndarray of bool, optional
        Territory masks; ``None`` puts every site in the territory.

    Returns
    -------
    pandas.DataFrame
        Aligned with ``db.index``: ``channel`` (one of
        :data:`CHANNELS`, or None), ``in_universe`` (bool),
        ``universe_reason`` (``"coding"``, ``"splice"``,
        ``"noncoding"``, ``"no_transcript"``, ``"ref_mismatch"``,
        ``"outside_territory"``), ``transcript_id``,
        ``variant_label`` (protein change on the chosen transcript,
        e.g. ``KRAS p.G12D``, ``TP53 p.R213*``, ``APC p.T1493=``; a
        splice call is ``c.<anchor><+1|+2|-1|-2><ref>><alt>`` on the
        coding strand) and ``routes`` (``;``-joined SBS types of every
        single-nucleotide route to that change; for a splice call its
        own ``type``).

    Notes
    -----
    A call is looked up only in the transcript of the gene the MAF
    assigned it to. A call whose gene has no usable transcript is out
    of the universe, even if it falls in another gene's coding
    sequence.
    """
    n = len(db)
    out = pd.DataFrame(
        {
            "channel": pd.Series(
                [None] * n, index=db.index, dtype=object
            ),
            "in_universe": False,
            "universe_reason": "noncoding",
            "transcript_id": None,
            "variant_label": None,
            "routes": None,
        },
        index=db.index,
    )
    gene_idx = (
        db["ensembl_gene_id"]
        .map(models.gene_index)
        .fillna(-1)
        .astype(np.int64)
    ).to_numpy()
    out.loc[gene_idx < 0, "universe_reason"] = "no_transcript"
    has = gene_idx >= 0
    out.loc[has, "transcript_id"] = models.selection[
        "transcript_id"
    ].to_numpy()[gene_idx[has]]
    pos = db["Start_Position"].to_numpy().astype(np.int64)
    ref_g = db["Reference_Allele"].astype(str).str.upper().to_numpy()
    alt_g = db["Tumor_Seq_Allele2"].astype(str).str.upper().to_numpy()
    ref_code = _CODE[
        np.array(
            [ord(r[0]) if len(r) == 1 else 0 for r in ref_g],
            dtype=np.uint8,
        )
    ]
    alt_code = _CODE[
        np.array(
            [ord(a[0]) if len(a) == 1 else 0 for a in alt_g],
            dtype=np.uint8,
        )
    ]
    strand = models.selection["strand"].to_numpy()
    minus = np.zeros(n, dtype=bool)
    minus[has] = strand[gene_idx[has]] == "-"
    ref_c = np.where(minus, _COMPLEMENT_CODE[ref_code], ref_code)
    alt_c = np.where(minus, _COMPLEMENT_CODE[alt_code], alt_code)

    # Coding positions keyed by (gene, genomic position).
    coding = np.flatnonzero(models.gpos >= 0)
    keys = (
        models.gene_of[coding].astype(np.int64) * (1 << 32)
        + models.gpos[coding]
    )
    order = np.argsort(keys, kind="stable")
    query = gene_idx * (1 << 32) + pos
    flat = np.where(
        has, _lookup(keys[order], coding[order], query), -1
    )

    # Splice sites keyed the same way.
    sp = models.splice
    skeys = (
        sp["gene"].to_numpy().astype(np.int64) * (1 << 32)
        + sp["gpos"].to_numpy()
    )
    sorder = np.argsort(skeys, kind="stable")
    srow = np.where(
        has & (flat < 0),
        _lookup(skeys[sorder], np.arange(len(sp))[sorder], query),
        -1,
    )

    channel = np.array(out["channel"], dtype=object)
    reason = np.array(out["universe_reason"], dtype=object)
    label = np.array(out["variant_label"], dtype=object)
    routes = np.array(out["routes"], dtype=object)
    inside = np.zeros(n, dtype=bool)
    genes = db["gene"].astype(str).to_numpy()
    types = db["type"].astype(str).to_numpy()
    codes = models.codes
    local = models.local
    offsets = models.offsets

    route_cache = {}
    for r in np.flatnonzero(flat >= 0):
        i = flat[r]
        if codes[i] != ref_c[r] or alt_c[r] >= 4:
            reason[r] = "ref_mismatch"
            continue
        g = models.gene_of[i]
        li = local[i]
        q = li % 3
        cs = i - q
        n_pad = int(models.selection["n_pad"].iat[g])
        codon_codes = codes[cs : cs + 3]
        complete = (
            cs + 2 < offsets[g] + models.lengths[g]
            and (codon_codes < 4).all()
        )
        if complete:
            codon = "".join(_BASES[c] for c in codon_codes)
            new = codon[:q] + _BASES[alt_c[r]] + codon[q + 1 :]
            aa_ref, aa_alt = _CODONS[codon], _CODONS[new]
            number = li // 3 + 1
            if aa_ref == aa_alt:
                ch, tag = "syn", f"p.{aa_ref}{number}="
                target = None
            elif aa_alt == "*":
                ch, tag, target = "non", f"p.{aa_ref}{number}*", "*"
            else:
                ch, tag, target = (
                    "mis",
                    f"p.{aa_ref}{number}{aa_alt}",
                    aa_alt,
                )
            key = (cs, target, ch)
            if key not in route_cache:
                route_cache[key] = ";".join(
                    _routes(models, cs, target, ch, coding_in)
                )
            routes[r] = route_cache[key]
        else:
            ch = "mis"
            tag = f"c.{li - n_pad + 1}{_BASES[codes[i]]}>{_BASES[alt_c[r]]}"
            routes[r] = types[r]
        channel[r] = ch
        label[r] = f"{genes[r]} {tag}"
        reason[r] = "coding"
        inside[r] = True if coding_in is None else bool(coding_in[i])

    kinds = sp["kind"].to_numpy()
    anchors = sp["anchor"].to_numpy()
    for r in np.flatnonzero(srow >= 0):
        s = srow[r]
        if alt_c[r] >= 4 or ref_c[r] >= 4:
            reason[r] = "ref_mismatch"
            continue
        channel[r] = "spl"
        reason[r] = "splice"
        label[r] = (
            f"{genes[r]} c.{anchors[s]}{_SPLICE_OFFSETS[kinds[s]]}"
            f"{_BASES[ref_c[r]]}>{_BASES[alt_c[r]]}"
        )
        routes[r] = types[r]
        inside[r] = True if splice_in is None else bool(splice_in[s])

    classified = np.isin(reason, ["coding", "splice"])
    reason = np.where(
        classified & ~inside, "outside_territory", reason
    )
    # Explicit object dtype: pandas 3 would otherwise infer its string
    # dtype and turn the None of an unclassified call into NaN.
    for column, values in (
        ("channel", channel),
        ("universe_reason", reason),
        ("variant_label", label),
        ("routes", routes),
    ):
        out[column] = pd.Series(values, index=out.index, dtype=object)
    out["in_universe"] = classified & inside
    return out


# MAF Variant_Classification -> the channel it names, for the
# agreement table. Classes that name no channel map to None.
MAF_CLASS_CHANNEL = {
    "Silent": "syn",
    "Missense_Mutation": "mis",
    "Nonstop_Mutation": "mis",
    "Translation_Start_Site": "mis",
    "Nonsense_Mutation": "non",
    "Splice_Site": "spl",
}


def agreement_table(db, same_transcript_only=True):
    """Cross-tabulate the MAF's class against our channel.

    Parameters
    ----------
    db : pandas.DataFrame
        A classified mutation table (``channel``, ``universe_reason``,
        ``transcript_id``, ``Variant_Classification`` and, for the
        same-transcript restriction, ``Transcript_ID``).
    same_transcript_only : bool
        Keep only calls whose MAF transcript (VEP's pick) is the one
        chosen here, where the two labels should agree.

    Returns
    -------
    pandas.DataFrame
        Rows: MAF class; columns: our channel, or the reason a call
        has none.
    """
    sub = db
    if same_transcript_only:
        ours = sub["transcript_id"].astype(str).str.split(".").str[0]
        theirs = (
            sub["Transcript_ID"].astype(str).str.split(".").str[0]
        )
        sub = sub[ours == theirs]
    column = sub["channel"].where(
        sub["channel"].notna(), "(" + sub["universe_reason"] + ")"
    )
    return pd.crosstab(
        sub["Variant_Classification"], column, margins=True
    )


def model_calls(db, scope=None):
    """The calls a model input reads, optionally one consequence scope.

    With the channel universe (a ``in_universe`` column), only
    in-universe calls, and the scope is read off ``channel``:
    ``"silent"`` is ``syn`` and ``"non-silent"`` is ``mis``, ``non``
    or ``spl``. A call on a germline-masked allele (a ``True`` in
    ``germline_masked``, see
    :meth:`models.MutationDataset.set_germline_mask`) is left out too:
    the opportunity gives that allele no weight, so the data must not
    count the few calls a pipeline rescued there. A table classified
    before the channel universe existed has no such column and keeps
    the old behaviour: every call, scoped by the MAF's
    ``Variant_Classification``.

    Parameters
    ----------
    db : pandas.DataFrame
        A mutation table.
    scope : {None, "all", "any", "silent", "non-silent"}

    Returns
    -------
    pandas.DataFrame
        A row subset of ``db``.
    """
    if scope not in (None, "all", "any", "silent", "non-silent"):
        raise ValueError(
            "scope must be None, 'all', 'any', 'silent' or "
            f"'non-silent'; got {scope!r}."
        )
    if "in_universe" in db.columns:
        keep = db["in_universe"].astype(bool)
        if "germline_masked" in db.columns:
            keep &= ~db["germline_masked"].fillna(False).astype(bool)
        db = db[keep]
        if scope == "silent":
            return db[db["channel"] == "syn"]
        if scope == "non-silent":
            return db[db["channel"].isin(NONSYN_CHANNELS)]
        return db
    if scope == "silent":
        return db[db["Variant_Classification"] == "Silent"]
    if scope == "non-silent":
        return db[db["Variant_Classification"] != "Silent"]
    return db


def resolve_keep_ids_for_universe(restrict_to_db):
    """Gene ids from a ``restrict_to_db`` argument, or None for all."""
    from .contexts_by_gene import resolve_keep_ids

    return resolve_keep_ids(restrict_to_db)


# ---------------------------------------------------------------------
# Splice-site flank composition, from a genome
# ---------------------------------------------------------------------


def sigprofiler_genome_dir(build="GRCh38"):
    """SigProfilerMatrixGenerator's installed reference, or None.

    SigProfilerMatrixGenerator (a dependency of this package) keeps
    each installed genome as one ``<chrom>.txt`` file per chromosome
    under ``references/chromosomes/tsb/<build>``: one byte per base,
    the base being ``byte % 4`` (A, C, G, T) for bytes below 16 and N
    otherwise (the higher bits carry transcriptional-strand
    annotation). It is present only once ``genInstall`` has run.
    """
    try:
        import SigProfilerMatrixGenerator
    except ImportError:
        return None
    path = (
        Path(SigProfilerMatrixGenerator.__file__).parent
        / "references"
        / "chromosomes"
        / "tsb"
        / build
    )
    return path if path.is_dir() else None


def splice_flank_composition(
    models, genome_dir=None, rules=("mane_select",)
):
    """Base composition at donor +3 and acceptor -3, read from a genome.

    Reproduces :data:`DONOR_PLUS3_COMPOSITION` and
    :data:`ACCEPTOR_MINUS3_COMPOSITION`: over every coding intron of
    the chosen transcripts (restricted to genes whose transcript came
    from ``rules``; ``None`` keeps all), reads the canonical
    dinucleotides and the third intronic base at each end, on the
    coding strand, and returns the frequencies over GT-AG introns.
    Each chromosome file is memory-mapped, so only the bytes at the
    splice sites are read.

    Parameters
    ----------
    models : TranscriptModels
    genome_dir : path, optional
        A directory of per-chromosome files in SigProfilerMatrixGenerator's
        ``tsb`` format (see :func:`sigprofiler_genome_dir`, the default).
    rules : tuple of str or None

    Returns
    -------
    dict
        ``donor_plus3`` and ``acceptor_minus3`` (A/C/G/T frequencies),
        ``introns``, ``canonical_introns`` and ``gt_ag_fraction``.
    """
    genome_dir = (
        Path(genome_dir) if genome_dir else sigprofiler_genome_dir()
    )
    if genome_dir is None:
        raise FileNotFoundError(
            "No genome: install SigProfilerMatrixGenerator's GRCh38 "
            "reference (genInstall) or pass genome_dir."
        )
    sel = models.selection
    sp = models.splice.copy()
    if rules is not None:
        keep = np.flatnonzero(sel["rule"].isin(rules).to_numpy())
        sp = sp[sp["gene"].isin(keep)]
    chrom = sel["chrom"].to_numpy()[sp["gene"].to_numpy()]
    minus = sel["strand"].to_numpy()[sp["gene"].to_numpy()] == "-"
    donor = sp["kind"].str.startswith("d").to_numpy()
    pos = sp["gpos"].to_numpy()
    # the third intronic base sits one step further from the exon
    # than the +2 / -2 site, in transcript direction
    step = np.where(minus, -1, 1) * np.where(donor, 1, -1)
    base = np.full(len(sp), 4, dtype=np.int8)
    third = np.full(len(sp), 4, dtype=np.int8)
    for c in pd.unique(chrom):
        path = genome_dir / f"{str(c).removeprefix('chr')}.txt"
        if not path.exists():
            continue
        genome = np.memmap(path, dtype=np.uint8, mode="r")
        idx = np.flatnonzero(chrom == c)
        for target, at in (
            (base, pos[idx]),
            (third, pos[idx] + step[idx]),
        ):
            ok = (at >= 1) & (at <= len(genome))
            raw = np.full(len(idx), 16, dtype=np.uint8)
            raw[ok] = genome[at[ok] - 1]
            target[idx] = np.where(raw < 16, raw % 4, 4)
    base = np.where(minus, _COMPLEMENT_CODE[base], base)
    third = np.where(minus, _COMPLEMENT_CODE[third], third)

    sp = sp.assign(base=base, third=third)
    sp["intron"] = np.where(donor, sp["anchor"], sp["anchor"] - 1)
    wide = sp.pivot_table(
        index=["gene", "intron"],
        columns="kind",
        values=["base", "third"],
    )
    A, G, T = 0, 2, 3
    canonical = (
        (wide[("base", "d1")] == G)
        & (wide[("base", "d2")] == T)
        & (wide[("base", "a2")] == A)
        & (wide[("base", "a1")] == G)
    )

    def freq(values):
        counts = np.bincount(values.astype(int), minlength=5)[:4]
        return {
            b: float(counts[i] / counts.sum())
            for i, b in enumerate(_BASES)
        }

    return {
        "donor_plus3": freq(
            wide.loc[canonical, ("third", "d2")].to_numpy()
        ),
        "acceptor_minus3": freq(
            wide.loc[canonical, ("third", "a2")].to_numpy()
        ),
        "introns": len(wide),
        "canonical_introns": int(canonical.sum()),
        "gt_ag_fraction": float(canonical.mean()),
    }
