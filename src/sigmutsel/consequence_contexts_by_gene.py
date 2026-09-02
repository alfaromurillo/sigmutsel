"""Consequence-split (synonymous/non-synonymous) opportunity counts.

:func:`contexts_by_gene.compute_contexts_by_gene` counts the 32
pyrimidine-centred trinucleotide *contexts* of each gene's CDS.  A
context says nothing about what a mutation *does*: the very same
context/substitution (e.g. C>T at an NCG context) is synonymous at one
codon position and missense at another, purely depending on reading
frame.

This module adds that missing axis.  It walks the same longest-CDS
sequence per gene, codon by codon, and splits every substitution
opportunity into a **synonymous** and a **non-synonymous** channel,
bucketed by the 96 canonical SBS types (:data:`constants.
canonical_types_order`) rather than by the 32 contexts --- synonymy
depends on *which* alternate base is substituted, so the split has to
be finer-grained than the context table.

The two tables are built so that, for every gene *g* and every SBS
type *τ*::

    contexts_by_gene_syn[τ] + contexts_by_gene_nonsyn[τ]
        == contexts_by_gene[extract_context(τ)]

i.e. each of the 3 types sharing a context splits exactly that
context's count.  This identity is what guarantees, downstream, that
``p_gτ^(syn) + p_gτ^(nonsyn) == p_gτ`` for the same denominator
:func:`estimate_mus.compute_mu_g_per_tumor` already uses, and hence
that ``μ_g^(syn) + μ_g^(nonsyn) == μ_g`` exactly.  It is checked
directly in ``tests/test_consequence_contexts_by_gene.py``.

Caveats (inherited or structural, none of them fixed here)
---------------------------------------------------------
* **SBS only.**  "Synonymous" is a codon-level concept that only makes
  sense for single-base substitutions; it has no clean analogue for
  DBS/ID/CN/SV.  Callers must gate on ``signature_class == "SBS"``
  (:meth:`models.MutationDataset.generate_consequence_contexts_by_gene`
  does).
* **Exon-junction flanking bases.**  The Ensembl CDS FASTA is spliced,
  so *codons* --- and therefore every consequence call here --- are
  exact.  The *trinucleotide context* is not: at an exon--exon
  junction the flanking base comes from the neighbouring exon rather
  than the true intronic neighbour (≈1.3% of CDS positions, ≈0.95%
  with an actually-wrong context).  That is inherited unchanged from
  :func:`contexts_by_gene.compute_contexts_by_gene`, and keeping it
  identical is what preserves the sum identity above.
* **Essential splice sites are intronic** and therefore absent from a
  CDS-FASTA-based opportunity model entirely --- they are not part of
  either channel here.  Flagged as an open scope question, not
  resolved.
* **Reading frame is assumed to start at position 0** of each CDS
  record, which holds for complete Ensembl CDS entries but not for
  5'-incomplete ones (≈2.8% of the genes in
  ``Homo_sapiens.GRCh38.cds.all.fa`` have a length that is not a
  multiple of 3, and the FASTA does not say which end is incomplete).
  Positions whose codon is truncated or contains a non-ACGT base are
  counted as non-synonymous (see
  :func:`compute_consequence_contexts_by_gene`'s Notes); measured over
  the whole FASTA that is 537 of 39.6M counted positions (0.001%).
  Both are reported in the log.
* **Stop codons participate.**  A substitution inside a stop codon
  that keeps it a stop (e.g. TAA>TAG) translates identically and is
  therefore counted as synonymous; stop-loss and nonsense are
  non-synonymous.
"""

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from Bio.Seq import Seq

from .constants import canonical_types_order, reverse_complement
from .contexts_by_gene import (
    normalize_fasta_paths,
    resolve_keep_ids,
    select_longest_sequences,
)

logger = logging.getLogger(__name__)

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _build_codon_table() -> dict[str, str]:
    """Translate all 64 unambiguous codons once, via Biopython.

    Done up front so the per-position walk is a dict lookup rather
    than 10^8 calls into :meth:`Bio.Seq.Seq.translate`.
    """
    codon_table = {}
    for first in "ACGT":
        for second in "ACGT":
            for third in "ACGT":
                codon = first + second + third
                codon_table[codon] = str(Seq(codon).translate())

    return codon_table


def _build_synonymous_alts() -> dict[str, tuple[frozenset, ...]]:
    """Map each codon to its synonymous alternates, per codon position.

    ``_SYNONYMOUS_ALTS[codon][q]`` is the set of bases that, when
    substituted at position ``q`` (0, 1 or 2) of ``codon``, leave the
    translated amino acid (or stop) unchanged.
    """
    codon_table = _build_codon_table()

    synonymous_alts = {}
    for codon, residue in codon_table.items():
        per_position = []
        for position in range(3):
            reference = codon[position]
            per_position.append(
                frozenset(
                    alt
                    for alt in "ACGT"
                    if alt != reference
                    and codon_table[
                        codon[:position] + alt + codon[position + 1 :]
                    ]
                    == residue
                )
            )
        synonymous_alts[codon] = tuple(per_position)

    return synonymous_alts


def _build_substitutions_by_trinucleotide() -> (
    dict[str, tuple[tuple[str, str], ...]]
):
    """Map a coding-strand trinucleotide to its 3 SBS opportunities.

    ``_SUBSTITUTIONS[tri]`` is a 3-tuple of ``(alt, sbs_type)`` where
    *alt* is the alternate base **on the coding strand** (what the
    consequence call needs) and *sbs_type* is the strand-collapsed,
    pyrimidine-centred canonical type (what the output columns are).

    The two differ whenever the central base is a purine: the type is
    then read off the reverse complement, exactly as
    :func:`contexts_by_gene.compute_contexts_by_gene` collapses purine-
    centred contexts onto their pyrimidine-centred partner.
    """
    substitutions = {}
    for first in "ACGT":
        for middle in "ACGT":
            for third in "ACGT":
                trinucleotide = first + middle + third
                labelled = []
                for alt in "ACGT":
                    if alt == middle:
                        continue
                    if middle in "CT":
                        left, ref, right = first, middle, third
                        collapsed_alt = alt
                    else:
                        left, ref, right = reverse_complement(
                            trinucleotide
                        )
                        collapsed_alt = alt.translate(_COMPLEMENT)
                    labelled.append(
                        (alt, f"{left}[{ref}>{collapsed_alt}]{right}")
                    )
                substitutions[trinucleotide] = tuple(labelled)

    return substitutions


_SYNONYMOUS_ALTS = _build_synonymous_alts()
_SUBSTITUTIONS = _build_substitutions_by_trinucleotide()

# Guard the two precomputed tables against a silent mismatch with the
# canonical type list they are meant to index.
_ALL_LABELS = {
    sbs_type
    for opportunities in _SUBSTITUTIONS.values()
    for _, sbs_type in opportunities
}
if _ALL_LABELS != set(canonical_types_order):
    raise RuntimeError(
        "Consequence-split substitution table does not cover the 96 "
        "canonical SBS types exactly."
    )


def compute_consequence_contexts_by_gene(
    fasta_files: str | Path | list[str | Path] | None = None,
    restrict_to_db: pd.DataFrame | Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each gene's SBS opportunities into syn/non-syn channels.

    Uses the same FASTA resolution, gene restriction and
    longest-transcript selection as
    :func:`contexts_by_gene.compute_contexts_by_gene` (the helpers are
    shared, not reimplemented), then walks each selected sequence
    codon by codon and classifies every (position, alternate base)
    opportunity as synonymous or non-synonymous.

    Parameters
    ----------
    fasta_files : str | Path | list[str | Path] | None
        One path or a list of paths to Ensembl **CDS** FASTA files.
        If None, defaults to ``locations.location_cds_fasta``.  A
        non-coding FASTA has no reading frame and would classify
        every opportunity as non-synonymous; the log reports the
        undetermined fraction so that case is visible rather than
        silent.
    restrict_to_db : pandas.DataFrame | iterable[str] | None, optional
        • *None* (default) -- process **all** genes in the FASTA(s).
        • DataFrame with an ``'ensembl_gene_id'`` column *or* any
          iterable of Ensembl IDs -- compute counts only for those IDs.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(contexts_by_gene_syn, contexts_by_gene_nonsyn)``.  Both are
        indexed by stable Ensembl gene ID (version stripped, sorted)
        and have the 96 canonical SBS types as columns, in
        ``constants.canonical_types_order``.

    Notes
    -----
    Which positions are counted is dictated by the identity this
    function has to preserve (see the module docstring): exactly the
    positions :func:`contexts_by_gene.compute_contexts_by_gene`
    counts, i.e. every centre position from 1 to ``len(seq) - 2``
    whose trinucleotide window is free of non-ACGT bases.  Each such
    position contributes 1 opportunity to each of the 3 SBS types
    sharing its (strand-collapsed) context, landing in exactly one of
    the two channels.

    A position whose *codon* cannot be resolved --- truncated at the
    end of a CDS whose length is not a multiple of 3, or containing a
    non-ACGT base --- has an undetermined consequence.  Such positions
    are counted as **non-synonymous**, which mirrors the deck's
    definition of the non-synonymous channel as the remainder
    (``p^(nonsyn) = p_gτ - p^(syn)``) and keeps the sum identity
    exact.  They are rare in a complete CDS FASTA and their total is
    logged.
    """
    logger.info(
        "Splitting SBS opportunities into synonymous/non-synonymous "
        "channels...."
    )

    fasta_paths = normalize_fasta_paths(fasta_files)
    keep_ids = resolve_keep_ids(restrict_to_db)
    best_seq = select_longest_sequences(fasta_paths, keep_ids)

    substitutions = _SUBSTITUTIONS
    synonymous_alts = _SYNONYMOUS_ALTS

    genes = []
    syn_rows = []
    nonsyn_rows = []
    n_positions = 0
    n_undetermined = 0
    n_partial_genes = 0

    for ensg, seq in best_seq.items():
        syn = dict.fromkeys(canonical_types_order, 0)
        nonsyn = dict.fromkeys(canonical_types_order, 0)

        if len(seq) % 3:
            n_partial_genes += 1

        for centre in range(1, len(seq) - 1):
            opportunities = substitutions.get(
                seq[centre - 1 : centre + 2]
            )
            if opportunities is None:
                # non-ACGT in the context window: skipped by
                # compute_contexts_by_gene too
                continue

            n_positions += 1
            codon_start = centre - centre % 3
            codon_alts = synonymous_alts.get(
                seq[codon_start : codon_start + 3]
            )

            if codon_alts is None:
                # truncated or ambiguous codon: consequence
                # undetermined, folded into the non-synonymous channel
                n_undetermined += 1
                for _, sbs_type in opportunities:
                    nonsyn[sbs_type] += 1
                continue

            position_alts = codon_alts[centre % 3]
            for alt, sbs_type in opportunities:
                if alt in position_alts:
                    syn[sbs_type] += 1
                else:
                    nonsyn[sbs_type] += 1

        genes.append(ensg)
        syn_rows.append(syn)
        nonsyn_rows.append(nonsyn)

    if n_partial_genes:
        logger.warning(
            f"{n_partial_genes} of {len(genes)} sequences have a "
            "length that is not a multiple of 3, i.e. an incomplete "
            "CDS. Their trailing codon is undetermined, and if it is "
            "the 5' end that is incomplete every codon in them is "
            "read out of frame (Ensembl's CDS FASTA does not say "
            "which end)."
        )
    if n_undetermined:
        logger.warning(
            f"{n_undetermined} of {n_positions} counted positions "
            f"({100 * n_undetermined / n_positions:.2f}%) had an "
            "undetermined consequence and were counted as "
            "non-synonymous. A large fraction here means the input "
            "FASTA is not a coding-sequence FASTA."
        )

    def _to_frame(rows):
        return (
            pd.DataFrame(
                rows, index=genes, columns=canonical_types_order
            )
            .fillna(0)
            .astype(int)
            .sort_index()
        )

    logger.info("...done.")
    return _to_frame(syn_rows), _to_frame(nonsyn_rows)
