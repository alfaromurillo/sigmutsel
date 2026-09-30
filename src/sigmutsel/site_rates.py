"""Site rates: the rate of any single-nucleotide variant, on demand.

A channel model's variant rate factorises. For an opportunity element
``o`` (a coding position with one alternate base, or an essential
splice site with one) of type ``tau`` in gene ``g``, channel ``h`` and
tumor ``j``,

    mu^j_o = M[j, tau] * upsilon_o * m_{g,h},
    M[j, tau] = mu_bar^j_tau / D_tau,

with ``mu_bar^j_tau`` the tumor's type rate, ``D_tau`` the weighted
opportunity of the type over the gene universe
(:func:`estimate_mus.type_denominators`), ``upsilon_o`` the element's
site weight (:mod:`site_weights`; 1 at splice sites) and ``m_{g,h}``
the gene multiplier: the covariate scale ``e^{c.x_g}`` (a fallback
gene's projected scale and block shifts included), ``e^delta`` on the
non-synonymous channels (``mis``, ``non``, ``spl``), and optionally
``r_g``. The gene's channel opportunity ``n^h_{g tau}`` cancels: the
model spreads its type rate evenly over the weighted elements it was
built from, so ``M`` is one tumors x 96 matrix shared by every channel
and gene. A protein change's rate sums its routes' rates (``R(m)``,
:func:`channel_universe._routes`).

Nothing here is a variants x tumors table (that would be ~400 GB for a
large cohort): :class:`SiteRates` keeps the site table, ``M`` and one
multiplier per gene and channel, and computes rates for whatever is
asked, per tumor or summed over tumors, including aggregates over any
grouping of the elements.

Masked alleles
--------------
An allele of the germline mask is not opportunity (a call there is
filtered out, so the model gives it ``upsilon_o = 0``), but it still
mutates. Every query reports both: ``rate`` is the **mutation rate**,
each element at its unmasked weight, and ``observable_rate`` the rate
of an observable call, masked elements at 0. Use the observable rate
for anything compared with calls -- expected counts, gamma, the
recurrence check -- and the mutation rate for simulating mutations.
:func:`estimate_mus.compute_mu_m_per_tumor` (``mu_ms``) gives every
route its unmasked weight, so it is the mutation rate; the two agree
for a variant with no masked route.

Splice contexts
---------------
The opportunity spreads a donor +2 or acceptor -2 site over the four
contexts its missing intronic base could give
(:data:`channel_universe.DONOR_PLUS3_COMPOSITION`), and
:meth:`SiteRates.opportunity` does the same, so aggregates reproduce
the model's tables. A query for one splice change instead reads the
site's trinucleotide from a genome when one is available (the call's
own context, as ``mu_ms`` uses it) and falls back to the composition
mixture otherwise; ``context`` in the result says which.
"""

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .channel_universe import (
    _BASES,
    _CODE,
    _CODONS,
    _COMPLEMENT_CODE,
    _TYPE_TABLE,
    ACCEPTOR_MINUS3_COMPOSITION,
    CHANNELS,
    DONOR_PLUS3_COMPOSITION,
    _coding_site_table,
    _normalised,
    _routes,
    _splice_site_weights,
)
from .constants import canonical_types_order

logger = logging.getLogger(__name__)

#: Channel of each consequence code of the site table (0 syn, 1 mis,
#: 2 non; 3 is the splice channel).
_NONSYN = np.array([False, True, True, True])
_SPLICE_KIND = {"+1": "d1", "+2": "d2", "-2": "a2", "-1": "a1"}
_KIND_REF = {"d1": 2, "d2": 3, "a2": 0, "a1": 2}  # G, T, A, G
_P_LABEL = re.compile(r"^p\.([A-Z*])(\d+)([A-Z*=])$")
_C_CODING = re.compile(r"^c\.(\d+)([ACGT])>([ACGT])$")
_C_SPLICE = re.compile(r"^c\.(\d+)([+-][12])([ACGT])>([ACGT])$")


@dataclass
class SiteTable:
    """Every opportunity element of a gene universe, as flat arrays.

    Coding elements are stored per position (``idx`` into the
    :class:`channel_universe.TranscriptModels` arrays) with one column
    per alternate base ``(ref + k) % 4``, ``k = 1, 2, 3``, exactly as
    :func:`channel_universe._coding_site_table` enumerates them.
    Splice elements are the triples of
    :func:`channel_universe._splice_site_weights` (contexts spread by
    composition), with and without the mask.

    Attributes
    ----------
    models : TranscriptModels
    genes : numpy.ndarray of str
        The gene universe (model genes), in the rate model's order.
    gene_row : numpy.ndarray of int
        Per transcript-model gene, its row in ``genes`` or -1.
    idx, dbin, orient : (n,) arrays
        Flat position, distance bin and strand orientation.
    types, cons : (n, 3) int8
        Type index and consequence (0 syn, 1 mis, 2 non) per alt.
    masked : (n, 3) bool
        Allele in the germline mask.
    splice_gene, splice_type : arrays
        Gene row and type index of each splice triple.
    splice_weight, splice_observable : arrays
        Its weight before and after the mask.
    site_weights : SiteWeights or None
    coding_in, splice_in : territory masks, or None
    """

    models: object
    genes: np.ndarray
    gene_row: np.ndarray
    idx: np.ndarray
    dbin: np.ndarray
    orient: np.ndarray
    types: np.ndarray
    cons: np.ndarray
    masked: np.ndarray
    splice_gene: np.ndarray
    splice_type: np.ndarray
    splice_weight: np.ndarray
    splice_observable: np.ndarray
    site_weights: object = None
    coding_in: object = None
    splice_in: object = None
    masked_keys: object = None
    _genomic: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        models,
        genes,
        coding_in=None,
        splice_in=None,
        masked_keys=None,
        site_weights=None,
    ):
        """Enumerate the elements of ``genes`` (stable Ensembl ids)."""
        from .site_weights import position_bins

        genes = np.asarray(genes, dtype=object)
        gene_row = np.full(len(models.gene_ids), -1, dtype=np.int64)
        row_of = {g: i for i, g in enumerate(genes)}
        for i, g in enumerate(models.gene_ids):
            gene_row[i] = row_of.get(g, -1)
        missing = [g for g in genes if g not in models.gene_index]
        if missing:
            raise ValueError(
                f"{len(missing)} universe genes have no transcript "
                f"model (e.g. {missing[:3]}); the site table cannot "
                "reproduce their opportunity."
            )

        gene, types, cons, idx, _ = _coding_site_table(
            models, coding_in
        )
        keep = gene_row[gene] >= 0
        idx, types, cons = idx[keep], types[keep], cons[keep]
        dbin, orient = position_bins(models)
        masked = np.zeros(types.shape, dtype=bool)
        if masked_keys is not None:
            masked = _masked_alleles(models, idx, masked_keys)

        sg, st, sw = _splice_site_weights(models, splice_in)
        if masked_keys is not None:
            _, _, mw = _splice_site_weights(
                models, splice_in, masked_keys=masked_keys
            )
        else:
            mw = np.zeros_like(sw)
        sk = gene_row[sg] >= 0 if len(sg) else np.zeros(0, bool)
        return cls(
            models=models,
            genes=genes,
            gene_row=gene_row,
            idx=idx,
            dbin=dbin[idx],
            orient=orient[idx],
            types=types.astype(np.int8),
            cons=cons.astype(np.int8),
            masked=masked,
            splice_gene=gene_row[sg[sk]] if len(sg) else sg,
            splice_type=st[sk].astype(np.int64),
            splice_weight=sw[sk],
            splice_observable=(sw - mw)[sk],
            site_weights=site_weights,
            coding_in=coding_in,
            splice_in=splice_in,
            masked_keys=masked_keys,
        )

    @property
    def gene(self):
        """Gene row of each coding position."""
        return self.gene_row[self.models.gene_of[self.idx]]

    def weights(self, rows=slice(None)):
        """``upsilon_o`` before the mask, (n, 3), for ``rows``."""
        from .site_weights import weights_at

        return weights_at(
            self.dbin[rows],
            self.orient[rows],
            self.types[rows].astype(np.int64),
            self.site_weights,
        )

    def position_rows(self, flat):
        """Row of each flat position in the table, or -1."""
        flat = np.asarray(flat, dtype=np.int64)
        j = np.searchsorted(self.idx, flat)
        j = np.clip(j, 0, max(len(self.idx) - 1, 0))
        hit = len(self.idx) > 0
        return np.where(hit & (self.idx[j] == flat), j, -1)

    def genomic_positions(self, chrom, pos):
        """All flat coding positions at (chrom, pos): (query, flat)."""
        if "keys" not in self._genomic:
            m = self.models
            coding = np.flatnonzero(m.gpos >= 0)
            keys = _chrom_codes(m.chrom_of[coding]) * (1 << 31) + (
                m.gpos[coding]
            )
            order = np.argsort(keys, kind="stable")
            self._genomic["keys"] = keys[order]
            self._genomic["flat"] = coding[order]
        keys, flat = self._genomic["keys"], self._genomic["flat"]
        q = _chrom_codes(chrom) * (1 << 31) + np.asarray(
            pos, dtype=np.int64
        )
        lo = np.searchsorted(keys, q, side="left")
        hi = np.searchsorted(keys, q, side="right")
        qi = np.repeat(np.arange(len(q)), hi - lo)
        starts = np.repeat(lo, hi - lo)
        offs = np.arange(len(qi)) - np.repeat(
            np.cumsum(hi - lo) - (hi - lo), hi - lo
        )
        return qi, flat[starts + offs]


def _chrom_codes(chrom):
    from .germline_mask import _CHROM_CODE

    return (
        pd.Series(np.asarray(chrom))
        .astype(str)
        .str.replace("chr", "", regex=False)
        .map(_CHROM_CODE)
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )


def _masked_alleles(models, idx, masked_keys, chunk=5_000_000):
    """(n, 3) bool: which alternate bases of each position are masked."""
    from .germline_mask import allele_keys

    minus_gene = models.selection["strand"].to_numpy() == "-"
    out = np.zeros((len(idx), 3), dtype=bool)
    for s in range(0, len(idx), chunk):
        i = idx[s : s + chunk]
        ref = models.codes[i].astype(np.int64)
        alts = np.stack([(ref + k) % 4 for k in (1, 2, 3)], axis=1)
        minus = minus_gene[models.gene_of[i]]
        genomic_alt = np.where(minus[:, None], 3 - alts, alts)
        site = allele_keys(models.chrom_of[i], models.gpos[i], 0)
        out[s : s + chunk] = np.isin(
            site[:, None] + genomic_alt, masked_keys
        )
    return out


def _edge_consequence(models, flat, coding_alt):
    """Consequence code (0 syn, 1 mis, 2 non) of coding substitutions.

    From the codon; an incomplete codon, or one with an N, is ``mis``
    as in :func:`channel_universe._coding_site_table`.
    """
    from .channel_universe import _CONSEQUENCE_TABLE

    out = np.ones(len(flat), dtype=np.int64)
    for j, (i, a) in enumerate(zip(flat, coding_alt)):
        g = models.gene_of[i]
        q = int(models.local[i] % 3)
        cs = i - q
        end = models.offsets[g] + models.lengths[g]
        codon = models.codes[cs : cs + 3]
        if cs + 2 < end and (codon < 4).all():
            out[j] = _CONSEQUENCE_TABLE[
                codon[0], codon[1], codon[2], q, a
            ]
    return out


def _genome_bases(genome_dir, chrom, gpos):
    """Plus-strand base codes (0..3; 4 unknown) from a ``tsb`` genome."""
    from pathlib import Path

    out = np.full(len(gpos), 4, dtype=np.int8)
    if genome_dir is None:
        return out
    chrom = np.asarray(chrom).astype(str)
    gpos = np.asarray(gpos, dtype=np.int64)
    for c in pd.unique(chrom):
        path = Path(genome_dir) / f"{c.removeprefix('chr')}.txt"
        if not path.exists():
            continue
        genome = np.memmap(path, dtype=np.uint8, mode="r")
        sel = np.flatnonzero(chrom == c)
        a = gpos[sel]
        ok = (a >= 1) & (a <= len(genome))
        raw = np.full(len(sel), 16, dtype=np.uint8)
        raw[ok] = genome[a[ok] - 1]
        out[sel] = np.where(raw < 16, raw % 4, 4)
    return out


def tumor_type_rates(mu_taus, denominators):
    """``M = mu_bar^j_tau / D_tau``, tumors x 96.

    A type with ``D_tau = 0`` has no element anywhere in the universe,
    so no site ever reads its column; it is 0 rather than a 0/0.
    """
    D = denominators.reindex(canonical_types_order).to_numpy(
        dtype=float
    )
    mu = mu_taus.loc[:, canonical_types_order].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(D > 0, mu / np.where(D > 0, D, 1.0), 0.0)
    return pd.DataFrame(
        M, index=mu_taus.index, columns=canonical_types_order
    )


class SiteRates:
    """Per-tumor rate of any variant, from a factored store.

    Parameters
    ----------
    table : SiteTable
    tumor_type_rates : pandas.DataFrame
        ``M``: tumors x 96, ``mu_bar^j_tau / D_tau``.
    multipliers : pandas.DataFrame
        Genes (``table.genes``) x ``["syn", "nonsyn"]``: ``m_{g,h}``.
    genome_dir : path-like or None
        A ``tsb`` genome for splice-query contexts (None: composition).
    description : dict
        What went into the multipliers (``r_g``, ``delta``), for
        reports.

    Build it from a fitted channel model with :meth:`from_model`.
    """

    def __init__(
        self,
        table,
        tumor_type_rates,
        multipliers,
        genome_dir=None,
        description=None,
    ):
        self.table = table
        self.M = tumor_type_rates.loc[:, canonical_types_order]
        self.tumors = self.M.index
        self._M = self.M.to_numpy(dtype=float)
        self._M_total = self._M.sum(axis=0)
        mult = multipliers.reindex(table.genes)
        self.multipliers = mult
        self._mult = mult[["syn", "nonsyn"]].to_numpy(dtype=float)
        self.genome_dir = genome_dir
        self.description = dict(description or {})
        self._symbol_index = None

    # -----------------------------------------------------------------
    # Construction from a model
    # -----------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model,
        r_g="evaluation",
        transcript_models=None,
        coding_in="dataset",
        splice_in="dataset",
        masked_keys="dataset",
        genome_dir="default",
        verify=True,
    ):
        """The site rates of a fitted channel-split model.

        Parameters
        ----------
        model : Model
            With channel baselines (:meth:`Model.compute_channel_base_mus`)
            on a dataset built with the channel universe, tau-dependent
            ``p_{g tau}`` and a signature-summed ``mu_taus``.
        r_g : {"evaluation", "none"}
            The gene rate correction in the multiplier. ``"evaluation"``
            (silent-channel ``r_g``, point estimate) is the one gamma
            uses; ``"none"`` reproduces ``mu_ms``. The production
            ``r_g`` is not offered: its non-silent expectation omits
            ``e^delta``.
        transcript_models : TranscriptModels, optional
            Default: the cached models the dataset's opportunity uses.
        coding_in, splice_in : "dataset", None or boolean arrays
            Territory masks; "dataset" reads the dataset's territory.
        masked_keys : "dataset", None or int64 array
            Germline-mask alleles; "dataset" loads the dataset's mask.
        genome_dir : "default", None or path
            Genome for splice-query contexts; "default" is the installed
            one (:func:`channel_universe.sigprofiler_genome_dir`).
        verify : bool
            Check that the site table rebuilds the dataset's four
            channel tables (to 1e-9 relative); raises if not.
        """
        from .channel_universe import (
            load_or_build_transcript_models,
            sigprofiler_genome_dir,
            territory_masks,
        )
        from .estimate_mus import (
            compute_mus_per_gene_per_sample,
            type_denominators,
        )

        if r_g not in ("evaluation", "none"):
            raise ValueError(
                f"r_g must be 'evaluation' or 'none', got {r_g!r}. The "
                "production r_g omits e^delta from its non-silent "
                "expectation and is not offered here."
            )
        dataset = model.dataset
        if not dataset.has_channel_universe():
            raise ValueError("Site rates need the channel universe.")
        if not model.has_channel_base_mus():
            raise ValueError(
                "Site rates need a channel-split model; call "
                "compute_channel_base_mus() first."
            )
        if model.prob_g_tau_tau_independent:
            raise NotImplementedError(
                "Site rates are defined for tau-dependent p_{g tau} "
                "only; under tau-independence a site's rate depends on "
                "its gene's context mix."
            )
        mu_taus = model.mu_taus
        if isinstance(mu_taus, dict):
            raise NotImplementedError(
                "Site rates need a signature-summed mu_taus."
            )

        params = dataset.channel_universe or {}
        models = (
            transcript_models or load_or_build_transcript_models()
        )
        if isinstance(coding_in, str) or isinstance(splice_in, str):
            if params.get("territory", "mc3") == "mc3":
                c_in, s_in = territory_masks(
                    models,
                    splice_padding=params.get("splice_padding", 0),
                )
            else:
                c_in = s_in = None
            coding_in = (
                c_in if isinstance(coding_in, str) else coding_in
            )
            splice_in = (
                s_in if isinstance(splice_in, str) else splice_in
            )
        if isinstance(masked_keys, str):
            masked_keys = None
            if dataset.germline_mask_af is not None:
                from .germline_mask import load_germline_mask

                masked_keys = load_germline_mask(
                    dataset.germline_mask_af
                )
        if isinstance(genome_dir, str) and genome_dir == "default":
            genome_dir = sigprofiler_genome_dir()

        contexts = dataset.contexts_by_gene
        table = SiteTable.build(
            models,
            contexts.index.to_numpy(),
            coding_in=coding_in,
            splice_in=splice_in,
            masked_keys=masked_keys,
            site_weights=dataset.site_weights,
        )

        denominators = type_denominators(
            contexts, getattr(dataset, "type_opportunity", None)
        )
        M = tumor_type_rates(mu_taus, denominators)

        # The covariate scale exactly as _compute_mu_g_taus applies it:
        # the same function, over the same genes, on a baseline of 1.
        ones = pd.DataFrame(1.0, index=contexts.index, columns=["x"])
        if (
            model.cov_effects is not None
            and model.cov_matrix is not None
        ):
            scale = compute_mus_per_gene_per_sample(
                db=dataset.mutation_db,
                base_mus=ones,
                cov_effect=model.cov_effects,
                cov_matrix=model.cov_matrix,
                fallback_log_scale=model._fallback_eta(),
            )["x"].reindex(contexts.index)
        else:
            scale = ones["x"]
        delta = model._rg_delta_intercept
        syn = scale.astype(float)
        nonsyn = syn * (np.exp(delta) if delta is not None else 1.0)
        rg = pd.Series(1.0, index=contexts.index)
        if r_g == "evaluation":
            rg = model.compute_r_g_for_evaluation().reindex(
                contexts.index
            )
        multipliers = pd.DataFrame(
            {"syn": syn * rg, "nonsyn": nonsyn * rg}
        )
        out = cls(
            table,
            M,
            multipliers,
            genome_dir=genome_dir,
            description={
                "r_g": r_g,
                "delta": None if delta is None else float(delta),
                "n_genes_without_r_g": int(rg.isna().sum()),
                "germline_mask_af": dataset.germline_mask_af,
                "site_weights": params.get("site_weights"),
            },
        )
        if verify:
            out.verify_opportunity(
                {
                    "syn": dataset.contexts_by_gene_syn,
                    "mis": dataset.contexts_by_gene_mis,
                    "non": dataset.contexts_by_gene_non,
                    "spl": dataset.contexts_by_gene_spl,
                }
            )
        return out

    # -----------------------------------------------------------------
    # Opportunity and aggregates
    # -----------------------------------------------------------------

    def _coding_groups(self, labels):
        """Label code per coding (position, alt), and the label names."""
        n = len(self.table.idx)
        if labels is None:
            return np.zeros((n, 3), dtype=np.int64), ["all"]
        labels = np.asarray(labels)
        if labels.shape == (n,):
            labels = np.repeat(labels[:, None], 3, axis=1)
        if labels.shape != (n, 3):
            raise ValueError(
                "labels must have one entry per coding position or per "
                f"(position, alternate base): {(n,)} or {(n, 3)}."
            )
        names, codes = np.unique(labels.ravel(), return_inverse=True)
        return codes.reshape(n, 3), [str(x) for x in names]

    def opportunity(
        self,
        labels=None,
        splice_label="splice",
        observable=True,
        by_gene=True,
        gene_weights=None,
        chunk=5_000_000,
    ):
        """Weighted element counts per (gene, channel, label) and type.

        ``labels`` groups the coding elements (one per position or per
        position and alternate base); splice elements take
        ``splice_label``. ``observable`` counts masked alleles as 0.
        With ``by_gene=False`` genes are summed, each weighted by
        ``gene_weights[gene row, channel index]`` when given (0 syn,
        1 nonsyn; used by :meth:`aggregate_rates`).

        Returns
        -------
        pandas.DataFrame
            Rows (gene, channel, label) or (channel, label), 96 columns.
        """
        t = self.table
        codes, names = self._coding_groups(labels)
        names = list(names)
        if splice_label not in names:
            names.append(splice_label)
        spl_code = names.index(splice_label)
        n_lab = len(names)
        n_genes = len(t.genes) if by_gene else 1
        size = n_genes * 4 * n_lab * 96
        total = np.zeros(size)
        gene = t.gene
        for s in range(0, len(t.idx), chunk):
            rows = slice(s, s + chunk)
            w = t.weights(rows)
            if observable:
                w = w * ~t.masked[rows]
            g = gene[rows] if by_gene else np.zeros_like(gene[rows])
            if gene_weights is not None:
                ch = _NONSYN[t.cons[rows]].astype(np.int64)
                w = w * gene_weights[gene[rows][:, None], ch]
            key = (
                (g[:, None] * 4 + t.cons[rows]) * n_lab + codes[rows]
            ) * 96 + t.types[rows]
            total += np.bincount(
                key.ravel(), weights=w.ravel(), minlength=size
            )
        sw = t.splice_observable if observable else t.splice_weight
        if gene_weights is not None and len(sw):
            sw = sw * gene_weights[t.splice_gene, 1]
        sg = (
            t.splice_gene if by_gene else np.zeros_like(t.splice_gene)
        )
        key = ((sg * 4 + 3) * n_lab + spl_code) * 96 + t.splice_type
        total += np.bincount(key, weights=sw, minlength=size)

        total = total.reshape(n_genes * 4 * n_lab, 96)
        if by_gene:
            index = pd.MultiIndex.from_product(
                [t.genes, CHANNELS, names],
                names=["ensembl_gene_id", "channel", "label"],
            )
        else:
            index = pd.MultiIndex.from_product(
                [CHANNELS, names], names=["channel", "label"]
            )
        out = pd.DataFrame(
            total, index=index, columns=canonical_types_order
        )
        return out[out.to_numpy().any(axis=1)]

    def verify_opportunity(self, channel_tables, rtol=1e-9):
        """Raise unless the table rebuilds the model's channel tables."""
        opp = self.opportunity(observable=True).droplevel("label")
        for channel in CHANNELS:
            expected = channel_tables[channel].reindex(
                index=self.table.genes, columns=canonical_types_order
            )
            try:
                got = opp.xs(channel, level="channel")
            except KeyError:
                got = expected.iloc[:0]
            got = got.reindex(
                index=self.table.genes, columns=canonical_types_order
            ).fillna(0.0)
            a, b = got.to_numpy(), expected.fillna(0.0).to_numpy()
            if not np.allclose(a, b, rtol=rtol, atol=1e-9):
                bad = np.argwhere(
                    ~np.isclose(a, b, rtol=rtol, atol=1e-9)
                )
                g, k = bad[0]
                raise AssertionError(
                    f"Site table does not rebuild the {channel} "
                    f"opportunity: {len(bad)} cells differ, e.g. "
                    f"{self.table.genes[g]} {canonical_types_order[k]}: "
                    f"{a[g, k]} vs {b[g, k]}."
                )

    def aggregate_rates(
        self,
        labels=None,
        splice_label="splice",
        observable=True,
        by_gene=True,
        per_tumor=True,
    ):
        """Summed rates per (gene, channel, label), without sites x tumors.

        ``sum_o mu^j_o`` over each group: the group's weighted type
        counts times ``M``, times the gene multiplier. ``per_tumor``
        gives groups x tumors, else the sum over tumors.
        """
        if by_gene:
            opp = self.opportunity(labels, splice_label, observable)
            g = pd.Index(self.table.genes).get_indexer(
                opp.index.get_level_values("ensembl_gene_id")
            )
            ch = np.isin(
                opp.index.get_level_values("channel"),
                ["mis", "non", "spl"],
            ).astype(int)
            mult = self._mult[g, ch]
        else:
            opp = self.opportunity(
                labels,
                splice_label,
                observable,
                by_gene=False,
                gene_weights=np.nan_to_num(self._mult),
            )
            mult = np.ones(len(opp))
        W = opp.to_numpy()
        if per_tumor:
            return pd.DataFrame(
                (W @ self._M.T) * mult[:, None],
                index=opp.index,
                columns=self.tumors,
            )
        return pd.Series((W @ self._M_total) * mult, index=opp.index)

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def _element_rates(self, g_row, nonsyn, tau, weight, per_tumor):
        """Rates of elements given gene row, channel, type, weight."""
        mult = np.where(
            g_row >= 0,
            self._mult[np.maximum(g_row, 0), nonsyn.astype(int)],
            np.nan,
        )
        factor = weight * mult
        if per_tumor:
            return self._M[:, tau].T * factor[:, None]
        return self._M_total[tau] * factor

    def snv_rates(self, chrom, pos, alt, per_tumor=True):
        """Rates of genomic SNVs (plus-strand ``alt``), vectorised.

        A position coding in several genes gives one row per gene; a
        position that is none of the table's elements gives none.

        Returns
        -------
        (pandas.DataFrame, pandas.DataFrame or None)
            ``info``: query, ensembl_gene_id, channel, type, weight,
            observable, context, rate_total, observable_rate_total;
            and, with ``per_tumor``, rows x tumors mutation rates
            (multiply by ``info["observable"]`` for the observable
            rate).
        """
        t = self.table
        m = t.models
        chrom = np.asarray(chrom).astype(str)
        pos = np.asarray(pos, dtype=np.int64)
        alt_code = _CODE[
            np.array([ord(str(a).upper()[0]) for a in alt], np.uint8)
        ]
        qi, flat = t.genomic_positions(chrom, pos)
        rows = t.position_rows(flat)
        minus = (m.selection["strand"].to_numpy() == "-")[
            m.gene_of[flat]
        ]
        a = np.where(
            minus, _COMPLEMENT_CODE[alt_code[qi]], alt_code[qi]
        ).astype(np.int64)
        ref = m.codes[flat].astype(np.int64)
        g_row = t.gene_row[m.gene_of[flat]]
        ok = (a < 4) & (a != ref) & (g_row >= 0)
        if t.coding_in is not None:
            ok &= np.asarray(t.coding_in, dtype=bool)[flat]
        qi, flat, rows, a, ref, g_row = (
            x[ok] for x in (qi, flat, rows, a, ref, g_row)
        )
        col = (a - ref) % 4 - 1
        n = len(rows)
        tau = np.full(n, -1, dtype=np.int64)
        cons = np.full(n, 1, dtype=np.int64)
        weight = np.ones(n)
        observable = np.ones(n, dtype=bool)
        context = np.full(n, "coding", dtype=object)
        inside = rows >= 0
        r = rows[inside]
        tau[inside] = t.types[r, col[inside]]
        cons[inside] = t.cons[r, col[inside]]
        weight[inside] = t.weights(r)[np.arange(len(r)), col[inside]]
        observable[inside] = ~t.masked[r, col[inside]]
        # Positions with no element (a CDS end, next to an N): as
        # mu_ms, the call's own genomic type at weight 1.
        edge = ~inside
        if edge.any():
            fe = flat[edge]
            tau[edge] = self._genomic_types(fe, a[edge])
            cons[edge] = _edge_consequence(m, fe, a[edge])
            observable[edge] = ~self._masked_at(fe, a[edge])
            context[edge] = "edge"
        keep = tau >= 0
        qi, g_row, tau, cons, weight, observable, context = (
            x[keep]
            for x in (
                qi,
                g_row,
                tau,
                cons,
                weight,
                observable,
                context,
            )
        )
        info = pd.DataFrame(
            {
                "query": qi,
                "chrom": chrom[qi],
                "pos": pos[qi],
                "alt": np.asarray(alt, dtype=object)[qi],
                "ensembl_gene_id": t.genes[g_row],
                "channel": np.array(CHANNELS, dtype=object)[cons],
                "type": np.array(canonical_types_order, dtype=object)[
                    tau
                ],
                "weight": weight,
                "observable": observable,
                "context": context,
            }
        )
        rates = self._element_rates(
            g_row, _NONSYN[cons], tau, weight, per_tumor
        )
        spl_info, spl_rates = self._splice_snvs(
            chrom, pos, alt_code, per_tumor
        )
        if len(spl_info):
            info = pd.concat([info, spl_info], ignore_index=True)
            rates = np.concatenate([rates, spl_rates])
        total = rates.sum(axis=1) if per_tumor else rates
        info["rate_total"] = total
        info["observable_rate_total"] = total * info["observable"]
        if per_tumor:
            return info, pd.DataFrame(rates, columns=self.tumors)
        return info, None

    def _splice_sites_at(self, chrom, pos):
        """(query, splice row) of every in-universe splice site hit."""
        t = self.table
        m = t.models
        sp = m.splice
        if "splice_keys" not in t._genomic:
            usable = t.gene_row[sp["gene"].to_numpy()] >= 0
            if t.splice_in is not None:
                usable &= np.asarray(t.splice_in, dtype=bool)
            rows = np.flatnonzero(usable)
            chroms = m.selection["chrom"].to_numpy()[
                sp["gene"].to_numpy()[rows]
            ]
            keys = (
                _chrom_codes(chroms) * (1 << 31)
                + sp["gpos"].to_numpy()[rows]
            )
            order = np.argsort(keys, kind="stable")
            t._genomic["splice_keys"] = keys[order]
            t._genomic["splice_rows"] = rows[order]
        keys = t._genomic["splice_keys"]
        srows = t._genomic["splice_rows"]
        q = _chrom_codes(chrom) * (1 << 31) + pos
        lo = np.searchsorted(keys, q, side="left")
        hi = np.searchsorted(keys, q, side="right")
        qi = np.repeat(np.arange(len(q)), hi - lo)
        starts = np.repeat(lo, hi - lo)
        offs = np.arange(len(qi)) - np.repeat(
            np.cumsum(hi - lo) - (hi - lo), hi - lo
        )
        return qi, srows[starts + offs]

    def _splice_element(self, srow, coding_alt):
        """Types and weights of one splice change (coding-strand alt).

        Returns (list of (type index, context weight), context source,
        masked flag).
        """
        from .germline_mask import allele_keys

        m = self.table.models
        sp = m.splice
        g = int(sp["gene"].iat[srow])
        gpos = int(sp["gpos"].iat[srow])
        kind = sp["kind"].iat[srow]
        exon = int(sp["exon_base"].iat[srow])
        chrom = m.selection["chrom"].iat[g]
        minus = m.selection["strand"].iat[g] == "-"
        ref = _KIND_REF[kind]
        masked = False
        if self.table.masked_keys is not None:
            genomic_alt = 3 - coding_alt if minus else coding_alt
            key = allele_keys([chrom], [gpos], [genomic_alt])
            masked = bool(np.isin(key, self.table.masked_keys)[0])

        if self.genome_dir is not None:
            step = -1 if minus else 1
            b = _genome_bases(
                self.genome_dir,
                [chrom] * 3,
                [gpos - step, gpos, gpos + step],
            )
            if minus:
                b = np.where(b < 4, _COMPLEMENT_CODE[b], 4)
            left, gref, right = (int(x) for x in b)
            if max(left, gref, right) < 4 and coding_alt != gref:
                tau = int(_TYPE_TABLE[left, gref, right, coding_alt])
                return [(tau, 1.0)], "genome", masked
        if kind == "d1":
            options = [((exon, 2, 3), 1.0)]
        elif kind == "a1":
            options = [((0, 2, exon), 1.0)]
        elif kind == "d2":
            options = [
                ((2, 3, b), w)
                for b, w in enumerate(
                    _normalised(DONOR_PLUS3_COMPOSITION)
                )
            ]
        else:
            options = [
                ((b, 0, 2), w)
                for b, w in enumerate(
                    _normalised(ACCEPTOR_MINUS3_COMPOSITION)
                )
            ]
        if coding_alt == ref:
            return [], "composition", masked
        return (
            [
                (int(_TYPE_TABLE[lf, rf, rt, coding_alt]), w)
                for (lf, rf, rt), w in options
            ],
            "composition",
            masked,
        )

    def _splice_snvs(self, chrom, pos, alt_code, per_tumor):
        qi, srows = self._splice_sites_at(chrom, pos)
        m = self.table.models
        infos, rates = [], []
        for q, s in zip(qi, srows):
            g = int(m.splice["gene"].iat[s])
            minus = m.selection["strand"].iat[g] == "-"
            a = int(alt_code[q])
            if a >= 4:
                continue
            coding_alt = 3 - a if minus else a
            options, source, masked = self._splice_element(
                s, coding_alt
            )
            if not options:
                continue
            g_row = self.table.gene_row[g]
            r = sum(
                self._element_rates(
                    np.array([g_row]),
                    np.array([True]),
                    [tau],
                    w,
                    per_tumor,
                )[0]
                for tau, w in options
            )
            rates.append(r)
            infos.append(
                {
                    "query": q,
                    "chrom": chrom[q],
                    "pos": pos[q],
                    "alt": _BASES[a],
                    "ensembl_gene_id": self.table.genes[g_row],
                    "channel": "spl",
                    "type": ";".join(
                        canonical_types_order[tau]
                        for tau, _ in options
                    ),
                    "weight": 1.0,
                    "observable": not masked,
                    "context": source,
                }
            )
        if not infos:
            return pd.DataFrame(), np.zeros((0, len(self.tumors)))
        return pd.DataFrame(infos), np.array(rates)

    def _gene_model_index(self, gene):
        """Transcript-model gene index from an Ensembl id or a symbol."""
        m = self.table.models
        if gene in m.gene_index:
            return m.gene_index[gene]
        if self._symbol_index is None:
            names = m.selection["gene_name"].astype(str).to_numpy()
            self._symbol_index = {}
            for i, name in enumerate(names):
                self._symbol_index.setdefault(name, i)
        if gene in self._symbol_index:
            return self._symbol_index[gene]
        raise KeyError(f"No transcript model for gene {gene!r}.")

    def variant_routes(self, gene, change):
        """The routes of one change on the gene's chosen transcript.

        ``change`` is a label as :func:`channel_universe.classify_calls`
        writes it: ``p.G12D``, ``p.R213*``, ``p.T1493=``, ``p.M1I``, a
        splice change ``c.559+1G>T`` or, for a position with no
        complete codon, ``c.123A>G``.

        Returns
        -------
        dict
            ``channel``, ``types`` (route types in ``_routes`` order),
            ``flat`` and ``alt`` (coding routes: flat position and
            coding-strand alternate base; None for splice), ``weights``,
            ``observable`` (per route), ``splice_row``, ``context``.
        """
        m = self.table.models
        g = self._gene_model_index(gene)
        if self.table.gene_row[g] < 0:
            raise KeyError(
                f"{gene} is outside the model's gene universe."
            )
        lo = int(m.offsets[g])
        match = _P_LABEL.match(change)
        if match:
            aa_ref, number, aa_alt = match.groups()
            cs = lo + 3 * (int(number) - 1)
            if cs + 3 > lo + m.lengths[g]:
                raise ValueError(f"{gene} {change}: past the CDS.")
            codon = "".join(
                _BASES[c] if c < 4 else "N"
                for c in m.codes[cs : cs + 3]
            )
            if _CODONS.get(codon) != aa_ref:
                raise ValueError(
                    f"{gene} {change}: codon {number} is {codon} "
                    f"({_CODONS.get(codon)}), not {aa_ref}."
                )
            if aa_alt == "=":
                channel, target = "syn", None
            elif aa_alt == "*":
                channel, target = "non", "*"
            else:
                channel, target = "mis", aa_alt
            types, sites = _routes(
                m,
                cs,
                target,
                channel,
                self.table.coding_in,
                with_sites=True,
            )
            if not types:
                # Every route sits where the CDS gives no context window
                # (a CDS end, next to an N): no element, but the calls
                # there are in the universe, and mu_ms falls back to the
                # call's own type at weight 1. Same here, with the type
                # read from the genome.
                return self._edge_route_record(g, cs, target, channel)
            flat = np.array([s for s, _ in sites], dtype=np.int64)
            alt = np.array([a for _, a in sites], dtype=np.int64)
            return self._coding_route_record(
                channel, types, flat, alt
            )
        match = _C_CODING.match(change)
        if match:
            number, ref, alt = match.groups()
            n_pad = int(m.selection["n_pad"].iat[g])
            i = lo + n_pad + int(number) - 1
            if m.codes[i] != _BASES.index(ref):
                raise ValueError(
                    f"{gene} {change}: reference mismatch."
                )
            row = self.table.position_rows([i])[0]
            if row < 0:
                return self._edge_record(
                    "mis", [(i, _BASES.index(alt))]
                )
            k = (_BASES.index(alt) - _BASES.index(ref)) % 4
            tau = int(self.table.types[row, k - 1])
            return self._coding_route_record(
                "mis",
                [canonical_types_order[tau]],
                np.array([i]),
                np.array([_BASES.index(alt)]),
            )
        match = _C_SPLICE.match(change)
        if match:
            anchor, offset, ref, alt = match.groups()
            sp = m.splice
            hit = np.flatnonzero(
                (sp["gene"].to_numpy() == g)
                & (sp["anchor"].to_numpy() == int(anchor))
                & (sp["kind"].to_numpy() == _SPLICE_KIND[offset])
            )
            if not len(hit):
                return self._empty_routes("spl")
            s = int(hit[0])
            if self.table.splice_in is not None and not bool(
                np.asarray(self.table.splice_in)[s]
            ):
                return self._empty_routes("spl")
            options, source, masked = self._splice_element(
                s, _BASES.index(alt)
            )
            return {
                "channel": "spl",
                "types": [
                    canonical_types_order[t] for t, _ in options
                ],
                "flat": None,
                "alt": None,
                "weights": [w for _, w in options],
                "observable": [not masked] * len(options),
                "splice_row": s,
                "context": source,
            }
        raise ValueError(f"Cannot parse variant change {change!r}.")

    def _genomic_types(self, flat, coding_alt):
        """Type of each (position, coding alt) from the genome's context.

        -1 where there is no genome or its window is not pure ACGT.
        """
        m = self.table.models
        flat = np.asarray(flat, dtype=np.int64)
        coding_alt = np.asarray(coding_alt, dtype=np.int64)
        if self.genome_dir is None or not len(flat):
            return np.full(len(flat), -1, dtype=np.int64)
        minus = (m.selection["strand"].to_numpy() == "-")[
            m.gene_of[flat]
        ]
        gp = m.gpos[flat]
        chrom = m.chrom_of[flat]
        before = _genome_bases(self.genome_dir, chrom, gp - 1)
        at = _genome_bases(self.genome_dir, chrom, gp)
        after = _genome_bases(self.genome_dir, chrom, gp + 1)

        def coding(b):
            return np.where(
                b < 4, _COMPLEMENT_CODE[np.minimum(b, 4)], 4
            )

        left = np.where(minus, coding(after), before)
        ref = np.where(minus, coding(at), at)
        right = np.where(minus, coding(before), after)
        ok = (
            (left < 4)
            & (ref < 4)
            & (right < 4)
            & (ref == m.codes[flat])
            & (coding_alt != ref)
        )
        out = np.full(len(flat), -1, dtype=np.int64)
        out[ok] = _TYPE_TABLE[
            left[ok], ref[ok], right[ok], coding_alt[ok]
        ]
        return out

    def _masked_at(self, flat, coding_alt):
        """Whether each (position, coding alt) allele is masked."""
        from .germline_mask import allele_keys

        m = self.table.models
        flat = np.asarray(flat, dtype=np.int64)
        if self.table.masked_keys is None or not len(flat):
            return np.zeros(len(flat), dtype=bool)
        minus = (m.selection["strand"].to_numpy() == "-")[
            m.gene_of[flat]
        ]
        a = np.asarray(coding_alt, dtype=np.int64)
        genomic_alt = np.where(minus, 3 - a, a)
        keys = allele_keys(
            m.chrom_of[flat], m.gpos[flat], genomic_alt
        )
        return np.isin(keys, self.table.masked_keys)

    def _edge_record(self, channel, sites):
        """Routes at positions with no element: genome type, weight 1."""
        t = self.table
        sites = [
            (i, a)
            for i, a in sites
            if t.coding_in is None or bool(np.asarray(t.coding_in)[i])
        ]
        if not sites:
            return self._empty_routes(channel)
        flat = np.array([i for i, _ in sites], dtype=np.int64)
        alt = np.array([a for _, a in sites], dtype=np.int64)
        tau = self._genomic_types(flat, alt)
        seen = ~self._masked_at(flat, alt)
        keep = tau >= 0
        if not keep.any():
            rec = self._empty_routes(channel)
            rec["context"] = "edge (no genome)"
            return rec
        return {
            "channel": channel,
            "types": [canonical_types_order[x] for x in tau[keep]],
            "flat": flat[keep],
            "alt": alt[keep],
            "weights": [1.0] * int(keep.sum()),
            "observable": list(seen[keep]),
            "splice_row": None,
            "context": "edge",
        }

    def _edge_route_record(self, g, cs, target, channel):
        """The routes _routes skips, as :meth:`_edge_record` sites."""
        m = self.table.models
        codon = "".join(
            _BASES[c] if c < 4 else "N" for c in m.codes[cs : cs + 3]
        )
        residue = _CODONS.get(codon)
        sites = []
        for q in range(3):
            ref = int(m.codes[cs + q])
            if ref >= 4:
                continue
            for a in range(4):
                if a == ref:
                    continue
                new = _CODONS.get(
                    codon[:q] + _BASES[a] + codon[q + 1 :]
                )
                hit = (
                    new == residue
                    if channel == "syn"
                    else new == target
                )
                if hit:
                    sites.append((cs + q, a))
        return self._edge_record(channel, sites)

    def _empty_routes(self, channel):
        return {
            "channel": channel,
            "types": [],
            "flat": None,
            "alt": None,
            "weights": [],
            "observable": [],
            "splice_row": None,
            "context": None,
        }

    def _coding_route_record(self, channel, types, flat, alt):
        from .site_weights import weights_at

        t = self.table
        rows = t.position_rows(flat)
        if (rows < 0).any():
            raise AssertionError(
                "A route sits outside the site table; the routes and the "
                "table enumerate different sites."
            )
        ref = t.models.codes[flat].astype(np.int64)
        col = (alt - ref) % 4 - 1
        t_idx = np.array(
            [canonical_types_order.index(x) for x in types],
            dtype=np.int64,
        )
        weights = weights_at(
            t.dbin[rows],
            t.orient[rows],
            t_idx[:, None],
            t.site_weights,
        )[:, 0]
        return {
            "channel": channel,
            "types": list(types),
            "flat": flat,
            "alt": alt,
            "weights": [float(w) for w in weights],
            "observable": list(~t.masked[rows, col]),
            "splice_row": None,
            "context": "coding",
        }

    def variant_rates(self, variants, genes=None, per_tumor=True):
        """Rates of protein (or splice) changes, summed over routes.

        Parameters
        ----------
        variants : sequence of str
            ``"GENE p.X"`` labels, or bare changes (``"p.G12D"``) when
            ``genes`` is given.
        genes : sequence of str, optional
            Ensembl id (or symbol) per variant; default: the label's
            first word, looked up as an id and then as a GENCODE symbol.
        per_tumor : bool
            Return the variants x tumors rates too.

        Returns
        -------
        (pandas.DataFrame, pandas.DataFrame or None, pandas.DataFrame or None)
            ``info`` per variant (channel, n_routes, n_masked_routes,
            routes, route_weights, context, rate_total,
            observable_rate_total, or ``error``), then the per-tumor
            mutation rates and observable rates.
        """
        variants = list(variants)
        infos = []
        rates = np.zeros((len(variants), len(self.tumors)))
        obs = np.zeros_like(rates)
        for i, v in enumerate(variants):
            if genes is not None:
                gene, change = genes[i], v.split(" ")[-1]
            else:
                gene, _, change = v.rpartition(" ")
            try:
                rec = self.variant_routes(gene, change)
            except (KeyError, ValueError) as err:
                infos.append({"variant": v, "error": str(err)})
                rates[i] = obs[i] = np.nan
                continue
            g_row = self.table.gene_row[self._gene_model_index(gene)]
            nonsyn = rec["channel"] != "syn"
            for tau_name, w, seen in zip(
                rec["types"], rec["weights"], rec["observable"]
            ):
                tau = canonical_types_order.index(tau_name)
                r = self._element_rates(
                    np.array([g_row]),
                    np.array([nonsyn]),
                    [tau],
                    w,
                    True,
                )[0]
                rates[i] += r
                if seen:
                    obs[i] += r
            infos.append(
                {
                    "variant": v,
                    "ensembl_gene_id": (
                        self.table.genes[g_row]
                        if g_row >= 0
                        else None
                    ),
                    "channel": rec["channel"],
                    "n_routes": len(rec["types"]),
                    "n_masked_routes": int(
                        len(rec["observable"])
                        - sum(rec["observable"])
                    ),
                    "routes": ";".join(rec["types"]),
                    "route_weights": rec["weights"],
                    "context": rec["context"],
                    "error": None,
                }
            )
        info = pd.DataFrame(infos).set_index("variant")
        info["rate_total"] = rates.sum(axis=1)
        info["observable_rate_total"] = obs.sum(axis=1)
        if not per_tumor:
            return info, None, None
        return (
            info,
            pd.DataFrame(rates, index=variants, columns=self.tumors),
            pd.DataFrame(obs, index=variants, columns=self.tumors),
        )
