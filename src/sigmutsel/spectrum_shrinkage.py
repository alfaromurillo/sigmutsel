"""Spectrum shrinkage: each tumor's per-type spectrum, shrunk twice.

The baseline rate of type ``t`` in tumor ``j`` is ``mu_bar_jt =
ell_hat_j p_jt``, with ``p_jt`` the tumor's spectrum. Without
shrinkage, ``p_jt = sum_sigma alpha_j,sigma s^sigma_t``, the
reconstruction from the tumor's fitted signature exposures. That
estimate fails at both ends of the burden range:

- with few mutations the fit can land on signatures that give some
  type probability 0 (SBS84 alone emits no T>A), and a mutation the
  tumor carries then has rate 0;
- with thousands it misfits: a mixture of reference signatures
  predicts a hypermutated tumor's unseen mutations worse than its own
  observed spectrum does.

Two pseudo-count steps fix both. The **floor** shrinks the signature
spectrum toward a cohort prior ``e``,

    q_jt = (n_j p_jt + kappa_1 e_t) / (n_j + kappa_1),
    e_t  = (1 - share) pool_t + share s^floor_t,

with ``n_j`` the tumor's fitted mutation count (its assignment
total), ``pool`` the cohort's pooled passenger spectrum and
``s^floor`` a signature with no zero type (SBS5) at a small ``share``
(0.1), so every type keeps a positive rate. The **hybrid** then
shrinks the tumor's own passenger counts toward ``q``,

    pi_jt = (n_jt + kappa_2 q_jt) / (n_j. + kappa_2),

with ``n_jt`` the tumor's in-universe *passenger* calls of type ``t``
and ``n_j.`` their total: signatures lead at low burden and the data
at high. Only passenger calls count, in the pool and in ``n_jt``: a
recurrent driver's calls (thyroid BRAF V600E is 6% of its cohort's
calls, all one type) would otherwise raise its own rate and let
selection into the rate model. ``kappa_2 = None`` is the floor alone;
``kappa_2 -> inf`` gives ``pi = q``, ``kappa_1 -> inf`` gives ``q =
e`` for every tumor. The burden ``ell_hat_j`` is untouched, since
``pi_j`` sums to 1.

``kappa_1`` and ``kappa_2`` are per-cohort parameters, chosen once on
held-out mutations (:func:`heldout_spectrum_scores`,
:func:`select_kappas`) and stored with the dataset
(:meth:`models.MutationDataset.set_spectrum_shrinkage`), as the site
weights are. The choice is reported with a nested cross-validation
over tumors, which measures how much choosing two numbers per cohort
overfits.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import canonical_types_order

DEFAULT_KAPPA_1_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0)
DEFAULT_KAPPA_2_GRID = (
    10.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
)


@dataclass
class SpectrumShrinkage:
    """Per-cohort parameters of the floor and the hybrid.

    ``kappa_2`` None applies the floor alone. ``selection`` records how
    the kappas were chosen (:func:`select_kappas`), or None if they
    were set by hand.
    """

    kappa_1: float
    kappa_2: float | None = None
    floor_signature: str = "SBS5"
    floor_share: float = 0.1
    selection: dict | None = field(default=None, compare=False)

    def __post_init__(self):
        if not self.kappa_1 > 0:
            raise ValueError("kappa_1 must be positive")
        if self.kappa_2 is not None and not self.kappa_2 > 0:
            raise ValueError("kappa_2 must be positive or None")
        if not 0 < self.floor_share <= 1:
            raise ValueError(
                "floor_share must be in (0, 1]: with no share of the "
                "floor signature the prior can have zero types"
            )

    def to_dict(self):
        return {
            "kappa_1": float(self.kappa_1),
            "kappa_2": (
                None if self.kappa_2 is None else float(self.kappa_2)
            ),
            "floor_signature": self.floor_signature,
            "floor_share": float(self.floor_share),
            "selection": self.selection,
        }

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        if isinstance(d, cls):
            return d
        return cls(
            kappa_1=d["kappa_1"],
            kappa_2=d.get("kappa_2"),
            floor_signature=d.get("floor_signature", "SBS5"),
            floor_share=d.get("floor_share", 0.1),
            selection=d.get("selection"),
        )

    def label(self):
        k2 = (
            "floor only"
            if self.kappa_2 is None
            else f"{self.kappa_2:g}"
        )
        return f"kappa_1 {self.kappa_1:g}, kappa_2 {k2}"


def passenger_calls(db):
    """In-universe calls outside every non-passenger gene.

    :func:`channel_universe.model_calls` minus the genes of
    :func:`estimate_presence.non_passenger_ensembl_ids` (matched on
    Ensembl ids when ``db`` has ``ensembl_gene_id``, otherwise on the
    ``gene`` symbol).
    """
    from .channel_universe import model_calls
    from .estimate_presence import (
        non_passenger_ensembl_ids,
        non_passenger_symbols,
    )

    calls = model_calls(db)
    if "ensembl_gene_id" in calls.columns:
        ids = calls["ensembl_gene_id"].astype(str)
        ids = ids.str.replace(r"\.\d+$", "", regex=True)
        return calls[
            ~ids.isin(non_passenger_ensembl_ids()).to_numpy()
        ]
    return calls[~calls["gene"].isin(non_passenger_symbols())]


def type_counts(calls, tumors, types):
    """Tumors x types call counts (zeros where a tumor has none)."""
    counts = pd.crosstab(calls["Tumor_Sample_Barcode"], calls["type"])
    return counts.reindex(
        index=pd.Index(tumors), columns=pd.Index(types), fill_value=0
    ).astype(float)


def floor_prior(pool_counts, floor_spectrum, floor_share):
    """``e = (1 - share) pool + share s^floor`` over the types.

    ``pool_counts`` (types) are normalized; with no calls at all the
    prior is the floor signature alone.
    """
    pool = np.asarray(pool_counts, dtype=float)
    floor = np.asarray(floor_spectrum, dtype=float)
    if (floor <= 0).any():
        raise ValueError(
            "the floor signature has types with probability 0; it "
            "cannot guarantee every type a positive rate"
        )
    floor = floor / floor.sum()
    if pool.sum() <= 0:
        return floor
    return (1 - floor_share) * pool / pool.sum() + floor_share * floor


def floored_spectra(p, n, prior, kappa_1):
    """``q = (n p + kappa_1 e) / (n + kappa_1)``, row by row.

    ``p``: tumors x types signature spectra (rows sum to 1, or to 0
    for a tumor with no fit); ``n``: the tumors' fitted mutation
    counts; ``prior``: ``e`` over the types.
    """
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)[:, None]
    e = np.asarray(prior, dtype=float)[None, :]
    return (n * p + kappa_1 * e) / (n + kappa_1)


def hybrid_spectra(counts, q, kappa_2):
    """``pi = (n_jt + kappa_2 q) / (n_j. + kappa_2)``, row by row.

    ``counts``: tumors x types passenger calls; ``q``: the floored
    spectra. ``kappa_2`` None returns ``q`` (the floor alone).
    """
    q = np.asarray(q, dtype=float)
    if kappa_2 is None:
        return q
    c = np.asarray(counts, dtype=float)
    return (c + kappa_2 * q) / (c.sum(axis=1)[:, None] + kappa_2)


def shrink_spectra(
    alphas, sig_matrix, n, counts, floor_spectrum, params
):
    """Shrunk spectra (tumors x types) for one cohort.

    ``alphas``: tumors x signatures exposures (rows sum to 1);
    ``sig_matrix``: types x signatures; ``n``: fitted mutation counts
    (tumors); ``counts``: tumors x types passenger calls of the same
    tumors and types; ``floor_spectrum``: the floor signature over the
    types; ``params``: :class:`SpectrumShrinkage`.
    """
    params = SpectrumShrinkage.from_dict(params)
    types = sig_matrix.index
    counts = counts.reindex(
        index=alphas.index, columns=types, fill_value=0
    )
    p = alphas.to_numpy(float) @ sig_matrix.to_numpy(float).T
    prior = floor_prior(
        counts.sum(axis=0).to_numpy(),
        np.asarray(floor_spectrum, dtype=float),
        params.floor_share,
    )
    q = floored_spectra(
        p, np.asarray(n, dtype=float), prior, params.kappa_1
    )
    pi = hybrid_spectra(counts.to_numpy(), q, params.kappa_2)
    return pd.DataFrame(pi, index=alphas.index, columns=types)


# ---------------------------------------------------------------------
# Choosing the kappas on held-out mutations
# ---------------------------------------------------------------------


def _sigprofiler_fit(matrix, signatures, workdir):
    """SigProfilerAssignment activities (tumors x signatures)."""
    from pathlib import Path

    from .signature_decomposition import run_signature_decomposition

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mpath, spath = workdir / "matrix.txt", workdir / "signatures.txt"
    matrix.to_csv(mpath, sep="\t")
    signatures.to_csv(spath, sep="\t")
    run_signature_decomposition(
        str(mpath),
        str(workdir / "out"),
        signature_database=str(spath),
        input_type="matrix",
        exome=True,
        cosmic_version=3.6,
        genome_build="GRCh38",
        export_probabilities=False,
        export_probabilities_per_mutation=False,
    )
    acts = pd.read_csv(
        workdir
        / "out"
        / "Assignment_Solution"
        / "Activities"
        / "Assignment_Solution_Activities.txt",
        sep="\t",
        index_col=0,
    )
    return acts.reindex(index=matrix.columns, fill_value=0).reindex(
        columns=signatures.columns, fill_value=0
    )


def arm_label(kappa_1, kappa_2):
    """Column name of one (kappa_1, kappa_2) arm in the score table."""
    k2 = "floor" if kappa_2 is None else f"{kappa_2:g}"
    return f"{kappa_1:g}|{k2}"


def parse_arm_label(label):
    """Inverse of :func:`arm_label`."""
    k1, k2 = label.split("|")
    return float(k1), (None if k2 == "floor" else float(k2))


def heldout_spectrum_scores(
    db,
    sig_matrix,
    kappa_1_grid=DEFAULT_KAPPA_1_GRID,
    kappa_2_grid=DEFAULT_KAPPA_2_GRID,
    floor_signature="SBS5",
    floor_share=0.1,
    folds=5,
    seed=0,
    fit=None,
    workdir=None,
):
    """Each tumor's held-out log-probability under each pair of kappas.

    Each tumor's in-universe calls are split at random into ``folds``
    parts. For each part, signatures are fitted on the other parts
    (``fit(matrix, signatures, workdir)``, by default
    SigProfilerAssignment against ``sig_matrix``, every call as in
    production) and the part's passenger calls are scored under the
    shrunk spectrum, with the pool and the hybrid's own counts taken
    from the other parts' passenger calls. Arms: every ``(kappa_1,
    kappa_2)`` of the grids plus the floor alone (``kappa_2`` None)
    at every ``kappa_1``, and ``"none"``, the signature spectrum as
    is.

    Returns
    -------
    pandas.DataFrame
        One row per tumor: ``burden`` (its in-universe calls), the
        log-probability under ``"none"`` over the held-out calls it can
        score, ``none_impossible`` (held-out calls it gives
        probability 0), and one column per arm (:func:`arm_label`).
        Every shrunk arm scores every call, since the prior has no
        zero type; compare an arm with ``"none"`` only on tumors with
        ``none_impossible == 0``, or on the arm-to-arm differences.
    """
    import tempfile
    from pathlib import Path

    fit = fit or _sigprofiler_fit
    types = sig_matrix.index
    if floor_signature not in sig_matrix.columns:
        raise ValueError(
            f"floor signature {floor_signature!r} is not in the "
            "signature matrix"
        )
    floor = sig_matrix[floor_signature].to_numpy(float)

    from .channel_universe import model_calls

    calls = model_calls(db)
    calls = calls[calls["type"].isin(types)].reset_index(drop=True)
    pcalls = passenger_calls(calls)
    passenger = calls.index.isin(pcalls.index)
    tumors = sorted(calls["Tumor_Sample_Barcode"].unique())
    rng = np.random.default_rng(seed)
    part = rng.integers(0, folds, len(calls))

    def matrix(mask):
        return type_counts(calls[mask], tumors, types)

    arms = [(k1, None) for k1 in kappa_1_grid] + [
        (k1, k2) for k1 in kappa_1_grid for k2 in kappa_2_grid
    ]
    score = {arm: np.zeros(len(tumors)) for arm in arms}
    none = np.zeros(len(tumors))
    impossible = np.zeros(len(tumors))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(workdir) if workdir is not None else Path(tmp)
        for h in range(folds):
            train = matrix(part != h)
            ptrain = matrix((part != h) & passenger)
            test = matrix((part == h) & passenger).to_numpy()
            mtx = train.T
            if set(mtx.index) == set(canonical_types_order):
                mtx = mtx.reindex(canonical_types_order)
            mtx.index.name = "MutationType"
            acts = fit(mtx, sig_matrix, root / f"fold{h}")
            acts = acts.reindex(index=tumors, fill_value=0)
            n = acts.sum(axis=1).to_numpy(float)
            alpha = acts.div(
                pd.Series(n, index=acts.index).replace(0, np.nan),
                axis=0,
            ).fillna(0.0)
            p = (
                alpha.to_numpy(float)
                @ sig_matrix[acts.columns].to_numpy(float).T
            )
            with np.errstate(divide="ignore"):
                lp = np.where(test > 0, test * np.log(p), 0.0)
            ok = np.isfinite(lp)
            none += np.where(ok, lp, 0.0).sum(axis=1)
            impossible += np.where(ok, 0, test).sum(axis=1)
            c = ptrain.to_numpy()
            prior = floor_prior(c.sum(axis=0), floor, floor_share)
            for k1 in kappa_1_grid:
                q = floored_spectra(p, n, prior, k1)
                for arm in [a for a in arms if a[0] == k1]:
                    pi = hybrid_spectra(c, q, arm[1])
                    score[arm] += np.where(
                        test > 0, test * np.log(pi), 0.0
                    ).sum(axis=1)
    burden = calls.groupby("Tumor_Sample_Barcode").size()
    out = pd.DataFrame(
        {
            "burden": burden.reindex(tumors).to_numpy(),
            "none": none,
            "none_impossible": impossible,
        },
        index=pd.Index(tumors, name="Tumor_Sample_Barcode"),
    )
    for arm in arms:
        out[arm_label(*arm)] = score[arm]
    out.attrs.update(
        floor_signature=floor_signature,
        floor_share=floor_share,
        folds=folds,
        seed=seed,
    )
    return out


def select_kappas(
    scores,
    family="hybrid",
    outer_folds=5,
    seed=0,
    floor_signature="SBS5",
    floor_share=0.1,
):
    """Choose the kappas, and measure the choice's optimism.

    ``scores``: :func:`heldout_spectrum_scores`'s table. ``family``:
    ``"hybrid"`` (both kappas) or ``"floor"`` (``kappa_1`` alone).
    The chosen pair maximizes the summed held-out log-probability over
    every tumor. The nested estimate splits the tumors into
    ``outer_folds`` groups, chooses on all groups but one and scores
    the one left out; ``optimism`` is the in-sample gain minus the
    nested gain. Gains are against the floor at the smallest
    ``kappa_1`` of the grid (an arm that scores every call), and
    against ``"none"`` on the tumors it can score.

    Returns
    -------
    SpectrumShrinkage
        With ``selection`` recording the grid, the nested and
        in-sample gains, the optimism and each outer fold's choice.
    """
    if family not in ("hybrid", "floor"):
        raise ValueError("family must be 'hybrid' or 'floor'")
    labels = [
        c
        for c in scores.columns
        if "|" in c
        and (parse_arm_label(c)[1] is None) == (family == "floor")
    ]
    if not labels:
        raise ValueError(f"no {family} arms in the score table")
    floors = [c for c in scores.columns if c.endswith("|floor")]
    ref = min(floors, key=lambda c: parse_arm_label(c)[0])
    gains = scores[labels].sub(scores[ref], axis=0)
    total = gains.sum()
    best = total.idxmax()

    rng = np.random.default_rng(seed)
    group = rng.permutation(len(scores)) % outer_folds
    nested, choices = 0.0, []
    for g in range(outer_folds):
        inner = gains[group != g].sum().idxmax()
        nested += float(gains.loc[group == g, inner].sum())
        choices.append(inner)

    scorable = scores["none_impossible"] == 0
    vs_none = float(
        (
            scores.loc[scorable, best] - scores.loc[scorable, "none"]
        ).sum()
    )
    k1, k2 = parse_arm_label(best)
    selection = {
        "family": family,
        "reference_arm": ref,
        "kappa_1_grid": sorted(
            {parse_arm_label(c)[0] for c in labels}
        ),
        "kappa_2_grid": sorted(
            {
                parse_arm_label(c)[1]
                for c in labels
                if parse_arm_label(c)[1] is not None
            }
        ),
        "n_tumors": len(scores),
        "heldout_folds": int(scores.attrs.get("folds", 0)),
        "outer_folds": int(outer_folds),
        "seed": int(seed),
        "gain_in_sample": float(total[best]),
        "gain_nested": nested,
        "optimism": float(total[best]) - nested,
        "gain_vs_none_scorable": vs_none,
        "n_tumors_scorable_by_none": int(scorable.sum()),
        "outer_fold_choices": choices,
        "at_grid_edge": _at_edge(best, labels),
    }
    return SpectrumShrinkage(
        kappa_1=k1,
        kappa_2=k2,
        floor_signature=floor_signature,
        floor_share=floor_share,
        selection=selection,
    )


def _at_edge(best, labels):
    """Which kappa of the chosen arm sits at its grid's end, if any."""
    k1, k2 = parse_arm_label(best)
    arms = [parse_arm_label(c) for c in labels]
    k1s = sorted({a[0] for a in arms})
    k2s = sorted({a[1] for a in arms if a[1] is not None})
    edge = []
    if k1 in (k1s[0], k1s[-1]) and len(k1s) > 1:
        edge.append("kappa_1")
    if k2 is not None and k2 in (k2s[0], k2s[-1]) and len(k2s) > 1:
        edge.append("kappa_2")
    return edge
