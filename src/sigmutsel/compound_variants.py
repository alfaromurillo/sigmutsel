"""Compound variants: several sites, one selection coefficient.

Some biology is not one substitution. A gene's whole hotspot codon,
the several ways to truncate a tumor suppressor, an in-frame region
whose disruptions are interchangeable -- each is a set of distinct
variants that plausibly share a selection coefficient, and each is
individually too rare to fit one.

Estimating them jointly is arithmetic, not a new model: the rates of
mutually exclusive events add, so a tumor's rate of acquiring *any*
member is the sum of its members' rates, and it carries the compound
variant if it carries any member. One gamma is then fitted against
that pair, exactly as for a single variant
(:meth:`.Model.estimate_gamma_compound`).

This module is the other half: deciding which variants group
together. :func:`define_compound_variants` splits a variant table by
whatever annotation columns you name and then merges, within each
group, variants that sit within `merge_distance` of each other --
the same two-stage design cancereffectsizeR uses, where `by` carries
the biology and `merge_distance` the genomic proximity.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def define_compound_variants(
    variant_db,
    variants=None,
    by=None,
    merge_distance=0,
    prefix="compound",
):
    """Group variants into compound variants.

    Parameters
    ----------
    variant_db : pd.DataFrame
        Variant table indexed by variant key, as
        ``MutationDataset.variant_db``. Needs ``Chromosome`` and
        ``Start_Position`` unless `merge_distance` is infinite, plus
        whatever columns `by` names.
    variants : sequence, optional
        Restrict to these variants. Default: all of `variant_db`.
    by : str or sequence of str, optional
        Column(s) to split on before merging by distance -- e.g.
        ``"gene"``, or an annotation column of your own. Rows with a
        missing value in a `by` column are kept, in a group of their
        own named for the column, rather than dropped.
    merge_distance : int or float, default 0
        Maximum distance in base pairs between a variant and the
        nearest variant already in a compound for it to join that
        compound. Merging is transitive: a chain of variants each
        within `merge_distance` of the next becomes one compound,
        however far apart its ends are. ``0`` merges only variants
        at the same position, and ``np.inf`` merges every variant in
        a `by` group, across chromosomes.
    prefix : str, default "compound"
        Name stem used when `by` is not given.

    Returns
    -------
    dict
        Name -> list of variant keys, the list ordered by position;
        variants with no ``Start_Position`` (a multi-site splice
        annotation, say) cannot be placed and come last, each as a
        compound of its own, with a warning naming the count.
        Names are ``"<group>.<n>"``, as in cancereffectsizeR.
        Single-variant groups are included: a compound of one is a
        plain variant, and dropping it silently would make the
        result depend on `merge_distance` in a way the caller did
        not ask for.

    Examples
    --------
    Every hotspot codon of a gene, as its own compound::

        define_compound_variants(
            dataset.variant_db, variants=hotspots, by="gene",
            merge_distance=2,
        )

    Truncating variants anywhere in a gene, pooled::

        define_compound_variants(
            dataset.variant_db, variants=truncating, by="gene",
            merge_distance=np.inf,
        )
    """
    table = variant_db
    if variants is not None:
        variants = list(variants)
        missing = [v for v in variants if v not in table.index]
        if missing:
            raise ValueError(
                f"{len(missing)} variant(s) are not in variant_db, "
                f"e.g. {missing[:3]}."
            )
        table = table.loc[variants]

    if table.empty:
        return {}

    if merge_distance != np.inf and merge_distance < 0:
        raise ValueError(
            "merge_distance must be non-negative or np.inf; got "
            f"{merge_distance}."
        )

    group_names = _group_names(table, by, prefix)

    if merge_distance == np.inf:
        chunks = [
            (name, list(index))
            for name, index in table.groupby(
                group_names, sort=True
            ).groups.items()
        ]
    else:
        chunks = _merge_by_distance(
            table, group_names, merge_distance
        )

    compounds = {}
    counters = {}
    for name, members in chunks:
        counters[name] = counters.get(name, 0) + 1
        compounds[f"{name}.{counters[name]}"] = members
    return compounds


def _group_names(table, by, prefix):
    """One group label per row: the `by` values joined, or `prefix`."""
    if by is None:
        return pd.Series(prefix, index=table.index)

    by = [by] if isinstance(by, str) else list(by)
    missing = [column for column in by if column not in table]
    if missing:
        raise ValueError(
            f"by names column(s) not in variant_db: {missing}."
        )

    # A missing annotation is a group, not a reason to drop the
    # variant -- silently losing rows here would silently shrink
    # somebody's gamma denominator later.
    labels = [
        table[column]
        .astype(object)
        .where(table[column].notna(), f"{column}.NA")
        .astype(str)
        for column in by
    ]
    names = labels[0]
    for label in labels[1:]:
        names = names + "." + label
    return names


def _merge_by_distance(table, group_names, merge_distance):
    """(group name, members) chunks, merged by genomic proximity.

    One sort and one pass: a chunk breaks wherever the group label
    changes, the chromosome changes, or the gap to the previous
    variant exceeds `merge_distance`. Merging is therefore
    transitive by construction, since each variant is compared only
    with its nearest neighbour to the left.
    """
    required = {"Chromosome", "Start_Position"}
    if not required.issubset(table.columns):
        raise ValueError(
            "A finite merge_distance needs Chromosome and "
            "Start_Position columns in variant_db; got "
            f"{sorted(table.columns)}."
        )

    # A variant with no single position -- a multi-site splice
    # annotation, say -- cannot be placed on the number line, so it
    # cannot be merged by distance. It becomes its own compound
    # rather than blocking the call or being dropped: both of those
    # would decide something the caller did not ask about.
    placeless = table.index[table["Start_Position"].isna()]
    if len(placeless):
        logger.warning(
            "%d variant(s) have no Start_Position and were left as "
            "compounds of their own, e.g. %s.",
            len(placeless),
            ", ".join(map(str, placeless[:3])),
        )
        table = table.drop(index=placeless)

    ordered = (
        pd.DataFrame(
            {
                "group": group_names.loc[table.index],
                "chromosome": table["Chromosome"],
                "position": table["Start_Position"].astype(float),
            }
        )
        .sort_values(["group", "chromosome", "position"])
        .reset_index()
    )

    chunks = []
    if len(ordered):
        group = ordered["group"].to_numpy()
        chromosome = ordered["chromosome"].to_numpy()
        position = ordered["position"].to_numpy()
        variant = ordered.iloc[:, 0].to_numpy()

        breaks = np.ones(len(ordered), dtype=bool)
        breaks[1:] = (
            (group[1:] != group[:-1])
            | (chromosome[1:] != chromosome[:-1])
            | ((position[1:] - position[:-1]) > merge_distance)
        )
        edges = np.flatnonzero(breaks)
        for start, end in zip(
            edges, list(edges[1:]) + [len(ordered)]
        ):
            chunks.append((group[start], list(variant[start:end])))

    chunks.extend(
        (group_names.loc[name], [name]) for name in placeless
    )
    return chunks


def compound_rates(mu_ms, members):
    """Per-tumor rate of acquiring any member of a compound.

    The members are distinct substitutions, so they are mutually
    exclusive in one tumor and their Poisson rates add.
    """
    return mu_ms.loc[list(members)].sum(axis=0)


def compound_presence(variants_present, members):
    """Whether each tumor carries any member of a compound."""
    return (
        variants_present.loc[list(members)].max(axis=0) == 1
    ).astype(bool)


def summarize_compounds(compounds, variant_db=None):
    """One row per compound: size, and the genes it spans.

    A compound that spans more than one gene is legal and sometimes
    intended, but it is also what an over-large `merge_distance`
    produces by accident, so it is worth being able to see.
    """
    rows = []
    for name, members in compounds.items():
        row = {"compound": name, "n_variants": len(members)}
        if variant_db is not None and "gene" in variant_db.columns:
            genes = sorted(
                set(variant_db.loc[list(members), "gene"].dropna())
            )
            row["genes"] = ", ".join(genes)
            row["n_genes"] = len(genes)
        rows.append(row)

    return pd.DataFrame(rows).set_index("compound")
