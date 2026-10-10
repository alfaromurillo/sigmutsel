"""Not passengers: the Cancer Gene Census, IntOGen and AID targets."""

import pandas as pd

from sigmutsel.estimate_presence import (
    aid_target_ensembl_ids,
    aid_target_symbols,
    cancer_gene_ensembl_ids,
    cancer_gene_symbols,
    filter_passenger_genes,
    filter_passenger_genes_ensembl,
    non_passenger_ensembl_ids,
    non_passenger_symbols,
)
from sigmutsel.locations import (
    location_aid_target_genes,
    location_intogen_driver_genes,
)


def test_intogen_genes_are_cancer_genes():
    intogen = pd.read_csv(location_intogen_driver_genes, sep="\t")
    assert intogen["ensembl_gene_id"].notna().all()
    symbols = cancer_gene_symbols()
    # In both lists, and in IntOGen only.
    assert {"TP53", "KRAS", "TOP2A", "CD58"} <= symbols
    ids = cancer_gene_ensembl_ids()
    top2a = intogen.set_index("symbol").at["TOP2A", "ensembl_gene_id"]
    assert top2a in ids
    assert "ENSG00000141510" in ids  # TP53, from the Census


def test_passenger_filters_drop_intogen_only_drivers():
    intogen = pd.read_csv(location_intogen_driver_genes, sep="\t")
    top2a = intogen.set_index("symbol").at["TOP2A", "ensembl_gene_id"]
    ttn = "ENSG00000155657"  # TTN: in neither list
    kept = filter_passenger_genes_ensembl(pd.Index([top2a, ttn]))
    assert list(kept) == [ttn]
    db = pd.DataFrame(
        {"gene": ["TOP2A", "TTN"], "variant": ["TOP2A x", "TTN y"]}
    )
    assert list(filter_passenger_genes(db)) == ["TTN"]


def test_aid_targets_are_not_passengers():
    table = pd.read_csv(location_aid_target_genes, sep="\t")
    assert table["ensembl_gene_id"].notna().all()
    assert table["ensembl_gene_id"].is_unique
    aid = aid_target_symbols()
    # Immunoglobulin loci, IGLL5, and 2012 targets under current names.
    assert {"IGHG1", "IGHM", "IGKC", "IGLC2", "IGLL5"} <= aid
    assert {"ZFP36L1", "IRF8", "H2AC6", "CLEC2D"} <= aid
    # Cancer genes stay cancer genes; AID targets are added beside them.
    assert "IGHG1" not in cancer_gene_symbols()
    assert non_passenger_symbols() == cancer_gene_symbols() | aid
    ighg1 = table.set_index("symbol").at["IGHG1", "ensembl_gene_id"]
    assert ighg1 in aid_target_ensembl_ids()
    assert ighg1 in non_passenger_ensembl_ids()
    ttn = "ENSG00000155657"
    kept = filter_passenger_genes_ensembl(pd.Index([ighg1, ttn]))
    assert list(kept) == [ttn]
    db = pd.DataFrame(
        {"gene": ["IRF8", "TTN"], "variant": ["IRF8 x", "TTN y"]}
    )
    assert list(filter_passenger_genes(db)) == ["TTN"]
