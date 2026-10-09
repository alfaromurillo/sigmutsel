"""Known cancer genes: the Cancer Gene Census and IntOGen, unioned."""

import pandas as pd

from sigmutsel.estimate_presence import (
    cancer_gene_ensembl_ids,
    cancer_gene_symbols,
    filter_passenger_genes,
    filter_passenger_genes_ensembl,
)
from sigmutsel.locations import location_intogen_driver_genes


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
