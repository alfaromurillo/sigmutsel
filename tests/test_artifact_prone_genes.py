"""The artifact-prone gene flag: a report column, never a fit input."""

from pathlib import Path

import pandas as pd

import sigmutsel
from sigmutsel.estimate_presence import (
    artifact_prone_ensembl_ids,
    artifact_prone_symbols,
    filter_passenger_genes,
    filter_passenger_genes_ensembl,
    non_passenger_ensembl_ids,
    non_passenger_symbols,
)
from sigmutsel.locations import location_artifact_prone_genes


def test_table_is_resolved_and_has_both_sources():
    table = pd.read_csv(location_artifact_prone_genes, sep="\t")
    assert table["ensembl_gene_id"].notna().all()
    assert table["ensembl_gene_id"].is_unique
    assert table["symbol"].is_unique
    flags = table["source"].str.contains("FLAGS_Shyr2014")
    bailey = table["source"].str.contains("Bailey2018_tableS7")
    assert flags.sum() == 100
    assert bailey.sum() == 488
    assert (flags | bailey).all()
    symbols = artifact_prone_symbols()
    # FLAGS only, Table S7 only, and both.
    assert {"MUC5B", "ABCA1", "TTN", "MUC16"} <= symbols
    # A renamed gene is resolved to its current symbol.
    assert "ADGRV1" in symbols and "GPR98" not in symbols
    assert "ENSG00000155657" in artifact_prone_ensembl_ids()  # TTN


def test_passenger_filters_ignore_the_flag():
    """A flagged gene that is not a known cancer gene stays a
    passenger in both filters, and the non-passenger sets do not
    contain the flagged genes that are not otherwise excluded."""
    flagged_ids = artifact_prone_ensembl_ids()
    not_excluded = flagged_ids - non_passenger_ensembl_ids()
    assert len(not_excluded) > 400
    kept = filter_passenger_genes_ensembl(
        pd.Index(sorted(flagged_ids))
    )
    assert set(kept) == not_excluded

    flagged = artifact_prone_symbols()
    passengers = flagged - non_passenger_symbols()
    assert {"TTN", "MUC16", "ABCA1"} & passengers == {"TTN", "ABCA1"}
    db = pd.DataFrame(
        {
            "gene": sorted(flagged),
            "variant": [f"{gene} x" for gene in sorted(flagged)],
        }
    )
    assert set(filter_passenger_genes(db)) == passengers


def test_no_module_reads_the_flag_but_its_loader():
    """The flag must never reach a fit, a passenger set or a rate:
    only the loader (estimate_presence) and the path registry
    (locations) may mention it."""
    package = Path(sigmutsel.__file__).parent
    allowed = {"estimate_presence.py", "locations.py"}
    readers = sorted(
        path.name
        for path in package.rglob("*.py")
        if "artifact_prone" in path.read_text()
        and path.name not in allowed
    )
    assert readers == []
