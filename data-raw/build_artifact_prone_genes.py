"""Build artifact_prone_genes.tsv from two published gene lists.

Genes whose mutation calls are often artifacts or uninformative
about selection, for flagging in reports only:

- FLAGS, the 100 genes most frequently mutated in public exomes
  (Shyr et al. 2014, BMC Med Genomics 7:64, Additional file 4, the
  "FLAGS" column):
  https://static-content.springer.com/esm/art%3A10.1186%2Fs12920-014-0064-y/MediaObjects/12920_2014_64_MOESM4_ESM.txt
- the compiled false-positive genes of the TCGA PanCancer driver
  consensus (Bailey et al. 2018, Cell 173:371, Table S7, "Compiled
  False Positive Genes"; the sheet "Table S7" of mmc1):
  https://ars.els-cdn.com/content/image/1-s2.0-S009286741830237X-mmc1.xlsx

Ensembl ids are resolved through HGNC as in
``published_driver_genes.tsv``: the current symbol, else a unique
previous symbol, else a unique alias symbol.

Usage::

    python data-raw/build_artifact_prone_genes.py \\
        MOESM4_ESM.txt mmc1.xlsx hgnc_complete_set.txt

Output: src/sigmutsel/data/artifact_prone_genes.tsv
"""

import sys
from pathlib import Path

import pandas as pd

OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "sigmutsel"
    / "data"
    / "artifact_prone_genes.tsv"
)


def read_flags(path):
    """The 100 FLAGS genes (the file uses CR line endings)."""
    text = Path(path).read_bytes().decode("ascii").replace("\r", "\n")
    lines = [line.split("\t") for line in text.splitlines() if line]
    header = lines[0]
    col = header.index("FLAGS")
    genes = [
        row[col].strip()
        for row in lines[1:]
        if len(row) > col and row[col].strip()
    ]
    if len(genes) != 100:
        raise ValueError(
            f"FLAGS should have 100 genes, got {len(genes)}"
        )
    return genes


def read_bailey_s7(path):
    """Table S7's "Compiled False Positive Genes" column (488)."""
    sheet = pd.read_excel(path, sheet_name="Table S7", header=None)
    header_row = next(
        i
        for i, value in enumerate(sheet[2])
        if str(value).strip() == "Compiled False Positive Genes"
    )
    genes = [
        str(value).strip()
        for value in sheet[2].iloc[header_row + 1 :]
        if pd.notna(value) and str(value).strip()
    ]
    if len(genes) != 488:
        raise ValueError(
            f"Table S7 should list 488 genes, got {len(genes)}"
        )
    return genes


def hgnc_resolver(path):
    hgnc = pd.read_csv(path, sep="\t", dtype=str)
    hgnc = hgnc[hgnc["status"] == "Approved"]
    current = dict(zip(hgnc["symbol"], hgnc["ensembl_gene_id"]))
    by_field = {}
    for field in ("prev_symbol", "alias_symbol"):
        pairs = (
            hgnc[["symbol", field]]
            .dropna()
            .assign(old=lambda d, f=field: d[f].str.split("|"))
            .explode("old")
        )
        by_field[field] = pairs.groupby("old")["symbol"].unique()

    def resolve(symbol):
        if symbol in current:
            return symbol, current[symbol], "symbol"
        for field in ("prev_symbol", "alias_symbol"):
            hits = by_field[field].get(symbol)
            if hits is not None and len(hits) == 1:
                return hits[0], current[hits[0]], field
        return symbol, None, "unresolved"

    return resolve


def main(flags_path, bailey_path, hgnc_path):
    resolve = hgnc_resolver(hgnc_path)
    sources = {}
    for source, genes in (
        ("FLAGS_Shyr2014", read_flags(flags_path)),
        (
            "Bailey2018_tableS7_false_positive",
            read_bailey_s7(bailey_path),
        ),
    ):
        for gene in genes:
            sources.setdefault(gene, []).append(source)

    rows = []
    for gene, gene_sources in sources.items():
        symbol, ensembl, how = resolve(gene)
        rows.append(
            {
                "symbol": symbol,
                "ensembl_gene_id": ensembl,
                "symbol_in_source": gene,
                "ensembl_from": how,
                "source": "|".join(sorted(gene_sources)),
            }
        )
    table = pd.DataFrame(rows)
    unresolved = table[table["ensembl_gene_id"].isna()]
    if len(unresolved):
        print(
            "unresolved:", ", ".join(unresolved["symbol_in_source"])
        )
    table = (
        table.dropna(subset=["ensembl_gene_id"])
        .groupby(["symbol", "ensembl_gene_id"], as_index=False)
        .agg(
            symbol_in_source=("symbol_in_source", "|".join),
            ensembl_from=("ensembl_from", "first"),
            source=(
                "source",
                lambda s: "|".join(
                    sorted(set("|".join(s).split("|")))
                ),
            ),
        )
        .sort_values("symbol")
    )
    table.to_csv(OUT, sep="\t", index=False)
    print(f"{len(table)} genes -> {OUT}")
    print(table["source"].value_counts().to_string())
    print(table["ensembl_from"].value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
