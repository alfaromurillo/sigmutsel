"""Germline masks: alleles a somatic call set was filtered on.

Somatic pipelines often drop calls at known germline alleles, to keep
unfiltered polymorphisms out of the catalogue. Those alleles are then
unobservable, and an opportunity that still counts them over-states
the chance of seeing a mutation there. The error is not neutral:
common germline alleles concentrate at synonymous sites, because
germline purifying selection spares them, so the unmasked synonymous
opportunity is inflated more than the non-synonymous one, and the
non-synonymous/synonymous offset absorbs the difference (Martincorena
et al. 2017, *Cell* 171:1029, STAR Methods, "Impact of germline SNP
contamination or SNP over-filtering").

The fix is the one used for the capture territory: remove the masked
(site, allele) pairs from the opportunity, channel by channel.

The mask is built from gnomAD v2.1.1 exomes (GRCh38 liftover), the
release VEP annotates GDC MAFs with (their ``gnomAD_AF`` ...
``gnomAD_SAS_AF`` columns). :func:`build_gnomad_snv_table` streams the
per-chromosome sites VCFs once (about 92 GB compressed; nothing but
the resulting table is stored) and keeps every SNV with its overall
allele frequency, so the frequency threshold can be chosen later
without downloading again. :func:`load_germline_mask` turns the table
into the sorted allele keys every other function takes.

Keys encode ``(chromosome, GRCh38 position, alternate base)`` of the
plus strand as one ``int64``; see :func:`allele_keys`.
"""

import gzip
import logging
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GNOMAD_V211_EXOMES_GRCH38_URL = (
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/"
    "2.1.1/liftover_grch38/vcf/exomes/"
    "gnomad.exomes.r2.1.1.sites.{chrom}.liftover_grch38.vcf.bgz"
)
CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y"]
_BASE_CODE = {"A": 0, "C": 1, "G": 2, "T": 3}
_CHROM_CODE = {
    **{str(i): i for i in range(1, 23)},
    "X": 23,
    "Y": 24,
}


def default_table_path():
    """Where the gnomAD SNV table is cached (under the data dir)."""
    from .locations import location_germline_mask_dir

    return (
        Path(location_germline_mask_dir)
        / "gnomad_v2.1.1_exomes_grch38_snv_af.tsv.gz"
    )


def mask_label(af_threshold):
    """Short name of a mask, used in cache paths and manifests."""
    return f"gnomad-v2.1.1-exomes-af{af_threshold:g}"


def allele_keys(chrom, pos, alt_code):
    """One ``int64`` per (chromosome, position, alternate base).

    ``chrom`` may carry a ``chr`` prefix; unknown chromosomes get code
    0 and so never match a gnomAD key. ``alt_code`` is 0..3 for
    A, C, G, T on the plus strand.
    """
    codes = (
        pd.Series(np.asarray(chrom))
        .astype(str)
        .str.replace("chr", "", regex=False)
        .map(_CHROM_CODE)
        .fillna(0)
        .astype(np.int64)
        .to_numpy()
    )
    pos = np.asarray(pos, dtype=np.int64)
    return (codes * (1 << 30) + pos) * 4 + np.asarray(
        alt_code, dtype=np.int64
    )


def _parse_af(info):
    """Overall AF from a VCF INFO field (bytes), or None."""
    if info.startswith(b"AF="):
        start = 3
    else:
        i = info.find(b";AF=")
        if i < 0:
            return None
        start = i + 4
    end = info.find(b";", start)
    value = info[start:] if end < 0 else info[start:end]
    try:
        return float(value)
    except ValueError:
        return None


def _stream_chromosome(chrom, url_template, out_path, retries=3):
    """Write one chromosome's SNVs (AF > 0) to ``out_path``; stats."""
    url = url_template.format(chrom=chrom)
    for attempt in range(1, retries + 1):
        n_records = n_snv = n_kept = 0
        try:
            with (
                urllib.request.urlopen(url) as resp,
                gzip.GzipFile(fileobj=resp) as vcf,
                open(out_path, "w") as out,
            ):
                for line in vcf:
                    if line[:1] == b"#":
                        continue
                    n_records += 1
                    f = line.split(b"\t", 8)
                    ref, alt = f[3], f[4]
                    if len(ref) != 1 or len(alt) != 1:
                        continue
                    n_snv += 1
                    af = _parse_af(f[7])
                    if af is None or af <= 0:
                        continue
                    n_kept += 1
                    out.write(
                        f"{f[0].decode()}\t{f[1].decode()}\t"
                        f"{ref.decode()}\t{alt.decode()}\t"
                        f"{af:.6g}\t{f[6].decode()}\n"
                    )
            return {
                "chrom": chrom,
                "records": n_records,
                "snvs": n_snv,
                "snvs_with_af": n_kept,
            }
        except OSError as err:
            logger.warning(
                "gnomAD chr%s attempt %d failed: %s",
                chrom,
                attempt,
                err,
            )
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not stream gnomAD chromosome {chrom}")


def build_gnomad_snv_table(
    path=None,
    chromosomes=None,
    n_workers=4,
    url_template=GNOMAD_V211_EXOMES_GRCH38_URL,
):
    """Stream gnomAD v2.1.1 exomes and keep every SNV with its AF.

    Nothing but the table is written: each chromosome's VCF is read
    straight off the network. About 92 GB are streamed in all (15-40
    minutes on a university link, a few hours on a home one); the
    table has about 16 million rows.

    Parameters
    ----------
    path : path-like, optional
        Output (gzipped TSV: chrom, pos, ref, alt, af, filter);
        default :func:`default_table_path`.
    chromosomes : list of str, optional
        Default all of 1-22, X, Y.
    n_workers : int
        Chromosomes streamed at once.
    url_template : str
        With a ``{chrom}`` field.

    Returns
    -------
    pathlib.Path
    """
    path = Path(path or default_table_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    chromosomes = chromosomes or CHROMOSOMES
    parts = {
        c: path.parent / f".{path.name}.chr{c}.part"
        for c in chromosomes
    }
    logger.info(
        "Streaming gnomAD v2.1.1 exomes (%d chromosomes, %d at a time)",
        len(chromosomes),
        n_workers,
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        stats = list(
            pool.map(
                _stream_chromosome,
                chromosomes,
                [url_template] * len(chromosomes),
                [parts[c] for c in chromosomes],
            )
        )
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as out:
        out.write("chrom\tpos\tref\talt\taf\tfilter\n")
        for c in chromosomes:
            with open(parts[c]) as part:
                for line in part:
                    out.write(line)
    tmp.replace(path)
    for part in parts.values():
        part.unlink(missing_ok=True)
    total = sum(s["snvs_with_af"] for s in stats)
    logger.info("gnomAD SNV table: %d SNVs -> %s", total, path)
    return path


def load_germline_mask(
    af_threshold, path=None, build_if_missing=True
):
    """Sorted allele keys of the gnomAD SNVs with ``AF > af_threshold``.

    The keys are cached next to the table, one file per threshold.

    Parameters
    ----------
    af_threshold : float
        Overall allele frequency above which an allele is masked.
    path : path-like, optional
        The table of :func:`build_gnomad_snv_table`.
    build_if_missing : bool
        Build the table when it does not exist yet (a long stream;
        see :func:`build_gnomad_snv_table`).

    Returns
    -------
    numpy.ndarray of int64
    """
    path = Path(path or default_table_path())
    cache = path.parent / f"{mask_label(af_threshold)}.keys.npy"
    if cache.exists() and (
        not path.exists()
        or cache.stat().st_mtime >= path.stat().st_mtime
    ):
        return np.load(cache)
    if not path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"{path} does not exist; run "
                "sigmutsel.germline_mask.build_gnomad_snv_table()."
            )
        logger.warning(
            "The gnomAD SNV table is missing; building it now "
            "(streams about 92 GB, once)."
        )
        build_gnomad_snv_table(path)
    table = pd.read_csv(
        path,
        sep="\t",
        usecols=["chrom", "pos", "alt", "af"],
        dtype={
            "chrom": str,
            "pos": np.int64,
            "alt": str,
            "af": float,
        },
    )
    table = table[table["af"] > af_threshold]
    alt = table["alt"].map(_BASE_CODE)
    table = table[alt.notna()]
    keys = np.unique(
        allele_keys(
            table["chrom"], table["pos"], alt[alt.notna()].astype(int)
        )
    )
    np.save(cache, keys)
    logger.info(
        "Germline mask %s: %d alleles",
        mask_label(af_threshold),
        len(keys),
    )
    return keys


def calls_on_masked_alleles(db, keys):
    """Boolean Series: which calls of a MAF-style table are masked.

    Uses ``Chromosome``, ``Start_Position`` and ``Tumor_Seq_Allele2``
    (plus strand). A masked MAF should have almost none; the few that
    survive are calls a pipeline rescued (for GDC, calls also seen by
    MC3 or listed in COSMIC).
    """
    alt = db["Tumor_Seq_Allele2"].map(_BASE_CODE)
    ok = alt.notna().to_numpy()
    out = np.zeros(len(db), dtype=bool)
    if ok.any():
        k = allele_keys(
            db["Chromosome"].to_numpy()[ok],
            db["Start_Position"].to_numpy()[ok],
            alt[ok].astype(int).to_numpy(),
        )
        out[ok] = np.isin(k, keys)
    return pd.Series(out, index=db.index)
