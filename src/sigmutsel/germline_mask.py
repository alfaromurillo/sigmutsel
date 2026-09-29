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
``gnomAD_SAS_AF`` columns). Two tables, one format (chrom, pos, ref,
alt, af, filter):

- **The distributed table** (default): every SNV with overall AF above
  ``DISTRIBUTED_FLOOR`` (5e-5; 1.8 million SNVs, 15 MB), downloaded
  from a sigmutsel GitHub release and checked against its SHA-256. It
  serves any threshold at or above the floor, which covers where a
  frequency filter acts (GDC's removes nothing below 1e-4).
- **The full table**: every SNV with AF > 0 (14.6 million), built by
  :func:`build_gnomad_snv_table`, which streams the per-chromosome
  sites VCFs once -- about 92 GB. Only needed for thresholds below the
  floor, or to measure call survival by frequency bin. Used whenever
  it exists.

gnomAD data are CC0; the project asks for attribution (Karczewski et
al. 2020, *Nature* 581:434). :func:`load_germline_mask` turns a table
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


DISTRIBUTED_FLOOR = 5e-5
DISTRIBUTED_NAME = "gnomad_v2.1.1_exomes_grch38_snv_af_gt5e-05.tsv.gz"
DISTRIBUTED_URL = (
    "https://github.com/alfaromurillo/sigmutsel/releases/download/"
    "data-germline-mask/" + DISTRIBUTED_NAME
)
DISTRIBUTED_SHA256 = (
    "53f3220cc609301280a5f55a19aec3683bec94fc683b1bcc0fd6b36210bd236d"
)


def default_table_path():
    """Where the full gnomAD SNV table is cached (under the data dir)."""
    from .locations import location_germline_mask_dir

    return (
        Path(location_germline_mask_dir)
        / "gnomad_v2.1.1_exomes_grch38_snv_af.tsv.gz"
    )


def distributed_table_path():
    """Where the distributed (AF > 5e-5) table is cached."""
    from .locations import location_germline_mask_dir

    return Path(location_germline_mask_dir) / DISTRIBUTED_NAME


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_germline_mask_table(force=False, url=DISTRIBUTED_URL):
    """Download the distributed table (15 MB) and check its SHA-256.

    Returns
    -------
    pathlib.Path
    """
    dest = distributed_table_path()
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    logger.info("Downloading the germline-mask table from %s", url)
    urllib.request.urlretrieve(url, tmp)
    digest = _sha256(tmp)
    if digest != DISTRIBUTED_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {url}: got {digest}, expected "
            f"{DISTRIBUTED_SHA256}."
        )
    tmp.replace(dest)
    return dest


def write_distributed_table(
    full_path=None, out_path=None, floor=DISTRIBUTED_FLOOR
):
    """Cut the distributed table out of a full one (for maintainers).

    Deterministic (gzip ``mtime=0``), so the same full table always
    gives the same file and checksum.
    """
    full_path = Path(full_path or default_table_path())
    out_path = Path(out_path or distributed_table_path())
    n = 0
    with (
        gzip.open(full_path, "rt") as fin,
        gzip.GzipFile(
            out_path, "wb", compresslevel=9, mtime=0
        ) as fout,
    ):
        fout.write(fin.readline().encode())
        for line in fin:
            if float(line.split("\t", 5)[4]) > floor:
                fout.write(line.encode())
                n += 1
    logger.info("Distributed table: %d SNVs -> %s", n, out_path)
    return out_path


def resolve_table(af_threshold, path=None, build_full=False):
    """The table that serves ``af_threshold``, fetched if need be.

    ``path`` wins; then the full table if it exists; then the
    distributed one (downloaded), if the threshold is at or above its
    floor. Below the floor the full table is required: it is built
    only when ``build_full`` is True, because that streams ~92 GB.
    """
    if path is not None:
        return Path(path)
    full = default_table_path()
    if full.exists():
        return full
    if af_threshold >= DISTRIBUTED_FLOOR:
        return download_germline_mask_table()
    if not build_full:
        raise FileNotFoundError(
            f"A germline mask at AF > {af_threshold:g} needs the full "
            f"gnomAD SNV table (the distributed one starts at "
            f"{DISTRIBUTED_FLOOR:g}). Build it with "
            "build_gnomad_snv_table() -- it streams about 92 GB -- or "
            "pass build_full=True."
        )
    logger.warning(
        "Building the full gnomAD SNV table: this streams about 92 GB "
        "(15-40 minutes on a fast link, hours on a slow one)."
    )
    return build_gnomad_snv_table(full)


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


def load_germline_mask(af_threshold, path=None, build_full=False):
    """Sorted allele keys of the gnomAD SNVs with ``AF > af_threshold``.

    The keys are cached next to the table, one file per threshold; a
    threshold at or above the distributed floor gives the same keys
    from either table.

    Parameters
    ----------
    af_threshold : float
        Overall allele frequency above which an allele is masked.
    path : path-like, optional
        A table in :func:`build_gnomad_snv_table`'s format; default
        :func:`resolve_table`.
    build_full : bool
        Allow building the full table (a ~92 GB stream) when the
        threshold is below the distributed floor and it is missing.

    Returns
    -------
    numpy.ndarray of int64
    """
    if path is None:
        cache_dir = default_table_path().parent
        cache = cache_dir / f"{mask_label(af_threshold)}.keys.npy"
        if cache.exists():
            return np.load(cache)
    path = resolve_table(af_threshold, path, build_full=build_full)
    cache = path.parent / f"{mask_label(af_threshold)}.keys.npy"
    if (
        cache.exists()
        and cache.stat().st_mtime >= path.stat().st_mtime
    ):
        return np.load(cache)
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
