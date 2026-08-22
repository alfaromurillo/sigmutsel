"""Tests for tcga_sample_selection.

Barcode parsing and the GDC cases-API lookup themselves are tested in
gdcfetch (`gdcfetch.tcga_barcode`, moved there so any gdcfetch-based
download can use them, see this module's docstring). Tests here cover
only what's specific to this pipeline: reading barcodes out of MAF
files, sample-type filtering, and duplicate-case resolution.
"""

from pathlib import Path

import pytest

from sigmutsel.tcga_sample_selection import (
    DEFAULT_KEEP_SAMPLE_TYPES,
    SAMPLE_TYPE_CODES,
    TcgaBarcodeInfo,
    catalog_maf_files,
    fetch_gdc_case_metadata,
    filter_by_sample_type,
    parse_tcga_barcode,
    read_maf_tumor_sample_barcode,
    select_one_per_case,
    select_tcga_maf_files,
)

_HEADER = "\t".join(  # noqa: FLY002
    [
        "Hugo_Symbol",
        "Chromosome",
        "Tumor_Sample_Barcode",
        "Variant_Type",
    ]
)


def _maf_text(barcode, *, with_data_row=True):
    text = "# version 2.4\n" + _HEADER + "\n"
    if with_data_row:
        text += f"KRAS\tchr12\t{barcode}\tSNP\n"
    return text


def _write_maf(tmp_path, name, barcode, **kwargs):
    p = tmp_path / name
    p.write_text(_maf_text(barcode, **kwargs))
    return p


def _catalog_from_barcodes(barcodes):
    return {
        Path(f"{bc}.maf"): parse_tcga_barcode(bc) for bc in barcodes
    }


# --- re-exports from gdcfetch.tcga_barcode ------------------------------


def test_reexports_from_gdcfetch():
    # sigmutsel.tcga_sample_selection re-exports these so callers
    # don't need a separate gdcfetch import for them.
    assert TcgaBarcodeInfo is not None
    assert SAMPLE_TYPE_CODES["01"] == "Primary Solid Tumor"
    assert parse_tcga_barcode("TCGA-AA-3971-01A").case_id == (
        "TCGA-AA-3971"
    )


def test_default_keep_sample_types_matches_cancereffectsizer():
    # cancereffectsizeR's get_TCGA_project_MAF() default
    # (exclude_TCGA_nonprimary = TRUE) keeps tissue types 01 and 03.
    assert DEFAULT_KEEP_SAMPLE_TYPES == frozenset({"01", "03"})


# --- read_maf_tumor_sample_barcode / catalog_maf_files ------------------


def test_read_maf_tumor_sample_barcode(tmp_path):
    p = _write_maf(tmp_path, "a.maf", "TCGA-AA-3971-01A-01W-0995-10")
    assert (
        read_maf_tumor_sample_barcode(p)
        == "TCGA-AA-3971-01A-01W-0995-10"
    )


def test_read_maf_tumor_sample_barcode_no_data_rows(tmp_path):
    p = _write_maf(
        tmp_path,
        "empty.maf",
        "TCGA-AA-3971-01A-01W-0995-10",
        with_data_row=False,
    )
    assert read_maf_tumor_sample_barcode(p) is None


def test_read_maf_tumor_sample_barcode_missing_column(tmp_path):
    p = tmp_path / "bad.maf"
    p.write_text("Hugo_Symbol\tVariant_Type\nKRAS\tSNP\n")
    assert read_maf_tumor_sample_barcode(p) is None


def test_catalog_maf_files(tmp_path):
    _write_maf(tmp_path, "a.maf", "TCGA-AA-3971-01A-01W-0995-10")
    _write_maf(tmp_path, "b.maf", "TCGA-BB-1111-06A-11D-A19A-08")
    _write_maf(
        tmp_path, "c.maf", "ignored", with_data_row=False
    )  # zero-mutation aliquot, skipped

    catalog = catalog_maf_files(tmp_path)

    assert len(catalog) == 2
    barcodes = {info.barcode for info in catalog.values()}
    assert barcodes == {
        "TCGA-AA-3971-01A-01W-0995-10",
        "TCGA-BB-1111-06A-11D-A19A-08",
    }


# --- filter_by_sample_type ----------------------------------------------


def test_filter_by_sample_type_default_keeps_primary_only(tmp_path):
    primary = _write_maf(
        tmp_path, "primary.maf", "TCGA-AA-3971-01A-01W-0995-10"
    )
    met = _write_maf(
        tmp_path, "met.maf", "TCGA-AA-3971-06A-11D-A19A-08"
    )
    catalog = {
        primary: parse_tcga_barcode("TCGA-AA-3971-01A-01W-0995-10"),
        met: parse_tcga_barcode("TCGA-AA-3971-06A-11D-A19A-08"),
    }

    kept = filter_by_sample_type(catalog)

    assert kept == {primary: catalog[primary]}


def test_filter_by_sample_type_custom_set():
    barcodes = {
        "TCGA-AA-3971-01A-01W-0995-10": "01",
        "TCGA-AA-3971-06A-11D-A19A-08": "06",
    }
    catalog = {
        Path(f"{bc}.maf"): parse_tcga_barcode(bc) for bc in barcodes
    }

    kept = filter_by_sample_type(catalog, keep_sample_types={"06"})

    assert len(kept) == 1
    assert next(iter(kept.values())).sample_type_code == "06"


# --- select_one_per_case --------------------------------------------


def test_select_one_per_case_keep_all_is_noop():
    catalog = _catalog_from_barcodes(
        [
            "TCGA-AA-3971-01A-01W-0995-10",
            "TCGA-AA-3971-01B-04D-A270-10",
        ]
    )
    result = select_one_per_case(catalog, "keep_all")
    assert result == catalog


def test_select_one_per_case_random_keeps_one_per_case():
    catalog = _catalog_from_barcodes(
        [
            "TCGA-AA-3971-01A-01W-0995-10",
            "TCGA-AA-3971-01B-04D-A270-10",
            "TCGA-BB-1111-01A-01W-0995-10",
        ]
    )
    result = select_one_per_case(catalog, "random", random_seed=0)
    case_ids = [info.case_id for info in result.values()]
    assert sorted(case_ids) == ["TCGA-AA-3971", "TCGA-BB-1111"]


def test_select_one_per_case_oldest_uses_days_to_collection():
    older = "TCGA-AA-3971-01A-01W-0995-10"
    newer = "TCGA-AA-3971-01B-04D-A270-10"
    catalog = _catalog_from_barcodes([older, newer])
    case_metadata = {
        "TCGA-AA-3971": {
            "sample_days_to_collection": {
                "TCGA-AA-3971-01A": 100,
                "TCGA-AA-3971-01B": 900,
            }
        }
    }

    result = select_one_per_case(
        catalog, "oldest", case_metadata=case_metadata
    )

    assert len(result) == 1
    assert next(iter(result.values())).barcode == older


def test_select_one_per_case_newest_uses_days_to_collection():
    older = "TCGA-AA-3971-01A-01W-0995-10"
    newer = "TCGA-AA-3971-01B-04D-A270-10"
    catalog = _catalog_from_barcodes([older, newer])
    case_metadata = {
        "TCGA-AA-3971": {
            "sample_days_to_collection": {
                "TCGA-AA-3971-01A": 100,
                "TCGA-AA-3971-01B": 900,
            }
        }
    }

    result = select_one_per_case(
        catalog, "newest", case_metadata=case_metadata
    )

    assert len(result) == 1
    assert next(iter(result.values())).barcode == newer


def test_select_one_per_case_oldest_falls_back_when_dates_missing():
    catalog = _catalog_from_barcodes(
        [
            "TCGA-AA-3971-01A-01W-0995-10",
            "TCGA-AA-3971-01B-04D-A270-10",
        ]
    )
    result = select_one_per_case(catalog, "oldest", case_metadata={})
    assert len(result) == 1  # deterministic fallback, doesn't crash


def test_select_one_per_case_requires_metadata_for_dated_policies():
    catalog = _catalog_from_barcodes(["TCGA-AA-3971-01A-01W-0995-10"])
    with pytest.raises(ValueError, match="requires case_metadata"):
        select_one_per_case(catalog, "oldest")


def test_select_one_per_case_invalid_policy_raises():
    with pytest.raises(ValueError, match="policy must be one of"):
        select_one_per_case({}, "bogus")


# --- fetch_gdc_case_metadata (thin alias onto gdcfetch; mocked HTTP) --


class _CasesResponse:
    def __init__(self, hits):
        self._hits = hits

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": {"hits": self._hits}}


class _CasesSession:
    def __init__(self, hits):
        self.hits = hits

    def post(self, url, json=None, timeout=None):
        return _CasesResponse(self.hits)


def test_fetch_gdc_case_metadata_is_gdcfetch_fetch_case_metadata():
    from gdcfetch.tcga_barcode import fetch_case_metadata

    assert fetch_gdc_case_metadata is fetch_case_metadata


def test_fetch_gdc_case_metadata_parses_response():
    hits = [
        {
            "submitter_id": "TCGA-BL-A0C8",
            "diagnoses": [{"prior_treatment": "No"}],
            "samples": [
                {
                    "submitter_id": "TCGA-BL-A0C8-01A",
                    "days_to_collection": 213,
                },
                {
                    "submitter_id": "TCGA-BL-A0C8-01B",
                    "days_to_collection": 806,
                },
            ],
        }
    ]

    metadata = fetch_gdc_case_metadata(
        ["TCGA-BL-A0C8"], session=_CasesSession(hits)
    )

    assert metadata["TCGA-BL-A0C8"]["prior_treatment"] == "No"
    assert metadata["TCGA-BL-A0C8"]["sample_days_to_collection"] == {
        "TCGA-BL-A0C8-01A": 213,
        "TCGA-BL-A0C8-01B": 806,
    }


# --- select_tcga_maf_files (integration, network mocked) -----------


def test_select_tcga_maf_files_default_filters_and_keeps_all_dups(
    tmp_path,
):
    _write_maf(
        tmp_path, "primary_a.maf", "TCGA-AA-3971-01A-01W-0995-10"
    )
    _write_maf(
        tmp_path, "primary_b.maf", "TCGA-AA-3971-01B-04D-A270-10"
    )
    _write_maf(tmp_path, "met.maf", "TCGA-AA-3971-06A-11D-A19A-08")

    selected = select_tcga_maf_files(tmp_path)

    assert len(selected) == 2  # both primary aliquots, met dropped
    kept_names = {p.name for p in selected}
    assert kept_names == {"primary_a.maf", "primary_b.maf"}


def test_select_tcga_maf_files_random_policy_dedupes(tmp_path):
    _write_maf(
        tmp_path, "primary_a.maf", "TCGA-AA-3971-01A-01W-0995-10"
    )
    _write_maf(
        tmp_path, "primary_b.maf", "TCGA-AA-3971-01B-04D-A270-10"
    )

    selected = select_tcga_maf_files(
        tmp_path, duplicate_policy="random", random_seed=0
    )

    assert len(selected) == 1


def test_select_tcga_maf_files_exclude_prior_treatment(
    tmp_path, monkeypatch
):
    _write_maf(tmp_path, "a.maf", "TCGA-AA-3971-01A-01W-0995-10")
    _write_maf(tmp_path, "b.maf", "TCGA-BB-1111-01A-01W-0995-10")

    hits = [
        {
            "submitter_id": "TCGA-AA-3971",
            "diagnoses": [{"prior_treatment": "Yes"}],
            "samples": [],
        },
        {
            "submitter_id": "TCGA-BB-1111",
            "diagnoses": [{"prior_treatment": "No"}],
            "samples": [],
        },
    ]

    def _fake_session_class():
        return _CasesSession(hits)

    monkeypatch.setattr(
        "gdcfetch.tcga_barcode.requests.Session",
        _fake_session_class,
    )

    selected = select_tcga_maf_files(
        tmp_path, exclude_prior_treatment=True
    )

    assert len(selected) == 1
    assert selected[0].name == "b.maf"
