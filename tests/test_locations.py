"""A custom data directory (SIGMUTSEL_DATA_DIR, sigmutsel-setup
--data-dir) must still find the files bundled with the package."""

import os
import subprocess
import sys

BUNDLED = (
    "location_exclusion_signatures_matrix",
    "location_inclusion_signatures_matrix",
    "location_cosmic_signature_etiology",
    "location_cancer_gene_census",
    "location_cosmic_cancer_gene_census",
    "location_intogen_driver_genes",
    "location_aid_target_genes",
    "location_published_driver_genes",
    "location_histone_genes",
    "location_artifact_prone_genes",
)


def _locations_under(data_dir):
    """The bundled paths ``sigmutsel.locations`` resolves under a fresh
    interpreter with ``SIGMUTSEL_DATA_DIR=data_dir``."""
    code = (
        "from sigmutsel import locations as L; "
        f"print(L.DATA_DIR); [print(getattr(L, n)) for n in {BUNDLED!r}]"
    )
    env = {**os.environ, "SIGMUTSEL_DATA_DIR": str(data_dir)}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return out[0], out[1:]


def test_bundled_files_found_under_an_empty_custom_data_dir(tmp_path):
    data_dir, paths = _locations_under(tmp_path)

    assert data_dir == str(tmp_path)
    for path in paths:
        assert os.path.exists(path), path
        assert not path.startswith(str(tmp_path))


def test_a_copy_in_the_custom_data_dir_takes_precedence(tmp_path):
    name = "histone_genes.tsv"
    (tmp_path / name).write_text("ensembl_gene_id\n")

    _, paths = _locations_under(tmp_path)

    assert str(tmp_path / name) in paths
