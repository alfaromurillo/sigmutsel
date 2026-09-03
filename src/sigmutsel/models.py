"""Data models for mutation rate analysis."""

import inspect
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(repr=False)
class MutationDataset:
    """Container for mutation and variant data.

    This class groups all the raw data that is shared across
    different models. Most data is lazy-loaded since not all
    analyses require it.

    Attributes
    ----------
    location_maf_files : str or Path
        Path to directory containing raw MAF files.
    signature_class : str, default "SBS"
        COSMIC signature class. Must be one of:
        - "SBS": Single base substitution signatures
        - "DBS": Doublet base substitution signatures
        - "ID": Insertion/deletion signatures
        - "CN": Copy number signatures
        - "SV": Structural variant signatures
        - "RNA-SBS": RNA single base substitution signatures
    source_maf : str or pathlib.Path, optional
        Path to a single multi-sample MAF file. If provided and
        ``location_maf_files`` does not yet contain per-sample
        ``.maf`` files, the file is split automatically on
        dataset creation. Skipped if per-sample files already
        exist.
    mutation_db : pd.DataFrame or None
        Mutation database with individual mutation records.
        Columns include: gene, tumor_sample_barcode,
        ensembl_gene_id, chromosome, position, variant, etc.
        Lazy-loaded via generate_mutation_db() or load_dataset().
    genes_present : pd.DataFrame or None
        Binary matrix (genes × samples) indicating which genes
        are mutated in which samples. Lazy-loaded.
    genes_present_non_silent : pd.DataFrame or None
        Same as genes_present but for non-silent mutations
        only. Lazy-loaded.
    genes_present_silent : pd.DataFrame or None
        Same as genes_present but for silent (synonymous)
        mutations only -- the other half of the consequence split,
        and the one that stays selection-free in driver genes too.
        Lazy-loaded.
    genes_counts_silent, genes_counts_non_silent : pd.DataFrame or None
        The same two matrices *before* they are censored to 0/1:
        mutation counts per gene per sample. Presence is a censored
        count, and the censoring throws away most of the information
        precisely in the hypermutated samples that carry most of a
        cohort's mutation mass. Lazy-loaded.
    variant_db : pd.DataFrame or None
        Table of unique variants annotated with genomic context
        and mutation types. Lazy-loaded via generate_variant_db()
        or load_dataset().
    variants_present : pd.DataFrame or None
        Binary matrix (variants × samples) indicating which variants
        are present in which samples. Lazy-loaded.
    sig_assignments : pd.DataFrame or None
        Signature assignment matrix (samples × signatures) from
        COSMIC signature decomposition. Each value represents the
        number of mutations attributed to each signature in each
        sample. Lazy-loaded via run_signature_decomposition().
    signature_matrix : pd.DataFrame or None
        Normalized signature matrix (mutation types × signatures)
        from COSMIC signature decomposition. Each column represents
        a signature and sums to 1, showing the probability
        distribution over mutation types for that signature.
        Automatically loaded after run_signature_decomposition().
    contexts_by_gene : pd.DataFrame or None
        Trinucleotide context counts by gene (genes × contexts). Each
        value represents the count of a specific trinucleotide context
        in a gene's coding sequence. Lazy-loaded via
        generate_contexts_by_gene(), whose `gene_universe` argument
        controls which genes are included ("own_cohort", the
        default: genes present in the mutation database only;
        "wes_target": the union of that set with MC3's TCGA WES-target
        gene set, a better-calibrated denominator for low-mutation-
        burden cohorts). These values are needed to compute later the
        probability that a mutation of a certain type lands on a
        gene. Thus, gene mutation rates can only be obtained for
        genes with this information, and so the index of
        contexts_by_gene is the maximal scope of the analysis.
    contexts_by_gene_syn, contexts_by_gene_nonsyn : pd.DataFrame or None
        Consequence-split opportunity counts (genes × 96 canonical SBS
        types): the synonymous and non-synonymous share of the same
        opportunities `contexts_by_gene` counts, so that
        ``syn[τ] + nonsyn[τ] == contexts_by_gene[context(τ)]`` for
        every type τ. SBS-only (see
        generate_consequence_contexts_by_gene()) and lazy-loaded --
        nothing in the default pipeline populates or consumes them
        yet.
    signature_reference_genome : str or None
        Reference genome (same as genome_build) used when generating
        mutational matrices for signature decomposition. Populated
        when run_signature_decomposition() succeeds.
    signature_exome : bool or None
        Whether exome-normalized signatures were used during
        decomposition. Populated when run_signature_decomposition()
        succeeds.
    signature_cosmic_version : float or None
        COSMIC signature release used during decomposition. Populated
        when run_signature_decomposition() succeeds.
    signature_genome_build : str or None
        Genome build passed to signature decomposition. Populated when
        run_signature_decomposition() succeeds.

    Examples
    --------
    >>> from models import MutationDataset
    >>> from coad_locations import location_all_maf_files
    >>>
    >>> # Create SBS dataset (default)
    >>> dataset = MutationDataset(
    ...     location_maf_files=location_all_maf_files,
    ...     signature_class="SBS")
    >>>
    >>> # Generate mutation database (optionally saving to disk)
    >>> dataset.generate_mutation_db("data/mutations_sbs.parquet")
    >>>
    >>> # Access data and compute derived matrices
    >>> print(f"Loaded {dataset.n_samples} samples")
    >>> dataset.compute_gene_presence()
    >>> dataset.generate_variant_db()
    >>> dataset.compute_variants_present()
    >>>
    >>> # Run signature decomposition (auto-generates matrices if needed)
    >>> assignments = dataset.run_signature_decomposition(
    ...     exome=True,
    ...     cosmic_version=3.4,
    ...     genome_build='GRCh38')
    >>> print(f"Found {assignments.shape[1]} active signatures")
    >>>
    >>> # Load trinucleotide contexts by gene
    >>> from locations import location_cds_fasta
    >>> contexts = dataset.generate_contexts_by_gene(
    ...     location_cds_fasta)
    >>> print(f"Contexts for {contexts.shape[0]} genes")
    >>>
    >>> # Create ID (indel) dataset
    >>> dataset_id = MutationDataset(
    ...     location_maf_files=location_all_maf_files,
    ...     signature_class="ID")
    >>> dataset_id.generate_mutation_db("data/mutations_id.parquet")

    """

    location_maf_files: str
    signature_class: str = "SBS"
    source_maf: str | None = None

    # Lazy-loaded attributes
    _mutation_db: pd.DataFrame = None
    _genes_present: pd.DataFrame = None
    _genes_present_non_silent: pd.DataFrame = None
    _genes_present_silent: pd.DataFrame = None
    _genes_counts_silent: pd.DataFrame = None
    _genes_counts_non_silent: pd.DataFrame = None
    _variant_db: pd.DataFrame = None
    _variants_present: pd.DataFrame = None
    _sig_assignments: pd.DataFrame = None
    _signature_matrix: pd.DataFrame = None
    _signature_reference_genome: str | None = None
    _signature_exome: bool | None = None
    _signature_cosmic_version: float | None = None
    _signature_genome_build: str | None = None
    _contexts_by_gene: pd.DataFrame = None
    _contexts_by_gene_gene_universe: str | None = None
    _contexts_by_gene_syn: pd.DataFrame = None
    _contexts_by_gene_nonsyn: pd.DataFrame = None
    _sample_qc_flags: pd.DataFrame = None
    dataset_directory: str | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        """Split source MAF or warn if MAF directory is empty."""
        maf_dir = Path(self.location_maf_files)
        maf_files_exist = maf_dir.exists() and any(
            maf_dir.glob("*.maf")
        )

        if self.source_maf is not None:
            from sigmutsel.split_maf_file import split_maf_file

            split_maf_file(self.source_maf, maf_dir)
        elif not maf_files_exist:
            logger.warning(
                f"No .maf files found in {maf_dir}. "
                "If you have a single multi-sample MAF file, "
                "pass it as source_maf='path/to/file.maf' "
                "when creating the dataset."
            )

    def __repr__(self):
        """Show loaded status of lazy attributes (custom repr)."""
        loaded = []
        if self._mutation_db is not None:
            loaded.append(
                f"mutation_db: {self._mutation_db.shape[0]} rows"
            )
        if self._genes_present is not None:
            loaded.append(
                f"genes_present: {self._genes_present.shape}"
            )
        if self._genes_present_non_silent is not None:
            loaded.append(
                f"genes_present_non_silent: "
                f"{self._genes_present_non_silent.shape}"
            )
        if self._genes_present_silent is not None:
            loaded.append(
                f"genes_present_silent: "
                f"{self._genes_present_silent.shape}"
            )
        if self._genes_counts_silent is not None:
            loaded.append(
                "genes_counts_silent/non_silent: "
                f"{self._genes_counts_silent.shape}"
            )
        if self._variant_db is not None:
            loaded.append(
                f"variant_db: {self._variant_db.shape[0]} variants"
            )
        if self._variants_present is not None:
            loaded.append(
                f"variants_present: {self._variants_present.shape}"
            )
        if self._sig_assignments is not None:
            loaded.append(
                f"sig_assignments: {self._sig_assignments.shape}"
            )
        if self._signature_matrix is not None:
            loaded.append(
                f"signature_matrix: {self._signature_matrix.shape}"
            )
        if self._contexts_by_gene is not None:
            loaded.append(
                f"contexts_by_gene: {self._contexts_by_gene.shape}"
            )
        if self._contexts_by_gene_syn is not None:
            loaded.append(
                "contexts_by_gene_syn/nonsyn: "
                f"{self._contexts_by_gene_syn.shape}"
            )
        if self._sample_qc_flags is not None:
            loaded.append(
                f"sample_qc_flags: {self._sample_qc_flags.shape[0]} "
                "samples"
            )

        loaded_str = "\n  ".join(loaded) if loaded else "None"

        return (
            f"MutationDataset(\n"
            f"  signature_class={self.signature_class!r}\n"
            f"  location_maf_files={self.location_maf_files!r}\n"
            f"  loaded_data:\n  {loaded_str}\n"
            f")"
        )

    def save_dataset(self, directory):
        """Persist loaded dataset artifacts to a directory."""

        directory = Path(directory)
        manifest_path = directory / "dataset_manifest.json"

        if manifest_path.exists():
            response = (
                input(
                    f"Dataset already exists at {directory}. "
                    "Overwrite? [y/N]: "
                )
                .strip()
                .lower()
            )
            if response not in {"y", "yes"}:
                raise FileExistsError(
                    f"Dataset directory {directory} already exists."
                )

        directory.mkdir(parents=True, exist_ok=True)

        data_specs = [
            (
                "mutation_db",
                "_mutation_db",
                "mutation_db.parquet",
                "parquet",
            ),
            (
                "genes_present",
                "_genes_present",
                "genes_present.parquet",
                "parquet",
            ),
            (
                "genes_present_non_silent",
                "_genes_present_non_silent",
                "genes_present_non_silent.parquet",
                "parquet",
            ),
            (
                "genes_present_silent",
                "_genes_present_silent",
                "genes_present_silent.parquet",
                "parquet",
            ),
            (
                "genes_counts_silent",
                "_genes_counts_silent",
                "genes_counts_silent.parquet",
                "parquet",
            ),
            (
                "genes_counts_non_silent",
                "_genes_counts_non_silent",
                "genes_counts_non_silent.parquet",
                "parquet",
            ),
            (
                "variant_db",
                "_variant_db",
                "variant_db.parquet",
                "parquet",
            ),
            (
                "variants_present",
                "_variants_present",
                "variants_present.parquet",
                "parquet",
            ),
            (
                "sig_assignments",
                "_sig_assignments",
                "sig_assignments.parquet",
                "parquet",
            ),
            (
                "signature_matrix",
                "_signature_matrix",
                "signature_matrix.parquet",
                "parquet",
            ),
            (
                "contexts_by_gene",
                "_contexts_by_gene",
                "contexts_by_gene.csv",
                "csv",
            ),
            (
                "contexts_by_gene_syn",
                "_contexts_by_gene_syn",
                "contexts_by_gene_syn.csv",
                "csv",
            ),
            (
                "contexts_by_gene_nonsyn",
                "_contexts_by_gene_nonsyn",
                "contexts_by_gene_nonsyn.csv",
                "csv",
            ),
            (
                "sample_qc_flags",
                "_sample_qc_flags",
                "sample_qc_flags.parquet",
                "parquet",
            ),
        ]

        saved_files = {}

        for public_name, private_name, filename, fmt in data_specs:
            value = getattr(self, private_name)
            if value is None:
                continue

            file_path = directory / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if fmt == "parquet":
                # Convert list columns to JSON strings for parquet compatibility
                value_to_save = value.copy()
                for col in value_to_save.columns:
                    if (
                        value_to_save[col]
                        .apply(lambda x: isinstance(x, list))
                        .any()
                    ):
                        value_to_save[col] = value_to_save[col].apply(
                            lambda x: (
                                json.dumps(x)
                                if isinstance(x, list)
                                else x
                            )
                        )
                value_to_save.to_parquet(file_path)
            elif fmt == "csv":
                value.to_csv(file_path)
            else:
                raise ValueError(
                    f"Unsupported format {fmt} for {public_name}"
                )

            saved_files[public_name] = {
                "filename": filename,
                "format": fmt,
            }

        manifest = {
            # Bumped 2 -> adds sample_qc_flags (see the L_low
            # low-burden-correction plan); 3 -> adds the optional
            # consequence-split opportunity tables
            # (contexts_by_gene_syn/nonsyn). Not read/validated on
            # load -- documentary only, so an old manifest (version
            # 1, no sample_qc_flags file) still loads fine, and a
            # version-3 manifest without the split tables is the
            # normal case, not a defect.
            "version": 3,
            "signature_class": self.signature_class,
            "location_maf_files": str(self.location_maf_files),
            "source_maf": (
                str(self.source_maf)
                if self.source_maf is not None
                else None
            ),
            "files": saved_files,
            "contexts_by_gene_gene_universe": (
                self._contexts_by_gene_gene_universe
            ),
            "signature_parameters": {
                "reference_genome": self._signature_reference_genome,
                "exome": self._signature_exome,
                "cosmic_version": self._signature_cosmic_version,
                "genome_build": self._signature_genome_build,
            },
        }

        manifest_path.write_text(json.dumps(manifest, indent=2))
        self.dataset_directory = str(directory)

    @classmethod
    def load_dataset(cls, directory):
        """Load dataset artifacts from a directory created by save_dataset."""

        directory = Path(directory)
        manifest_path = directory / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at {manifest_path}. "
                "Ensure save_dataset() was run first."
            )

        manifest = json.loads(manifest_path.read_text())
        dataset = cls(
            location_maf_files=manifest.get("location_maf_files"),
            signature_class=manifest.get("signature_class", "SBS"),
            source_maf=manifest.get("source_maf"),
        )
        # Manifests predating this field used the (then-only)
        # own-cohort-restricted behavior, so that's the correct
        # default for old manifests, not an unknown/null state.
        dataset._contexts_by_gene_gene_universe = manifest.get(
            "contexts_by_gene_gene_universe", "own_cohort"
        )

        for attr_name, info in manifest.get("files", {}).items():
            filename = info["filename"]
            fmt = info.get(
                "format", Path(filename).suffix.lstrip(".")
            )
            file_path = directory / filename

            if fmt == "parquet":
                value = pd.read_parquet(file_path)
                # Convert JSON strings back to lists for columns like mut_types
                for col in value.columns:
                    # Check if column contains JSON array strings
                    sample = (
                        value[col].dropna().iloc[0]
                        if not value[col].dropna().empty
                        else None
                    )
                    if isinstance(sample, str) and sample.startswith(
                        "["
                    ):
                        try:
                            value[col] = value[col].apply(
                                lambda x: (
                                    json.loads(x)
                                    if isinstance(x, str)
                                    and x.startswith("[")
                                    else x
                                )
                            )
                        except (json.JSONDecodeError, TypeError):
                            # If it's not valid JSON, leave it as is
                            pass
            elif fmt == "csv":
                value = pd.read_csv(file_path, index_col=0)
            else:
                raise ValueError(
                    f"Unsupported file format {fmt} for {attr_name}"
                )

            setattr(dataset, attr_name, value)

        signature_params = manifest.get("signature_parameters", {})
        dataset._signature_reference_genome = signature_params.get(
            "reference_genome"
        )
        dataset._signature_exome = signature_params.get("exome")
        dataset._signature_cosmic_version = signature_params.get(
            "cosmic_version"
        )
        dataset._signature_genome_build = signature_params.get(
            "genome_build"
        )

        dataset.dataset_directory = str(directory)
        return dataset

    @property
    def mutation_db(self):
        """Mutation database (lazy loaded)."""
        if self._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded. "
                "Call generate_mutation_db() or load_dataset() first."
            )
        return self._mutation_db

    @mutation_db.setter
    def mutation_db(self, value):
        """Set mutation database."""
        self._mutation_db = value

    @property
    def n_samples(self):
        """Number of tumor samples in the dataset."""
        if self._genes_present is not None:
            return self._genes_present.shape[1]
        # Fallback: compute from mutation_db
        if self._mutation_db is None:
            raise ValueError(
                "Cannot compute n_samples: mutation database not "
                "loaded. Call generate_mutation_db() or load_dataset() first."
            )
        return self._mutation_db["Tumor_Sample_Barcode"].nunique()

    @property
    def n_genes(self):
        """Number of genes in the dataset."""
        if self._genes_present is not None:
            return self._genes_present.shape[0]
        # Fallback: compute from mutation_db
        if self._mutation_db is None:
            raise ValueError(
                "Cannot compute n_genes: mutation database not "
                "loaded. Call generate_mutation_db() or load_dataset() first."
            )
        return self._mutation_db["ensembl_gene_id"].nunique()

    @property
    def genes_present(self):
        """Gene presence matrix (lazy loaded)."""
        if self._genes_present is None:
            raise ValueError(
                "Gene presence matrix not computed. "
                "Call compute_gene_presence() first."
            )
        return self._genes_present

    @genes_present.setter
    def genes_present(self, value):
        """Set gene presence matrix."""
        self._genes_present = value

    @property
    def genes_present_non_silent(self):
        """Gene presence matrix for non-silent mutations (lazy loaded)."""
        if self._genes_present_non_silent is None:
            raise ValueError(
                "Non-silent gene presence matrix not computed. "
                "Call compute_gene_presence_non_silent() first."
            )
        return self._genes_present_non_silent

    @genes_present_non_silent.setter
    def genes_present_non_silent(self, value):
        """Set non-silent gene presence matrix."""
        self._genes_present_non_silent = value

    @property
    def genes_present_silent(self):
        """Gene presence matrix for silent mutations (lazy loaded)."""
        if self._genes_present_silent is None:
            raise ValueError(
                "Silent gene presence matrix not computed. "
                "Call compute_gene_presence_silent() first."
            )
        return self._genes_present_silent

    @genes_present_silent.setter
    def genes_present_silent(self, value):
        """Set silent gene presence matrix."""
        self._genes_present_silent = value

    @property
    def genes_counts_silent(self):
        """Silent mutation counts per gene per tumor (lazy loaded)."""
        if self._genes_counts_silent is None:
            raise ValueError(
                "Silent gene count matrix not computed. "
                "Call compute_gene_counts_channels() first."
            )
        return self._genes_counts_silent

    @genes_counts_silent.setter
    def genes_counts_silent(self, value):
        """Set silent gene count matrix."""
        self._genes_counts_silent = value

    @property
    def genes_counts_non_silent(self):
        """Non-silent mutation counts per gene per tumor (lazy)."""
        if self._genes_counts_non_silent is None:
            raise ValueError(
                "Non-silent gene count matrix not computed. "
                "Call compute_gene_counts_channels() first."
            )
        return self._genes_counts_non_silent

    @genes_counts_non_silent.setter
    def genes_counts_non_silent(self, value):
        """Set non-silent gene count matrix."""
        self._genes_counts_non_silent = value

    @property
    def variant_db(self):
        """Variant database (lazy loaded)."""
        if self._variant_db is None:
            logger.warning(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first."
            )
            raise ValueError(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first."
            )
        return self._variant_db

    @variant_db.setter
    def variant_db(self, value):
        """Set variant database."""
        self._variant_db = value

    @property
    def variants_present(self):
        """Variant presence matrix (lazy loaded)."""
        if self._variants_present is None:
            raise ValueError(
                "Variant presence matrix not computed. "
                "Call compute_variants_present() first."
            )
        return self._variants_present

    @variants_present.setter
    def variants_present(self, value):
        """Set variant presence matrix."""
        self._variants_present = value

    @property
    def sample_qc_flags(self):
        """Per-sample QC flag (lazy loaded), e.g. from
        :func:`sample_qc.combine_sample_flags`.

        Unlike the other lazy attributes here, this one is not
        computed by any method on this class -- it is TCGA-specific
        (purity/VAF-shape evidence, sourced by the caller from
        outside `sigmutsel`, see :mod:`sample_qc`) and is expected to
        be assigned directly by the pipeline that builds this
        dataset (e.g. `tcga_analysis/code/main.py`).

        Returns
        -------
        pandas.Series
            Boolean, indexed by ``Tumor_Sample_Barcode``. ``True``
            means flagged (drop or downweight).
        """
        if self._sample_qc_flags is None:
            raise ValueError(
                "sample_qc_flags not set. Assign "
                "dataset.sample_qc_flags = combine_sample_flags(...) "
                "first."
            )
        return self._sample_qc_flags["flag"]

    @sample_qc_flags.setter
    def sample_qc_flags(self, value):
        """Set per-sample QC flags. Accepts a boolean Series or a
        single-column DataFrame (the latter is what `load_dataset`
        passes in, since parquet round-trips as a DataFrame)."""
        if isinstance(value, pd.Series):
            value = value.rename("flag").to_frame()
        elif value.shape[1] != 1:
            raise ValueError(
                "sample_qc_flags must be a Series or single-column "
                f"DataFrame, got {value.shape[1]} columns."
            )
        else:
            value = value.set_axis(["flag"], axis=1)
        self._sample_qc_flags = value.astype(bool)

    @property
    def n_variants(self):
        """Number of variants in the dataset.

        If variants haven't been loaded, computes from mutation_db.
        """
        if self._variant_db is not None:
            return self._variant_db.shape[0]

        if self._mutation_db is None:
            raise ValueError(
                "Cannot compute n_variants: mutation database "
                "not loaded. Call generate_mutation_db() or "
                "load_dataset() first."
            )
        return self._mutation_db["variant"].nunique()

    @property
    def variant_counts(self):
        """Number of tumors each variant appears in.

        Returns
        -------
        pd.Series
            Variant counts sorted descending by frequency.
        """
        return (
            self.mutation_db.groupby("variant")[
                "Tumor_Sample_Barcode"
            ]
            .nunique()
            .sort_values(ascending=False)
        )

    @property
    def gene_counts(self):
        """Number of tumors each gene is mutated in.

        Returns
        -------
        pd.DataFrame
            DataFrame with gene as index, ensembl_gene_id column,
            and count column, sorted descending by count.
        """
        # Get counts per gene
        counts = (
            self.mutation_db.groupby("gene")["Tumor_Sample_Barcode"]
            .nunique()
            .rename("count")
        )

        # Get ensembl_gene_id mapping (one per gene symbol)
        gene_mapping = (
            self.mutation_db[["gene", "ensembl_gene_id"]]
            .drop_duplicates("gene")
            .set_index("gene")
        )

        # Combine and sort
        result = counts.to_frame().join(gene_mapping)
        return result.sort_values("count", ascending=False)

    @property
    def variant_type_counts(self):
        """Number of different types each variant has.

        Returns
        -------
        pd.Series
            Variant type counts sorted descending.
        """
        return (
            self.mutation_db.groupby("variant")["type"]
            .nunique()
            .sort_values(ascending=False)
        )

    @property
    def variant_summary(self):
        """Summary of variants by number of types and tumors.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns num_types and num_tumors,
            indexed by variant, sorted by types then tumors.
        """
        return (
            self.mutation_db.groupby("variant")
            .agg(
                num_types=("type", "nunique"),
                num_tumors=("Tumor_Sample_Barcode", "nunique"),
            )
            .reset_index()
            .sort_values(
                by=["num_types", "num_tumors"],
                ascending=[False, False],
            )
            .set_index("variant")
        )

    def has_mutation_db(self):
        """Check if mutation database has been loaded."""
        return self._mutation_db is not None

    def generate_mutation_db(self, location_gene_set=None, **kwargs):
        """Generate mutation database from MAF files.

        This method wraps :func:`load_maf_files.generate_compact_db`
        and stores the result in the dataset's _mutation_db attribute.

        For ID signature class, automatically sets seqinfo_dir to
        `{location_maf_files}/output/vcf_files/ID/` if not
        explicitly provided.

        Parameters
        ----------
        location_gene_set : str, Path, or None, default None
            Path to gene set file for gene name standardization.
            If None (default), uses HGNC complete set from
            locations.py for automatic gene name updates.
            Set to a custom path to use a different gene set.
        **kwargs : dict
            Additional arguments passed to
            :func:`load_maf_files.generate_compact_db`.

            **For ID signature class:**
            - seqinfo_dir : str or Path, optional
                Directory containing *_seqinfo.txt files from
                SigProfilerMatrixGenerator. These files provide
                COSMIC ID-83 mutation type annotations for indels.
                If not provided, automatically uses
                `{location_maf_files}/output/vcf_files/ID/`.

        Notes
        -----
        The signature class used is determined by the dataset's
        signature_class attribute, which must be one of the
        COSMIC signature classes: SBS, DBS, ID, CN, SV, RNA-SBS.

        For ID (indel) signature class, seqinfo files are
        automatically located at the standard output location
        from SigProfilerMatrixGenerator. You only need to provide
        seqinfo_dir explicitly if the files are in a custom
        location.

        Examples
        --------
        >>> # Generate and save SBS database
        >>> dataset = MutationDataset(
        ...     location_maf_files="data/maf_files",
        ...     signature_class="SBS")
        >>> dataset.generate_mutation_db()
        >>>
        >>> # Generate without saving (load into memory only)
        >>> dataset.generate_mutation_db()
        >>>
        >>> # Generate ID database (seqinfo_dir auto-detected)
        >>> dataset_id = MutationDataset(
        ...     location_maf_files="data/maf_files",
        ...     signature_class="ID")
        >>> dataset_id.generate_mutation_db()
        >>> # seqinfo_dir automatically set to:
        >>> # data/maf_files/output/vcf_files/ID/

        See Also
        --------
        load_maf_files.generate_compact_db : Core generation function
        """
        from pathlib import Path

        from .load_maf_files import generate_compact_db

        # Auto-set seqinfo_dir for ID signature class if not provided
        if (
            self.signature_class == "ID"
            and "seqinfo_dir" not in kwargs
        ):
            kwargs["seqinfo_dir"] = (
                Path(self.location_maf_files)
                / "output"
                / "vcf_files"
                / "ID"
            )

        # Use generate_compact_db and store result
        self._mutation_db = generate_compact_db(
            self.location_maf_files,
            signature_class=self.signature_class,
            location_gene_set=location_gene_set,
            **kwargs,
        )

    def has_gene_presence(self):
        """Check if gene presence matrix has been computed."""
        return self._genes_present is not None

    def has_non_silent_presence(self):
        """Check if non-silent gene presence has been computed."""
        return self._genes_present_non_silent is not None

    def has_silent_presence(self):
        """Check if silent gene presence has been computed."""
        return self._genes_present_silent is not None

    def has_channel_counts(self):
        """Check if both channels' count matrices exist."""
        return (
            self._genes_counts_silent is not None
            and self._genes_counts_non_silent is not None
        )

    def has_variants(self):
        """Check if variants have been loaded."""
        return self._variant_db is not None

    def generate_variant_db(self, position_tolerance=3):
        """Generate variant database from the mutation database.

        Parameters
        ----------
        position_tolerance : int, default 3
            Maximum positional deviation (bp) allowed when collapsing
            variants across tumors (see extract_variants_from_db()).

        Returns
        -------
        pd.DataFrame
            Variant annotation table with mutation types.
        """
        from .variant_annotation import (
            annotate_variants_with_types,
            extract_variants_from_db,
        )

        if self._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded. "
                "Call generate_mutation_db() or load_dataset() first."
            )

        variants = extract_variants_from_db(
            self.mutation_db, position_tolerance=position_tolerance
        )
        variants = annotate_variants_with_types(
            variants, self.mutation_db
        )

        self._variant_db = variants
        return variants

    def compute_gene_presence(self):
        """Compute gene presence matrix from mutation database.

        Calls compute_genes_present() to create a binary matrix
        indicating which genes are mutated in which samples.
        """
        from .estimate_presence import compute_genes_present

        self._genes_present = compute_genes_present(self.mutation_db)

    def compute_gene_presence_non_silent(self):
        """Compute non-silent gene presence matrix.

        Calls compute_genes_present() with scope='non-silent'
        to create a binary matrix for non-silent mutations only.
        """
        from .estimate_presence import compute_genes_present

        self._genes_present_non_silent = compute_genes_present(
            self.mutation_db, scope="non-silent"
        )

    def compute_gene_presence_silent(self):
        """Compute silent (synonymous) gene presence matrix.

        Calls compute_genes_present() with scope='silent' to create a
        binary matrix for silent mutations only -- the mirror of
        :meth:`compute_gene_presence_non_silent`, and the channel
        that stays selection-free inside driver genes.
        """
        from .estimate_presence import compute_genes_present

        self._genes_present_silent = compute_genes_present(
            self.mutation_db, scope="silent"
        )

    def compute_gene_counts_channels(self):
        """Compute per-channel mutation *count* matrices.

        The uncensored counterparts of
        :meth:`compute_gene_presence_silent` and
        :meth:`compute_gene_presence_non_silent`, stored in
        ``genes_counts_silent`` / ``genes_counts_non_silent``. Both
        are built in one call because the count likelihood always
        needs the pair, and computing one without the other invites
        scoring a model against a channel it was not fit on.

        Nothing new is extracted from the MAFs -- these are the same
        mutations ``compute_genes_present`` groups, just not
        collapsed to 0/1.
        """
        from .estimate_presence import compute_genes_counts

        self._genes_counts_silent = compute_genes_counts(
            self.mutation_db, scope="silent"
        )
        self._genes_counts_non_silent = compute_genes_counts(
            self.mutation_db, scope="non-silent"
        )
        return (
            self._genes_counts_silent,
            self._genes_counts_non_silent,
        )

    def compute_variants_present(self):
        """Compute variant presence matrix.

        Requires variant_db to be loaded first (via
        generate_variant_db() or load_dataset()).
        """
        from .estimate_presence import compute_variants_present

        if self._variant_db is None:
            raise ValueError(
                "Variants database not loaded. "
                "Call generate_variant_db() or load_dataset() first."
            )

        self._variants_present = compute_variants_present(
            self.mutation_db, self._variant_db
        )

    @property
    def sig_assignments(self):
        """Signature assignments (lazy loaded)."""
        if self._sig_assignments is None:
            raise ValueError(
                "Signature assignments not loaded. "
                "Call run_signature_decomposition() first."
            )
        return self._sig_assignments

    @sig_assignments.setter
    def sig_assignments(self, value):
        """Set signature assignments."""
        self._sig_assignments = value

    @property
    def signature_matrix(self):
        """Normalized signature matrix (lazy loaded)."""
        if self._signature_matrix is None:
            raise ValueError(
                "Signature matrix not loaded. "
                "Call run_signature_decomposition() first."
            )
        return self._signature_matrix

    @signature_matrix.setter
    def signature_matrix(self, value):
        """Set normalized signature matrix."""
        self._signature_matrix = value

    @property
    def signature_reference_genome(self):
        """Reference genome used for mutational matrices (equals genome_build)."""
        if self._signature_reference_genome is None:
            raise ValueError(
                "Reference genome not recorded. "
                "Call run_signature_decomposition() first."
            )
        return self._signature_reference_genome

    @property
    def signature_exome(self):
        """Whether exome-normalized signatures were used."""
        if self._signature_exome is None:
            raise ValueError(
                "Exome flag not recorded. "
                "Call run_signature_decomposition() first."
            )
        return self._signature_exome

    @property
    def signature_cosmic_version(self):
        """COSMIC version used for signature decomposition."""
        if self._signature_cosmic_version is None:
            raise ValueError(
                "COSMIC version not recorded. "
                "Call run_signature_decomposition() first."
            )
        return self._signature_cosmic_version

    @property
    def signature_genome_build(self):
        """Genome build used for signature decomposition."""
        if self._signature_genome_build is None:
            raise ValueError(
                "Genome build not recorded. "
                "Call run_signature_decomposition() first."
            )
        return self._signature_genome_build

    @property
    def contexts_by_gene(self):
        """Trinucleotide context counts by gene (lazy loaded)."""
        if self._contexts_by_gene is None:
            raise ValueError(
                "Contexts by gene not loaded. "
                "Call generate_contexts_by_gene() or load_dataset() first."
            )
        return self._contexts_by_gene

    @contexts_by_gene.setter
    def contexts_by_gene(self, value):
        """Set trinucleotide context counts by gene."""
        self._contexts_by_gene = value

    @property
    def contexts_by_gene_syn(self):
        """Synonymous SBS opportunity counts by gene (lazy loaded)."""
        if self._contexts_by_gene_syn is None:
            raise ValueError(
                "Synonymous opportunity counts not loaded. Call "
                "generate_consequence_contexts_by_gene() or "
                "load_dataset() first."
            )
        return self._contexts_by_gene_syn

    @contexts_by_gene_syn.setter
    def contexts_by_gene_syn(self, value):
        """Set synonymous SBS opportunity counts by gene."""
        self._contexts_by_gene_syn = value

    @property
    def contexts_by_gene_nonsyn(self):
        """Non-synonymous SBS opportunity counts by gene (lazy)."""
        if self._contexts_by_gene_nonsyn is None:
            raise ValueError(
                "Non-synonymous opportunity counts not loaded. Call "
                "generate_consequence_contexts_by_gene() or "
                "load_dataset() first."
            )
        return self._contexts_by_gene_nonsyn

    @contexts_by_gene_nonsyn.setter
    def contexts_by_gene_nonsyn(self, value):
        """Set non-synonymous SBS opportunity counts by gene."""
        self._contexts_by_gene_nonsyn = value

    def has_consequence_contexts_by_gene(self):
        """Whether the consequence-split opportunity tables exist."""
        return (
            self._contexts_by_gene_syn is not None
            and self._contexts_by_gene_nonsyn is not None
        )

    def has_mutational_matrices(self):
        """Check if mutational matrices have been generated.

        Checks for the existence of the output directory created by
        SigProfilerMatrixGenerator at
        `location_maf_files/output/`.

        Returns
        -------
        bool
            True if mutational matrices exist, False otherwise.

        Examples
        --------
        >>> dataset = MutationDataset(
        ...     location_maf_files="data/maf_files",
        ...     signature_class="SBS")
        >>> if not dataset.has_mutational_matrices():
        ...     dataset.generate_mutational_matrices()
        """
        from pathlib import Path

        output_dir = Path(self.location_maf_files) / "output"
        return output_dir.exists()

    def generate_mutational_matrices(
        self,
        reference_genome="GRCh38",
        force_generation=False,
        **kwargs,
    ):
        """Generate mutational matrices using SigProfilerMatrixGenerator.

        Runs
        :func:`mutational_matrices_generator.mutational_matrices_generation`
        on the dataset's MAF files. Matrices are saved to
        `location_maf_files/output/`.

        Parameters
        ----------
        reference_genome : str, optional
            Reference genome assembly. Default is 'GRCh38'.
        force_generation : bool, optional
            If True, deletes existing output directory and
            regenerates all matrices. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`mutational_matrices_generator.mutational_matrices_generation`.
            Common options:
            - exome : bool (default True)
            - seqInfo : bool (default True, required for ID)
            - plot : bool (default False)

        Returns
        -------
        Path
            Path to the output directory containing generated
            matrices.

        Notes
        -----
        **IMPORTANT for ID signature class**: The seqInfo parameter
        defaults to True, which is required for proper ID mutation
        type annotation. Do not set it to False when
        signature_class='ID'.

        Examples
        --------
        >>> # Generate matrices for SBS
        >>> dataset = MutationDataset(
        ...     location_maf_files="data/maf_files",
        ...     signature_class="SBS")
        >>> output_path = dataset.generate_mutational_matrices()
        >>>
        >>> # Generate with custom parameters
        >>> output_path = dataset.generate_mutational_matrices(
        ...     reference_genome='GRCh37',
        ...     exome=True,
        ...     plot=True)

        See Also
        --------
        mutational_matrices_generator.mutational_matrices_generation
        has_mutational_matrices : Check if matrices exist
        run_signature_decomposition : Decompose signatures

        """
        from .mutational_matrices_generator import (
            mutational_matrices_generation,
        )

        return mutational_matrices_generation(
            path_to_input_files=self.location_maf_files,
            reference_genome=reference_genome,
            force_generation=force_generation,
            **kwargs,
        )

    def run_signature_decomposition(
        self,
        force_generation=False,
        exome=None,
        cosmic_version=None,
        genome_build="GRCh38",
        **kwargs,
    ):
        """Run COSMIC signature decomposition on mutational matrices.

        Runs signature decomposition using the appropriate mutational
        matrix based on the dataset's signature_class. If the required
        mutational matrices are missing, they will be generated first
        using :func:`generate_mutational_matrices`, passing through the
        reference genome and exome flag specified here.

        Results are automatically saved to:
        `{location_maf_files}/signature_decomposition/{signature_class}/`

        Parameters
        ----------
        exome : bool, optional
            Whether exome-normalized signatures are expected. When
            None (default), falls back to values provided via ``kwargs``
            or True. Passed both to mutational matrix generation (if
            needed) and to the signature decomposition step.
        cosmic_version : float, optional
            COSMIC signature version to use. Default is 3.4 unless
            overridden via ``kwargs``. Stored for future reference.
        genome_build : str, optional
            Genome build used both for mutational matrix generation
            (as the reference genome) and for the signature
            decomposition step. Default is 'GRCh38' unless overridden
            via ``kwargs``.
        force_generation : bool, optional
            If True, deletes existing results and re-runs
            decomposition. Default is False.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`signature_decomposition.signature_decomposition`.
            Common options:
            - exome : bool (default matches matrix)
            - cosmic_version : float (default 3.4)
            - genome_build : str (default 'GRCh38')
            - exclude_signature_subgroups : list or tuple
            - include_signature_subgroups : list or tuple
        For backward compatibility, `exome`, `cosmic_version`, and
        `genome_build` can also be provided via ``kwargs`` when the
        dedicated parameters are left as None.

        Returns
        -------
        pd.DataFrame
            Signature assignment matrix with samples as index and
            signatures as columns. Values are the number of
            mutations attributed to each signature in each sample.

        Raises
        ------
        FileNotFoundError
            If mutational matrices cannot be located even after
            attempting to generate them automatically. In that case
            re-run generate_mutational_matrices() manually to inspect
            the failure.

        Notes
        -----
        The function automatically:
        - Selects the appropriate matrix file based on
          signature_class (SBS96, DBS78, or ID83)
        - Sets input_type='matrix'
        - Sets collapse_to_SBS96=False for DBS and ID signature
          classes

        Standard matrix resolutions used:
        - SBS: SBS96 (96 trinucleotide contexts)
        - DBS: DBS78 (78 doublet base substitution contexts)
        - ID: ID83 (83 indel contexts)

        The results are stored in the dataset's _sig_assignments
        attribute and can be accessed via the sig_assignments
        property. The genome build (used both as reference genome and
        for COSMIC signatures), exome flag, and COSMIC version are
        also recorded for future reference.

        Examples
        --------
        >>> dataset = MutationDataset(
        ...     location_maf_files="data/maf_files",
        ...     signature_class="SBS")
        >>>
        >>> # Run signature decomposition (generates matrices if needed)
        >>> assignments = dataset.run_signature_decomposition(
        ...     exome=True,
        ...     cosmic_version=3.4)
        >>>
        >>> # Access results and normalized signature matrix
        >>> print(dataset.sig_assignments.head())
        >>> print(dataset.signature_matrix.head())

        See Also
        --------
        signature_decomposition.signature_decomposition
        generate_mutational_matrices : Generate matrices first
        has_mutational_matrices : Check if matrices exist
        """
        title = "Signature decomposition"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        import logging
        from pathlib import Path

        from .signature_decomposition import (
            signature_decomposition as run_sig_decomp,
        )

        logger = logging.getLogger(__name__)

        if exome is None:
            exome = kwargs.pop("exome", True)
        else:
            kwargs.pop("exome", None)

        if cosmic_version is None:
            cosmic_version = kwargs.pop("cosmic_version", 3.4)
        else:
            kwargs.pop("cosmic_version", None)

        if genome_build is None:
            genome_build = kwargs.pop("genome_build", "GRCh38")
        else:
            kwargs.pop("genome_build", None)

        if not self.has_mutational_matrices():
            logger.info(
                "Mutational matrices not found. "
                "Generating them before signature decomposition..."
            )
            matrix_kwargs = {}
            if exome is not None:
                matrix_kwargs["exome"] = exome
            self.generate_mutational_matrices(
                reference_genome=genome_build,
                force_generation=False,
                **matrix_kwargs,
            )
            logger.info("...done.")
            print()

        if not self.has_mutational_matrices():
            raise FileNotFoundError(
                f"Mutational matrices not found at "
                f"{self.location_maf_files}/output/. "
                "Automatic generation failed. "
                "Run generate_mutational_matrices() manually for "
                "more details."
            )

        # Determine which matrix to use based on signature_class
        matrix_map = {
            "SBS": "SBS96",
            "DBS": "DBS78",
            "ID": "ID83",
        }

        # Provide helpful error for common mistakes
        if self.signature_class not in matrix_map:
            # Suggest correction for old aliases
            if self.signature_class == "SNP":
                raise ValueError(
                    "signature_class='SNP' is not supported. "
                    "Use signature_class='SBS' instead. "
                    "Create a new dataset with: "
                    "MutationDataset(location_maf_files=..., "
                    "signature_class='SBS')"
                )
            elif self.signature_class == "INDEL":
                raise ValueError(
                    "signature_class='INDEL' is not supported. "
                    "Use signature_class='ID' instead. "
                    "Create a new dataset with: "
                    "MutationDataset(location_maf_files=..., "
                    "signature_class='ID')"
                )
            else:
                raise ValueError(
                    f"Signature decomposition not supported for "
                    f"signature_class='{self.signature_class}'. "
                    f"Supported classes: {list(matrix_map.keys())}"
                )

        matrix_resolution = matrix_map[self.signature_class]
        matrix_filename = (
            f"mutational_matrix.{matrix_resolution}.exome"
        )

        # Build path to matrix file
        output_dir = Path(self.location_maf_files) / "output"
        matrix_dir = output_dir / self.signature_class
        matrix_path = matrix_dir / matrix_filename

        # Check that the specific matrix file exists
        if not matrix_path.exists():
            raise FileNotFoundError(
                f"Matrix file not found at {matrix_path}. "
                f"Expected {matrix_resolution} matrix for "
                f"{self.signature_class} signature class. "
                "Run generate_mutational_matrices() first."
            )

        # Create results directory in standard location
        sig_decomp_dir = (
            Path(self.location_maf_files) / "signature_decomposition"
        )
        sig_decomp_dir.mkdir(parents=True, exist_ok=True)

        results_dir = sig_decomp_dir / self.signature_class

        # Check if results already exist
        solution_dir = results_dir / "Assignment_Solution"
        results_exist = solution_dir.exists() and not force_generation

        if results_exist:
            logger.info(
                f"Signature decomposition for {self.signature_class} "
                f"was previously run and will be loaded from "
                f"{results_dir}. To re-run decomposition, use "
                f"force_generation=True."
            )
        else:
            logger.info(
                f"Running signature decomposition for "
                f"{self.signature_class} using matrix: {matrix_path}"
            )

        # Set collapse_to_SBS96=False for ID and DBS
        collapse_to_SBS96 = (
            self.signature_class == "SBS"
            if "collapse_to_SBS96" not in kwargs
            else kwargs.pop("collapse_to_SBS96")
        )

        # Run signature decomposition
        self._sig_assignments = run_sig_decomp(
            results_dir=str(results_dir),
            input_data=str(matrix_path),
            input_type="matrix",
            collapse_to_SBS96=collapse_to_SBS96,
            force_generation=force_generation,
            exome=exome,
            cosmic_version=cosmic_version,
            genome_build=genome_build,
            **kwargs,
        )

        if not results_exist:
            logger.info("... done with signature decomposition.")
            print()

        # Load the normalized signature matrix
        sig_matrix_path = (
            solution_dir
            / "Signatures"
            / "Assignment_Solution_Signatures.txt"
        )

        if sig_matrix_path.exists():
            self._signature_matrix = pd.read_csv(
                sig_matrix_path, sep="\t", index_col=0
            )
            logger.info(
                f"Loaded normalized signature matrix from "
                f"{sig_matrix_path}"
            )
            print()
        else:
            logger.warning(
                f"Signature matrix not found at {sig_matrix_path}"
            )
            self._signature_matrix = None

        self._signature_reference_genome = genome_build
        self._signature_exome = exome
        self._signature_cosmic_version = cosmic_version
        self._signature_genome_build = genome_build

        print()
        return self._sig_assignments

    def run_two_pass_signature_decomposition(
        self,
        artifact_threshold=0.5,
        force_generation=False,
        exome=None,
        cosmic_version=None,
        genome_build="GRCh38",
        treatment_load_threshold=None,
        prevalence_override_min_fraction=None,
        min_burden_for_diagnostics=100,
        **kwargs,
    ):
        """Two-pass, diagnostic-aware signature decomposition.

        Fits signature decomposition twice rather than once. Pass A is
        a fully unrestricted diagnostic fit -- no per-cancer-type
        table, no `treatment_naive`, artifact signatures included --
        used to compute up to three independent signals before pass B:

        1. **Artifact-mutation detection** (always on): each
           mutation's probability mass on
           `constants.ARTIFACT_SIGNATURES`
           (`signature_attribution.compute_signature_probability_mass`),
           used to drop (:func:`qc.flag_artifact_signature_mutations`)
           mutations dominantly attributed to a technical artifact --
           without simply excluding artifact signatures from the fit,
           which would just force their mutations onto whichever real
           signature fits next-best (see this project's plan for the
           full rationale).
        2. **Treatment-signature-load sample QC** (opt-in via
           `treatment_load_threshold`): TCGA's "treatment-naive"
           clinical annotation is retrospective self-report and can
           miss real cases. A sample whose *unrestricted* fit
           attributes an implausible fraction of its burden to
           `constants.TREATMENT_ASSOCIATED_SIGNATURES` is dropped
           entirely (:func:`qc.flag_treatment_signature_samples`) --
           sample-level, not mutation-level, because real treatment
           mutagenesis reshapes a tumor's whole subsequent
           evolutionary history, unlike a sequencing artifact. Gated
           by `min_burden_for_diagnostics`: a naive threshold with no
           burden gate is dominated by low-count NNLS noise, not real
           contamination (see this project's plan and TODO.md's
           2026-08-25 groundwork).
        3. **Prevalence-based per-cancer-type table override** (opt-in
           via `prevalence_override_min_fraction`): a signature the
           table excludes for this cancer type but that recurs across
           enough of the cohort's own adequately-powered tumors is
           kept in pass B's basis instead
           (`signature_decomposition.compute_prevalence_overrides`) --
           a prior override, not a table replacement, and it never
           touches the treatment or artifact lists (see that
           function's docstring for the full rationale). Computed
           *after* step 2 removes treatment-flagged samples, so a
           contaminated tumor about to be dropped can't inflate the
           apparent prevalence of whatever signature it leaks into.

        Then:

        - Samples flagged by step 2 and mutations flagged by step 1
          are dropped from `self.mutation_db` -- guaranteeing they're
          excluded from burden, gene presence, and every other
          downstream calculation that reads `mutation_db`, not just
          from pass B's fit.
        - **Pass B**: rebuild the SBS96 matrix directly from the
          cleaned `mutation_db` (not the original MAF files -- single
          source of truth with what `mutation_db` now holds) and refit
          with the final exclusion set (per-cancer-type table, net of
          any step-3 overrides, unioned with the treatment list if
          `treatment_naive` and always with `ARTIFACT_SIGNATURES`).
          This becomes the "official" result: ``self.sig_assignments``
          / ``self.signature_matrix`` are overwritten with pass B's,
          and ``self.mutation_db`` is overwritten with the cleaned
          version, once this method returns.

        Requires ``self.mutation_db`` to already be built (e.g.
        ``self.generate_mutation_db(qc_mode=True, ...)``) *before*
        calling this -- pass A's diagnostics need it, and it can't be
        built afterward without regenerating over the cleaned data.
        When wiring this into :meth:`build_full_dataset`, pass
        ``regenerate_mutation_db=False`` there, since this method
        already updated ``mutation_db`` and a default rebuild would
        silently overwrite the cleaned version with the original.

        **Caching pitfall, found the hard way (2026-08-26): pass A and
        pass B cache in *two separate directories*, and
        `force_generation=False`'s cache check is existence-only, with
        no awareness of `cosmic_version`/exclusion lists/thresholds
        changing between calls.** Pass A caches under the standard
        ``{location_maf_files}/signature_decomposition/{signature_class}/``
        directory (shared with any plain :meth:`run_signature_decomposition`
        call); pass B caches separately, under a
        ``..._{signature_class}_artifact_cleaned`` directory. Clearing
        or backing up only one of the two before a rerun with
        different parameters (e.g. a COSMIC version bump) lets the
        *other* one silently serve stale results with no error and no
        warning -- caught only because pass B's stale cache reflected
        a different set of cleaned tumors than pass A's fresh one,
        producing a one-tumor mismatch between ``mutation_db`` and
        ``sig_assignments`` that surfaced as an all-NaN row deep in
        downstream covariate-effect estimation, far from the actual
        cause. Any rerun with different fit parameters must clear (or
        back up) **both** directories, not just one.

        Only supports ``signature_class="SBS"`` (the matrix rebuild
        uses `constants.canonical_types_order`, which is SBS96-specific).

        Parameters
        ----------
        artifact_threshold : float, default 0.5
            Forwarded to `qc.flag_artifact_signature_mutations`. Not
            derived from theory -- tune against the real distribution
            observed on a few pilot cohorts (see this project's plan).
        treatment_load_threshold : float, optional
            Forwarded to `qc.flag_treatment_signature_samples` as
            `threshold`. `None` (default) disables step 2 entirely --
            no sample is dropped for treatment-signature load.
        prevalence_override_min_fraction : float, optional
            Forwarded to `signature_decomposition.compute_prevalence_overrides`
            as `min_prevalence`. `None` (default) disables step 3
            entirely -- the per-cancer-type table is used as-is.
        min_burden_for_diagnostics : int, default 100
            Minimum fitted mutation count (pass A's per-sample
            assignment total) for a sample to be eligible for either
            step 2 or step 3 -- both are per-sample/per-cohort
            statistics that are too noisy to trust below this count
            (see this project's plan). Samples below this count are
            never flagged by step 2 and never contribute to step 3's
            prevalence counts.
        force_generation, exome, cosmic_version, genome_build :
            Forwarded to both passes (see :meth:`run_signature_decomposition`
            for defaults).
        **kwargs : dict
            Forwarded to pass A unchanged, *except*
            `exclude_signature_subgroups` and `treatment_naive`, which
            this method consumes itself (pass A must not receive
            either -- it has to stay fully unrestricted to serve as
            steps 2 and 3's diagnostic fit) and re-applies to pass B
            after resolving step 3's overrides.
            `exclude_signature_subgroups` may be a cancer-type
            shorthand, a `(location, cancer_type)` tuple, or an
            explicit signature list (e.g. main.py's missing-PCAWG-row
            fallback); step 3 only has a table to override against for
            the first two forms. `exclude_artifacts` is set explicitly
            by this method for each pass and must not be passed here.

        Returns
        -------
        pd.DataFrame
            Pass B's signature assignments (also stored in
            ``self.sig_assignments``).
        """
        if self.signature_class != "SBS":
            raise NotImplementedError(
                "run_two_pass_signature_decomposition only supports "
                "signature_class='SBS' (the pass-B matrix rebuild "
                "uses constants.canonical_types_order, which is "
                "SBS96-specific)."
            )
        if not self.has_mutation_db():
            raise ValueError(
                "mutation_db must be built first (e.g. "
                "generate_mutation_db(qc_mode=True, ...)) -- pass A's "
                "diagnostics need it."
            )
        if "exclude_artifacts" in kwargs:
            raise TypeError(
                "exclude_artifacts is set explicitly per pass by "
                "this method and must not be passed to "
                "run_two_pass_signature_decomposition."
            )

        from .constants import (
            ARTIFACT_SIGNATURES,
            TREATMENT_ASSOCIATED_SIGNATURES,
        )
        from .qc import (
            flag_artifact_signature_mutations,
            flag_treatment_signature_samples,
        )
        from .signature_attribution import (
            compute_signature_probability_mass,
        )
        from .signature_decomposition import (
            build_sbs96_matrix_from_mutation_db,
            compute_prevalence_overrides,
            resolve_exclusion_list,
        )
        from .signature_decomposition import (
            signature_decomposition as run_sig_decomp,
        )

        # Consumed here, not forwarded to pass A -- pass A must be
        # fully unrestricted to serve as the diagnostic fit for steps
        # 2 and 3 below, which is a real change from a design where
        # pass A already excluded the table and treatment signatures.
        raw_exclude = kwargs.pop("exclude_signature_subgroups", None)
        treatment_naive = kwargs.pop("treatment_naive", True)

        title = "Two-pass signature decomposition: pass A (unrestricted diagnostic fit)"
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        self.run_signature_decomposition(
            force_generation=force_generation,
            exome=exome,
            cosmic_version=cosmic_version,
            genome_build=genome_build,
            exclude_artifacts=False,
            **kwargs,
        )
        pass_a_assignments = self._sig_assignments
        pass_a_signature_matrix = self._signature_matrix
        if pass_a_signature_matrix is None:
            raise RuntimeError(
                "Pass A's normalized signature matrix wasn't loaded "
                "-- can't compute per-mutation artifact probabilities."
            )
        print()

        pass_a_totals = pass_a_assignments.sum(axis=1)

        # --- Step 2: treatment-signature-load sample QC (opt-in) ---
        working_db = self.mutation_db
        if treatment_load_threshold is not None:
            treatment_cols = [
                c
                for c in TREATMENT_ASSOCIATED_SIGNATURES
                if c in pass_a_assignments.columns
            ]
            treatment_totals = (
                pass_a_assignments[treatment_cols].sum(axis=1)
                if treatment_cols
                else pd.Series(0.0, index=pass_a_assignments.index)
            )
            treatment_load = treatment_totals / pass_a_totals.replace(
                0, pd.NA
            )
            treatment_load = treatment_load.astype(float)
            # Below the burden gate, an unrestricted fit's fraction is
            # too noisy to trust as a QC signal -- never flag these
            # (NaN > threshold is False, so this is sufficient).
            treatment_load[
                pass_a_totals < min_burden_for_diagnostics
            ] = float("nan")

            tagged_samples_db = flag_treatment_signature_samples(
                working_db,
                treatment_load,
                threshold=treatment_load_threshold,
            )
            flagged_mask = (
                tagged_samples_db["problem"]
                == "treatment_signature_sample"
            )
            n_samples_flagged = tagged_samples_db.loc[
                flagged_mask, "Tumor_Sample_Barcode"
            ].nunique()
            logger.info(
                f"Pass A: {n_samples_flagged} sample(s) flagged as "
                "treatment-signature-contaminated "
                f"(threshold={treatment_load_threshold}, "
                f"min_burden={min_burden_for_diagnostics}) and "
                "dropped entirely."
            )
            working_db = tagged_samples_db[
                tagged_samples_db["problem"].isna()
            ].drop(columns="problem")

        surviving_samples = set(
            working_db["Tumor_Sample_Barcode"].unique()
        )
        diagnostic_pool = pass_a_assignments.loc[
            pass_a_assignments.index.isin(surviving_samples)
            & (pass_a_totals >= min_burden_for_diagnostics)
        ]

        # --- Resolve pass B's exclusion set, always explicit -------
        # Built by hand rather than forwarded as a cancer-type
        # shorthand, so `exclude_artifacts` is never silently a no-op
        # (it only has an effect when exclude_signature_subgroups
        # resolves from a string/tuple, not when it's already a list
        # -- true for the missing-PCAWG-row fallback case below, so
        # this method never relies on that parameter at all).
        location = None
        cancer_type = None
        if isinstance(raw_exclude, str):
            cancer_type = raw_exclude
        elif isinstance(raw_exclude, tuple) and len(raw_exclude) == 2:
            location, cancer_type = raw_exclude

        if cancer_type is not None:
            base_table_exclusion = set(
                resolve_exclusion_list(
                    cancer_type,
                    location=location,
                    treatment_naive=False,
                    exclude_artifacts=False,
                )
            )
        elif raw_exclude is None:
            base_table_exclusion = set()
        else:
            # Already an explicit list.
            base_table_exclusion = (
                set(raw_exclude)
                - set(TREATMENT_ASSOCIATED_SIGNATURES)
                - set(ARTIFACT_SIGNATURES)
            )

        # --- Step 3: prevalence-based table override (opt-in) ------
        overrides = set()
        if (
            prevalence_override_min_fraction is not None
            and base_table_exclusion
        ):
            overrides = compute_prevalence_overrides(
                diagnostic_pool,
                base_table_exclusion,
                min_prevalence=prevalence_override_min_fraction,
            )
            if overrides:
                logger.info(
                    f"Pass A: prevalence override -- {sorted(overrides)} "
                    f"recur in >= {100 * prevalence_override_min_fraction:.0f}% "
                    f"of {len(diagnostic_pool)} adequately-powered "
                    "tumors despite the per-cancer-type table "
                    "excluding them; kept in pass B's basis."
                )

        final_exclusion = base_table_exclusion - overrides
        if treatment_naive:
            final_exclusion |= set(TREATMENT_ASSOCIATED_SIGNATURES)
        final_exclusion |= set(ARTIFACT_SIGNATURES)

        logger.info(
            "Computing per-mutation artifact-signature probability "
            "mass from pass A's fit..."
        )
        artifact_mass = compute_signature_probability_mass(
            working_db,
            pass_a_assignments,
            pass_a_signature_matrix,
            target_signatures=ARTIFACT_SIGNATURES,
        )
        artifact_mass = pd.Series(
            artifact_mass, index=working_db.index
        )

        tagged_db = flag_artifact_signature_mutations(
            working_db,
            artifact_mass,
            threshold=artifact_threshold,
        )
        n_flagged = (
            tagged_db["problem"] == "artifact_signature_mutation"
        ).sum()
        logger.info(
            f"Pass A: {n_flagged}/{len(tagged_db)} mutations "
            f"({100 * n_flagged / len(tagged_db):.2f}%) flagged as "
            "artifact-signature-attributed "
            f"(threshold={artifact_threshold})."
        )
        cleaned_db = tagged_db[tagged_db["problem"].isna()].drop(
            columns="problem"
        )

        # Pass B's input: rebuilt directly from the cleaned mutation
        # set, not re-derived from the original MAF files.
        matrix_dir = (
            Path(self.location_maf_files)
            / "output"
            / self.signature_class
        )
        pass_b_matrix_path = (
            matrix_dir
            / "mutational_matrix.SBS96.exome.artifact_cleaned"
        )
        build_sbs96_matrix_from_mutation_db(
            cleaned_db, pass_b_matrix_path
        )

        pass_b_results_dir = (
            Path(self.location_maf_files)
            / "signature_decomposition"
            / f"{self.signature_class}_artifact_cleaned"
        )

        title = "Two-pass signature decomposition: pass B (final fit)"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        resolved_exome = True if exome is None else exome
        resolved_cosmic_version = (
            3.4 if cosmic_version is None else cosmic_version
        )

        pass_b_assignments = run_sig_decomp(
            results_dir=str(pass_b_results_dir),
            input_data=str(pass_b_matrix_path),
            input_type="matrix",
            collapse_to_SBS96=True,
            force_generation=force_generation,
            exome=resolved_exome,
            cosmic_version=resolved_cosmic_version,
            genome_build=genome_build,
            exclude_signature_subgroups=sorted(final_exclusion),
            **kwargs,
        )

        pass_b_sig_matrix_path = (
            pass_b_results_dir
            / "Assignment_Solution"
            / "Signatures"
            / "Assignment_Solution_Signatures.txt"
        )
        if pass_b_sig_matrix_path.exists():
            pass_b_signature_matrix = pd.read_csv(
                pass_b_sig_matrix_path, sep="\t", index_col=0
            )
        else:
            logger.warning(
                f"Pass B signature matrix not found at "
                f"{pass_b_sig_matrix_path}"
            )
            pass_b_signature_matrix = None

        # Pass B is the official result: everything downstream (burden,
        # mu_tau, gene presence, R^2) must see the same cleaned
        # mutation set pass B was fit on.
        self._mutation_db = cleaned_db
        self._sig_assignments = pass_b_assignments
        self._signature_matrix = pass_b_signature_matrix

        print()
        return pass_b_assignments

    def generate_contexts_by_gene(
        self, fastas=None, gene_universe="own_cohort"
    ):
        """Generate trinucleotide context counts by gene.

        Computes trinucleotide context counts from FASTA files and
        stores them in ``self._contexts_by_gene``.

        Parameters
        ----------
        fastas : str, Path, list, or None, default None
            Path to FASTA file(s) containing gene sequences. Can be:
            - Single FASTA file path (str or Path)
            - List of FASTA file paths
            - None: automatically uses locations.location_cds_fasta
        gene_universe : {"own_cohort", "wes_target"}, default "own_cohort"
            Which genes to compute contexts (and, downstream, mutation
            rates) for:

            - ``"own_cohort"``: restricted to genes present in this
              dataset's own mutation database -- the original
              behavior. For low-mutation-burden cohorts this can be
              a small fraction of the exome, which biases downstream
              per-gene rate estimates (see ``"wes_target"``).
            - ``"wes_target"``: the union of genes present in this
              dataset's own mutation database *and* MC3's TCGA
              WES-target gene set (:func:`sigmutsel.wes_target.get_wes_target_gene_ids`)
              -- a principled, cohort-independent approximation of
              which genes were actually capturable by TCGA's
              sequencing. Always a superset of ``"own_cohort"``'s
              result: genes with real mutation evidence are never
              dropped just because they're absent from the WES-target
              set (which can happen for genes annotated only in
              newer GENCODE releases than the target BED's hg19
              build).

        Returns
        -------
        pd.DataFrame
            DataFrame with genes as index and trinucleotide contexts
            as columns. Each cell contains the count of that context
            in that gene's sequence.

        Notes
        -----
        The mutation database must be loaded before calling this
        method (e.g., via generate_mutation_db() or load_dataset()).

        Trinucleotide contexts are represented as 3-letter strings
        (e.g., 'ACA', 'TCG'). The counts represent how many times
        each trinucleotide appears in the gene's coding sequence.

        Examples
        --------
        >>> from locations import location_cds_fasta
        >>>
        >>> # Generate mutation database first
        >>> dataset.generate_mutation_db()
        >>>
        >>> # Generate contexts (caching handled internally)
        >>> contexts = dataset.generate_contexts_by_gene(
        ...     location_cds_fasta)
        >>>
        >>> # Access the data
        >>> print(dataset.contexts_by_gene.head())
        """
        from .contexts_by_gene import compute_contexts_by_gene

        restrict_to_db = self._resolve_gene_universe(gene_universe)

        self._contexts_by_gene = compute_contexts_by_gene(
            fastas, restrict_to_db=restrict_to_db
        )
        self._contexts_by_gene_gene_universe = gene_universe

        return self._contexts_by_gene

    def _resolve_gene_universe(self, gene_universe):
        """Turn a gene_universe name into a `restrict_to_db` argument.

        Shared by :meth:`generate_contexts_by_gene` and
        :meth:`generate_consequence_contexts_by_gene` so the two
        opportunity tables can never be built over different gene
        sets. See :meth:`generate_contexts_by_gene` for what the
        accepted values mean.
        """
        if self._mutation_db is None:
            raise ValueError(
                "Mutation database must be loaded before computing "
                "contexts. Call generate_mutation_db() or "
                "load_dataset() first."
            )

        if gene_universe == "own_cohort":
            return self.mutation_db

        if gene_universe == "wes_target":
            from .wes_target import get_wes_target_gene_ids

            own_mask = self.mutation_db["variant"].notna() & (
                self.mutation_db["ensembl_gene_id"].notna()
            )
            own_ids = set(
                self.mutation_db.loc[
                    own_mask, "ensembl_gene_id"
                ].astype(str)
            )
            return get_wes_target_gene_ids() | own_ids

        raise ValueError(
            f"Unknown gene_universe {gene_universe!r}; expected "
            "'own_cohort' or 'wes_target'."
        )

    def generate_consequence_contexts_by_gene(
        self, fastas=None, gene_universe="own_cohort"
    ):
        """Generate synonymous/non-synonymous opportunity counts.

        Splits the same per-gene CDS opportunities that
        :meth:`generate_contexts_by_gene` counts into a synonymous and
        a non-synonymous channel, bucketed by the 96 canonical SBS
        types, and stores them in ``self._contexts_by_gene_syn`` and
        ``self._contexts_by_gene_nonsyn``. See
        :mod:`sigmutsel.consequence_contexts_by_gene` for the exact
        counting rules and caveats.

        Nothing in the default pipeline calls this method or consumes
        its output yet: it is additive infrastructure for the
        consequence-split rate model, and
        :meth:`build_full_dataset` deliberately does not run it (it
        costs a second full pass over the CDS FASTA).

        Parameters
        ----------
        fastas : str, Path, list, or None, default None
            Forwarded to
            :func:`consequence_contexts_by_gene.compute_consequence_contexts_by_gene`.
            Must be a **CDS** FASTA (the default,
            ``locations.location_cds_fasta``, is one) -- a sequence
            with no reading frame has no synonymous channel.
        gene_universe : {"own_cohort", "wes_target"}, default "own_cohort"
            Same meaning as in :meth:`generate_contexts_by_gene`. Use
            the same value there and here: the sum identity below only
            holds gene-by-gene for genes present in both tables.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame)
            ``(contexts_by_gene_syn, contexts_by_gene_nonsyn)``, both
            genes × 96 canonical SBS types.

        Raises
        ------
        ValueError
            If ``signature_class`` is not ``"SBS"``. "Synonymous" is a
            codon-level concept that only makes sense for single-base
            substitutions; DBS/ID/CN/SV have no analogue and are left
            structurally untouched by this mechanism.

        Notes
        -----
        For every gene and every SBS type τ::

            contexts_by_gene_syn[τ] + contexts_by_gene_nonsyn[τ]
                == contexts_by_gene[extract_context(τ)]

        which is what makes ``p_gτ^(syn) + p_gτ^(nonsyn) == p_gτ``
        exact downstream.
        """
        from .consequence_contexts_by_gene import (
            compute_consequence_contexts_by_gene,
        )

        if self.signature_class != "SBS":
            raise ValueError(
                "Consequence-split opportunity counts are SBS-only; "
                f"this dataset has signature_class="
                f"{self.signature_class!r}. Synonymous/non-synonymous "
                "is a codon-level distinction with no clean analogue "
                "for DBS/ID/CN/SV."
            )

        restrict_to_db = self._resolve_gene_universe(gene_universe)

        (
            self._contexts_by_gene_syn,
            self._contexts_by_gene_nonsyn,
        ) = compute_consequence_contexts_by_gene(
            fastas, restrict_to_db=restrict_to_db
        )

        return (
            self._contexts_by_gene_syn,
            self._contexts_by_gene_nonsyn,
        )

    def build_full_dataset(
        self,
        fastas=None,
        gene_universe="own_cohort",
        regenerate_mutation_db=True,
        **kwargs,
    ):
        """Run the full data-generation pipeline for this dataset.

        Parameters
        ----------
        fastas : str, Path, list, or None, default None
            Forwarded to :meth:`generate_contexts_by_gene`.
        gene_universe : {"own_cohort", "wes_target"}, default "own_cohort"
            Forwarded to :meth:`generate_contexts_by_gene`.
        regenerate_mutation_db : bool, default True
            If False, skip calling :meth:`generate_mutation_db` and
            use whatever is already in ``self.mutation_db`` (raises if
            it hasn't been loaded). For callers that already built
            (and possibly modified) the mutation database themselves
            -- e.g. :meth:`run_two_pass_signature_decomposition`,
            which needs ``mutation_db`` built *before* it can compute
            per-mutation artifact probabilities, and updates it
            in-place with the artifact-mutation-cleaned version
            afterward. Calling this with the default `True` right
            after that would silently regenerate and overwrite the
            cleaned database with the original, uncleaned one.
        **kwargs : dict
            Forwarded to :meth:`generate_mutation_db` -- e.g.
            `qc_mode=True` (optionally with `qc_kwargs`) to enable
            the structured QC pipeline in :mod:`qc` instead of the
            default silent validation. Ignored if
            `regenerate_mutation_db` is False.
        """
        title = "Mutation data: building compact mutation database."
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        if regenerate_mutation_db:
            self.generate_mutation_db(**kwargs)
        elif not self.has_mutation_db():
            raise ValueError(
                "regenerate_mutation_db=False but no mutation_db is "
                "loaded -- call generate_mutation_db() (or "
                "run_two_pass_signature_decomposition()) first."
            )
        print()

        title = "Gene presence: computing matrices."
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        self.compute_gene_presence()
        self.compute_gene_presence_non_silent()
        print()

        title = "Contexts by gene: computing opportunities."
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        self.generate_contexts_by_gene(
            fastas=fastas, gene_universe=gene_universe
        )
        print()

        title = "Variant data: generating annotations and presence."
        print("=" * len(title))
        print(title)
        print("=" * len(title))
        self.generate_variant_db()
        self.compute_variants_present()
        print()


def _aggregate_signature_dict(
    sig_dict, signature_selection, include_other=False
):
    """Aggregate a {signature: DataFrame} dict per signature_selection.

    Shared grouping logic behind :meth:`Model.aggregate_signatures`,
    factored out so it can also be applied to ``mu_taus`` (not just
    ``base_mus``) when a model was built with ``signature_selection``
    --- summation commutes with the later gene/type mixing, so
    aggregating raw per-signature ``mu_taus`` first gives the same
    result as aggregating the derived per-signature ``base_mus``.
    See :meth:`Model.aggregate_signatures` for the grouping rules.
    """
    aggregated = {}
    matched_signatures = set()

    for item in signature_selection:
        if isinstance(item, (tuple, list)):
            group_name = "+".join(item)
            group_sum = None
            for sig in item:
                if sig in sig_dict:
                    matched_signatures.add(sig)
                    if group_sum is None:
                        group_sum = sig_dict[sig].copy()
                    else:
                        group_sum += sig_dict[sig]
            if group_sum is not None:
                aggregated[group_name] = group_sum

        elif isinstance(item, str):
            if item in sig_dict:
                aggregated[item] = sig_dict[item].copy()
                matched_signatures.add(item)
            else:
                matching_sigs = [
                    sig for sig in sig_dict if sig.startswith(item)
                ]

                if matching_sigs:
                    agg_sum = None
                    for sig in matching_sigs:
                        matched_signatures.add(sig)
                        if agg_sum is None:
                            agg_sum = sig_dict[sig].copy()
                        else:
                            agg_sum += sig_dict[sig]
                    aggregated[item] = agg_sum

    if include_other:
        other_sigs = [
            sig for sig in sig_dict if sig not in matched_signatures
        ]

        if other_sigs:
            other_sum = None
            for sig in other_sigs:
                if other_sum is None:
                    other_sum = sig_dict[sig].copy()
                else:
                    other_sum += sig_dict[sig]
            aggregated["other"] = other_sum

    return aggregated


@dataclass(repr=False, init=False)
class Model:
    """Signature based, mutation and selection model.

    Each Model instance represents one specific analysis with a fixed
    set of covariates. To test different covariate combinations,
    create multiple Model instances.

    Attributes
    ----------
    dataset : MutationDataset
        Reference to the shared mutation dataset.
    cov_matrix : pd.DataFrame or None
        Covariate matrix (genes × covariates) for THIS model.
        Should contain only the covariates this model uses.
        If None, this is a baseline model with no covariates.
        If provided during initialization, it will be automatically
        reindexed to match dataset.contexts_by_gene.index via
        assign_cov_matrix().
    cov_effects : np.ndarray or None
        Estimated covariate effect coefficients from MAP estimation.
        Lazy-loaded.
    cov_effects_posteriors : object or None
        Posterior samples from MCMC (arviz.InferenceData).
        Lazy-loaded.
    mu_taus : pd.DataFrame or dict or None
        Baseline mutation rates per type per tumor. Can be:
        - pd.DataFrame for signature-independent models
        - dict of DataFrames for signature-separated models (when
          separate_per_sigma=True)
        Lazy-loaded via compute_mu_taus().
    base_mus : dict or pd.DataFrame or None
        Baseline mutation rates per gene per tumor. Can be:
        - pd.DataFrame for signature-independent models
        - dict of DataFrames for signature-separated models
        Lazy-loaded via compute_base_mus().
    mu_gs : pd.DataFrame or None
        Mutation rates per gene per sample, incorporating covariate
        effects. Always a DataFrame with genes as index and samples
        as columns, regardless of whether base_mus is
        signature-separated.
        Lazy-loaded via compute_mu_gs().
    mu_ms : pd.DataFrame or None
        Mutation rates per variant per sample. Lazy-loaded.
    Auto-initialization parameters
        Optional keyword arguments ``L_low``, ``L_high``,
        ``cut_at_L_low``,
        ``cov_effects_per_sigma``,
        ``prob_g_tau_tau_independent``, ``signature_selection``, and
        ``include_other`` can be provided at initialization to
        automatically run the corresponding setup steps (mutation
        burdens, baseline rates, and optional signature
        aggregation). ``include_other`` (default False) is forwarded
        to :meth:`aggregate_signatures`: set it to True to bucket
        every signature not named in ``signature_selection`` into a
        single ``"other"`` group instead of dropping it from
        coefficient fitting entirely.
    prob_g_tau_tau_independent : bool or None
        Flag indicating whether base_mus were computed using
        type-independent gene probabilities. Set by
        compute_base_mus().
    cov_effects_kwargs : dict
        Keyword arguments for MAP estimation
        (estimate_cov_effects).
    passenger_genes_r2 : float or None
        R² value comparing model predictions with observed data of
        passenger genes. Lazy-loaded.
    saved_location : str or None
        Filesystem path where the model snapshot was last saved or
        loaded from. None until save_model() or load_model() runs.

    Examples
    --------
    >>> # Create a baseline model (no covariates)
    >>> model_no_cov = Model(dataset, None)
    >>> model_no_cov.has_covariates()
    False
    >>> model_no_cov.n_covariates
    0

    >>> # Create a simple model with MRT covariate
    >>> model = Model(
    ...     dataset,
    ...     cov_matrix_full[['mrt']])
    >>> model.has_covariates()
    True
    >>> model.n_covariates
    1

    >>> # Create a model with multiple covariates
    >>> model = Model(
    ...     dataset,
    ...     cov_matrix_full[['mrt', 'log1p_gexp', 'log1p_atac']])
    >>> model.n_covariates
    3
    >>> # Note: cov_matrix is automatically reindexed to match
    >>> # dataset.contexts_by_gene.index during initialization

    >>> # Compute baseline mutation rates
    >>> model.compute_mu_taus()
    >>> model.compute_base_mus()
    >>> model.has_sig_dependent_mus()
    False

    """

    dataset: MutationDataset | str | Path
    cov_matrix: pd.DataFrame | None = None
    cov_effects_kwargs: dict = field(default_factory=dict)

    # Results (populated by run functions, lazy-loaded)
    _base_mus: dict | pd.DataFrame = None
    _base_mus_syn: dict | pd.DataFrame = None
    _base_mus_nonsyn: dict | pd.DataFrame = None
    cov_effects: np.ndarray = None
    _n_in_cov_effects_estimation: int = None
    _passenger_genes_r2: float = None
    _passenger_genes_r2_non_silent: float = None
    _passenger_genes_r2_non_silent_counts: float = None
    _rg_theta: float = None
    _rg_statistics: dict = None
    _rg_separate_c: bool | str = False
    _rg_delta_intercept: float = None
    _channel_cov_effects: np.ndarray = None
    cov_effects_posteriors: object = None
    _mu_gs: pd.DataFrame = None
    mu_ms: pd.DataFrame = None
    _mu_taus: pd.DataFrame | dict = None
    _prob_g_tau_tau_independent: bool | None = None
    gammas: dict = field(default_factory=dict, init=False, repr=False)
    # Tracks which entries in `gammas` are literally the object
    # load_model() read from disk (see save_model's re-save guard).
    _gammas_loaded_from_disk: dict = field(
        default_factory=dict, init=False, repr=False
    )
    _saved_location: str | None = field(
        default=None, init=False, repr=False
    )
    _auto_mu_taus_kwargs: dict = field(
        default_factory=dict, init=False, repr=False
    )
    _auto_cov_effects_per_sigma: bool | None = field(
        default=None, init=False, repr=False
    )
    _auto_prob_g_tau_tau_independent: bool | None = field(
        default=None, init=False, repr=False
    )
    _auto_signature_selection: list | None = field(
        default=None, init=False, repr=False
    )
    _auto_include_other: bool = field(
        default=False, init=False, repr=False
    )

    def __init__(
        self,
        dataset: MutationDataset | str | Path,
        cov_matrix: pd.DataFrame | None = None,
        *,
        cov_effects_kwargs: dict | None = None,
        L_low: float | None = None,
        L_high: float | None = None,
        cut_at_L_low: bool | None = None,
        cov_effects_per_sigma: bool | None = None,
        prob_g_tau_tau_independent: bool | None = None,
        signature_selection: list | tuple | None = None,
        include_other: bool = False,
    ):
        self.dataset = dataset
        self.cov_matrix = cov_matrix
        self.cov_effects_kwargs = (
            cov_effects_kwargs.copy() if cov_effects_kwargs else {}
        )

        self._base_mus = None
        self._base_mus_syn = None
        self._base_mus_nonsyn = None
        self.cov_effects = None
        self._n_in_cov_effects_estimation = None
        self._passenger_genes_r2 = None
        self._passenger_genes_r2_non_silent = None
        self._passenger_genes_r2_non_silent_counts = None
        self._rg_theta = None
        self._rg_statistics = None
        self._rg_separate_c = False
        self._rg_delta_intercept = None
        self._channel_cov_effects = None
        self.cov_effects_posteriors = None
        self._mu_gs = None
        self.mu_ms = None
        self._mu_taus = None
        self._prob_g_tau_tau_independent = None
        self.gammas = {}
        self._gammas_loaded_from_disk = {}

        self._auto_mu_taus_kwargs = {
            "L_low": L_low,
            "L_high": L_high,
            "cut_at_L_low": cut_at_L_low,
        }
        self._auto_cov_effects_per_sigma = cov_effects_per_sigma
        self._auto_prob_g_tau_tau_independent = (
            prob_g_tau_tau_independent
        )
        self._auto_signature_selection = (
            list(signature_selection)
            if signature_selection is not None
            else None
        )
        self._auto_include_other = include_other
        self._saved_location = None

        self.__post_init__()
        self._apply_auto_configuration()

    def __post_init__(self):
        """Post-initialization processing.

        If a covariate matrix is provided during initialization,
        automatically calls assign_cov_matrix() to properly reindex
        it to match the genes in contexts_by_gene.
        """
        if isinstance(self.dataset, (str, Path)):
            self.dataset = MutationDataset.load_dataset(self.dataset)

        if self.cov_matrix is not None:
            # Store the original cov_matrix temporarily
            cov_matrix_input = self.cov_matrix
            # Reset to None to avoid issues in assign_cov_matrix
            self.cov_matrix = None
            # Call assign_cov_matrix to properly reindex
            self.assign_cov_matrix(cov_matrix_input)

    def _apply_auto_configuration(self):
        """Apply automatic mu_taus/base_mus/signature setup if requested."""
        auto_mu_kwargs = {
            key: value
            for key, value in self._auto_mu_taus_kwargs.items()
            if value is not None
        }

        need_mu_taus = (
            bool(auto_mu_kwargs)
            or self._auto_cov_effects_per_sigma is not None
            or self._auto_prob_g_tau_tau_independent is not None
            or self._auto_signature_selection is not None
        )

        if need_mu_taus and self._mu_taus is None:
            separate = bool(self._auto_cov_effects_per_sigma)
            self.compute_mu_taus(
                separate_per_sigma=separate, **auto_mu_kwargs
            )

        if (
            self._auto_prob_g_tau_tau_independent is not None
            and self._base_mus is None
        ):
            self.compute_base_mus(
                prob_g_tau_tau_independent=(
                    self._auto_prob_g_tau_tau_independent
                )
            )

        if self._auto_signature_selection and self._base_mus is None:
            self.compute_base_mus(prob_g_tau_tau_independent=False)

        if self._auto_signature_selection:
            self.aggregate_signatures(
                self._auto_signature_selection,
                include_other=self._auto_include_other,
            )

    def __repr__(self):
        """Show model configuration and loaded results (custom repr)."""
        # Model configuration
        parts = ["Model("]

        config = []
        config.append(f"n_covariates={self.n_covariates}")

        if self.n_covariates > 0:
            cov_names = ", ".join(self.covariate_names[:3])
            if self.n_covariates > 3:
                cov_names += f", ... (+{self.n_covariates - 3} more)"
            config.append(f"covariates=[{cov_names}]")

        parts.append("  " + ", ".join(config))

        # Loaded results (only show non-None attributes)
        loaded = []
        if self._base_mus is not None:
            if isinstance(self._base_mus, dict):
                loaded.append(
                    f"base_mus: dict with {len(self._base_mus)} signatures"
                )
            else:
                loaded.append(f"base_mus: {self._base_mus.shape}")
        if self._mu_taus is not None:
            if isinstance(self._mu_taus, dict):
                loaded.append(
                    f"mu_taus: dict with {len(self._mu_taus)} signatures"
                )
            else:
                loaded.append(f"mu_taus: {self._mu_taus.shape}")
        if self.cov_effects is not None:
            loaded.append(f"cov_effects: {self.cov_effects.shape}")
        if self._passenger_genes_r2 is not None:
            loaded.append(f"R²={self._passenger_genes_r2:.4f}")
        if self._passenger_genes_r2_non_silent is not None:
            loaded.append(
                "R²(non-silent)="
                f"{self._passenger_genes_r2_non_silent:.4f}"
            )
        if self._passenger_genes_r2_non_silent_counts is not None:
            loaded.append(
                "R²(non-silent counts)="
                f"{self._passenger_genes_r2_non_silent_counts:.4f}"
            )
        if self.cov_effects_posteriors is not None:
            loaded.append("posteriors: available")
        if self._mu_gs is not None:
            loaded.append(f"mu_gs: {self._mu_gs.shape}")
        if self.mu_ms is not None:
            loaded.append(f"mu_ms: {self.mu_ms.shape}")

        if loaded:
            loaded_str = "\n    ".join(loaded)
            parts.append(f"  loaded_results:\n    {loaded_str}")

        parts.append(")")
        return "\n".join(parts)

    @property
    def covariate_names(self):
        """List of covariate names used in this model."""
        if self.cov_matrix is None:
            return []
        return list(self.cov_matrix.columns)

    @property
    def n_covariates(self):
        """Number of covariates in this model."""
        if self.cov_matrix is None:
            return 0
        return self.cov_matrix.shape[1]

    def has_covariates(self):
        """Check if model uses covariates (not a baseline model)."""
        return self.cov_matrix is not None

    @property
    def n_in_cov_effects_estimation(self):
        """Number of passenger genes used in covariate effects estimation.

        Returns the count of passenger genes with complete covariate
        data (no NaN values) that are used for estimating covariate
        effects. This provides insight into the sample size for the
        estimation.

        If `estimate_cov_effects()` has not been called yet, this
        property computes the expected count based on current data
        and warns that the actual value will be set during
        estimation.

        Returns
        -------
        int
            Number of passenger genes with complete covariates used
            in covariate effects estimation.

        Raises
        ------
        ValueError
            If covariate matrix is not assigned. Call
            assign_cov_matrix() first.
        ValueError
            If contexts_by_gene is not loaded (needed to identify
            which genes are available).

        Notes
        -----
        The count is computed by:
        1. Identifying passenger genes (not in Cancer Gene Census)
        2. Filtering to genes in contexts_by_gene
        3. Filtering to genes with no NaN values in any covariate

        A warning is issued if this property is accessed before
        calling `estimate_cov_effects()`, as the returned value is
        a preview based on current data rather than the actual genes
        used in estimation.

        Examples
        --------
        >>> # Check how many genes will be used before estimation
        >>> model.assign_cov_matrix(cov_matrix_full[['mrt']])
        >>> print(f"Will use {model.n_in_cov_effects_estimation} genes")
        UserWarning: n_in_cov_effects_estimation not set yet...
        >>>
        >>> # After estimation, no warning
        >>> model.estimate_cov_effects()
        >>> print(f"Used {model.n_in_cov_effects_estimation} genes")
        """
        from .estimate_presence import filter_passenger_genes_ensembl

        if self.cov_effects is None:
            logger.warning(
                "Covariate effects have not been estimated yet. "
                "Run estimate_cov_effects() to set "
                "n_in_cov_effects_estimation."
            )

        # If already set, return it
        if self._n_in_cov_effects_estimation is not None:
            return self._n_in_cov_effects_estimation

        # Otherwise compute it and warn
        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix not assigned. "
                "Call assign_cov_matrix() first."
            )

        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first."
            )

        # Identify passenger genes with complete covariate data
        passenger_gene_ids = filter_passenger_genes_ensembl(
            self.cov_matrix.index
        )

        # Filter to genes with no NaN values in any covariate
        passenger_cov = self.cov_matrix.loc[passenger_gene_ids]
        complete_mask = ~passenger_cov.isna().any(axis=1)
        n_complete = complete_mask.sum()

        # Warn that this is a preview
        logger.warning(
            "n_in_cov_effects_estimation not set yet. "
            "Returning preview value (%s) based on current "
            "passenger genes with complete covariates. "
            "This will be set to the actual value when "
            "estimate_cov_effects() is called.",
            n_complete,
        )

        return n_complete

    @property
    def passenger_genes_r2(self):
        """R² for passenger gene mutation frequency predictions.

        Returns the R² score comparing predicted vs observed mutation
        frequency for passenger genes. For models without covariates,
        this property automatically calls `estimate_passenger_genes_r2()`
        if not yet computed.

        Returns
        -------
        float
            R² score for passenger genes. Values range from -∞ to 1.

        Notes
        -----
        **Automatic computation for baseline models:**

        For models without a covariate matrix (baseline models), this
        property automatically computes the R² if it hasn't been set
        yet. This provides convenient access without requiring explicit
        method calls.

        For models with covariates, R² is automatically computed by
        `estimate_cov_effects()`, so this property simply returns the
        stored value.

        Examples
        --------
        >>> # Baseline model - automatic computation
        >>> model_no_cov = Model(dataset, None)
        >>> model_no_cov.compute_mu_taus()
        >>> model_no_cov.compute_base_mus()
        >>> r2 = model_no_cov.passenger_genes_r2  # Computed automatically
        >>> print(f"Baseline R²: {r2:.4f}")
        >>>
        >>> # Model with covariates - already computed
        >>> model = Model(dataset)
        >>> model.assign_cov_matrix(cov_matrix)
        >>> model.compute_mu_taus()
        >>> model.compute_base_mus()
        >>> model.estimate_cov_effects()  # R² computed here
        >>> r2 = model.passenger_genes_r2  # Just returns stored value
        >>> print(f"R²: {r2:.4f}")
        """
        # If already computed, return it
        if self._passenger_genes_r2 is not None:
            return self._passenger_genes_r2

        # For baseline models (no covariates), compute automatically
        if self.cov_matrix is None:
            self.estimate_passenger_genes_r2()
            return self._passenger_genes_r2

        # If covariate effects exist, compute R² now
        if self.cov_effects is not None:
            logger.info(
                "Passenger genes R² not yet computed; running "
                "estimate_passenger_genes_r2() now."
            )
            self.estimate_passenger_genes_r2()
            return self._passenger_genes_r2

        logger.info(
            "Passenger genes R² unavailable: "
            "estimate_cov_effects() has not been run yet."
        )

        # For models with covariates but no R² yet, return None (user
        # should call estimate_cov_effects or
        # estimate_passenger_genes_r2)
        return None

    @property
    def passenger_genes_r2_non_silent_counts(self):
        """Passenger-gene R² against the non-silent *count* target.

        The headline for a count (Poisson) fit: expected
        ``Σ_j μ_g^(nonsyn,j)`` against observed
        ``Σ_j N_g^(nonsyn,j)``, with no ``1 - exp(-μ)`` censoring
        step. Never computed automatically -- call
        ``estimate_passenger_genes_r2(target="non_silent_counts")``.
        """
        return self._passenger_genes_r2_non_silent_counts

    @property
    def passenger_genes_r2_non_silent(self):
        """Passenger-gene R² against the non-silent target.

        The honest number whenever the model being scored has seen
        silent counts (see
        :meth:`estimate_passenger_genes_r2`'s ``target``). Unlike
        :attr:`passenger_genes_r2`, this is never computed
        automatically -- call
        ``estimate_passenger_genes_r2(target="non_silent")``
        explicitly, since it needs the consequence-split baselines.
        """
        return self._passenger_genes_r2_non_silent

    def assign_cov_matrix(
        self,
        cov_matrix,
        run_pca=False,
        pca_kwargs=None,
        dr_method=None,
        dr_kwargs=None,
    ):
        """Assign covariate matrix, restricting to dataset genes.

        This method assigns a covariate matrix to the model after
        reindexing it to match the genes in contexts_by_gene.
        Optionally applies dimensionality reduction to the covariates.

        Note: This method is automatically called during Model
        initialization if a cov_matrix is provided to the constructor
        (e.g., Model(dataset, cov_matrix)).

        This ensures that:
        1. Only genes with context information are included
        2. Gene order matches dataset.contexts_by_gene.index
        3. Missing genes are handled appropriately
        4. (Optional) Covariates are transformed via PCA or
           Riemannian STATS

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            Covariate matrix with genes as index and covariates as
            columns. Index should be Ensembl gene IDs.
        run_pca : bool, default False
            Deprecated shorthand for ``dr_method='pca'``. If True
            and ``dr_method`` is None, PCA is used.
        pca_kwargs : dict or None, default None
            Deprecated. Use ``dr_kwargs`` instead. Forwarded to
            :func:`utils.run_pca_on_covariates` when ``run_pca=True``
            and ``dr_kwargs`` is None.
        dr_method : str or None, default None
            Dimensionality reduction method. Options:

            - ``'pca'``: sklearn PCA (see
              :func:`utils.run_pca_on_covariates`)
            - ``'riemannian_stats'``: Riemannian STATS (see
              :func:`utils.run_riemannian_stats_on_covariates`)
            - ``None``: no reduction, use covariates as-is

        dr_kwargs : dict or None, default None
            Keyword arguments forwarded to the chosen DR function.
            Common options for both methods:
            - n_components : int, number of components to keep
            - columns : list[str], subset of columns to include
            - standardize : bool, default True
            - dropna : str, default 'any'
            Riemannian-only:
            - n_neighbors : int, default 15
            - min_dist : float, default 0.1
            - metric : str, default 'euclidean'

        Returns
        -------
        pd.DataFrame
            The reindexed (and optionally PCA-transformed) covariate
            matrix that was assigned to self.cov_matrix.

        Raises
        ------
        ValueError
            If contexts_by_gene has not been loaded in the dataset.
            Call dataset.generate_contexts_by_gene() or load_dataset() first.

        Notes
        -----
        **Reindexing behavior:**

        The input cov_matrix is reindexed using
        `dataset.contexts_by_gene.index`. This means:
        - Only genes with trinucleotide context information are kept
        - Genes are ordered to match contexts_by_gene
        - If a gene in contexts_by_gene is missing from cov_matrix,
          it will have NaN values for all covariates

        **PCA transformation:**

        When run_pca=True, the method performs:
        1. Reindex to dataset genes (as above)
        2. Run PCA using :func:`utils.run_pca_on_covariates`
        3. Replace covariates with principal components (PC1, PC2, ...)

        The PCA transformation is useful for:
        - Reducing dimensionality when many correlated covariates exist
        - Creating orthogonal features for modeling
        - Avoiding multicollinearity issues

        **Typical workflow:**

        1. Generate mutation database and contexts:
            >>> dataset.generate_mutation_db("data/mutations.parquet")
            >>> dataset.generate_contexts_by_gene(...)

        2. Create full covariate matrix for all genes:
            >>> cov_matrix_full = pd.DataFrame({
            ...     'mrt': mrt_per_gene,
            ...     'log1p_gexp': np.log1p(gexp_per_gene),
            ...     'log1p_atac': np.log1p(atac_per_gene)})

        3. Create model and assign restricted covariate matrix:
            >>> model = Model(dataset)
            >>> model.assign_cov_matrix(cov_matrix_full[['mrt', 'log1p_gexp']])

        This ensures the covariate matrix exactly matches the genes
        with trinucleotide context information, which determines the
        base_mus genes.

        Examples
        --------
        >>> # Basic usage: assign covariates
        >>> cov_matrix_full = pd.DataFrame({
        ...     'mrt': mrt_per_gene,
        ...     'log1p_gexp': np.log1p(gexp_per_gene)},
        ...     index=all_gene_ids)
        >>>
        >>> model = Model(dataset)
        >>> model.assign_cov_matrix(cov_matrix_full)
        >>>
        >>> # Check that genes match
        >>> assert (model.cov_matrix.index ==
        ...         dataset.contexts_by_gene.index).all()
        >>>
        >>> # Assign different subset of covariates
        >>> model.assign_cov_matrix(cov_matrix_full[['mrt']])
        >>>
        >>> # Run PCA to reduce dimensionality
        >>> model_pca = Model(dataset)
        >>> model_pca.assign_cov_matrix(
        ...     cov_matrix_full,
        ...     run_pca=True,
        ...     pca_kwargs={'n_components': 3})
        >>> # Now model_pca.cov_matrix has columns: PC1, PC2, PC3
        >>>
        >>> # PCA on subset of columns
        >>> model_pca2 = Model(dataset)
        >>> model_pca2.assign_cov_matrix(
        ...     cov_matrix_full,
        ...     run_pca=True,
        ...     pca_kwargs={
        ...         'columns': ['log1p_gexp', 'log1p_atac', 'log1p_h3k4me3'],
        ...         'n_components': 2,
        ...         'standardize': True})

        See Also
        --------
        generate_contexts_by_gene : Must be called first on dataset
        utils.run_pca_on_covariates : PCA implementation
        """
        # Ensure contexts_by_gene has been loaded
        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first."
            )

        # Reindex to match contexts_by_gene
        reindexed = cov_matrix.reindex(
            self.dataset.contexts_by_gene.index
        )

        # backwards-compat: run_pca=True maps to dr_method='pca'
        if run_pca and dr_method is None:
            dr_method = "pca"

        if dr_method == "pca":
            from .utils import run_pca_on_covariates

            kwargs = (
                dr_kwargs
                if dr_kwargs is not None
                else (pca_kwargs or {})
            )
            self.cov_matrix = run_pca_on_covariates(
                reindexed, **kwargs
            )
        elif dr_method == "riemannian_stats":
            from .utils import run_riemannian_stats_on_covariates

            self.cov_matrix = run_riemannian_stats_on_covariates(
                reindexed, **(dr_kwargs or {})
            )
        else:
            self.cov_matrix = reindexed

        return self.cov_matrix

    @property
    def base_mus(self):
        """Baseline mutation rates per gene per tumor (lazy loaded)."""
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates not computed. "
                "Call compute_base_mus() first."
            )
        return self._base_mus

    @property
    def base_mus_syn(self):
        """Synonymous-channel baseline rates (lazy loaded)."""
        if self._base_mus_syn is None:
            raise ValueError(
                "Synonymous-channel baseline rates not computed. "
                "Call compute_channel_base_mus() first."
            )
        return self._base_mus_syn

    @property
    def base_mus_nonsyn(self):
        """Non-synonymous-channel baseline rates (lazy loaded)."""
        if self._base_mus_nonsyn is None:
            raise ValueError(
                "Non-synonymous-channel baseline rates not computed. "
                "Call compute_channel_base_mus() first."
            )
        return self._base_mus_nonsyn

    def has_channel_base_mus(self):
        """Whether both consequence channels' baselines exist."""
        return (
            self._base_mus_syn is not None
            and self._base_mus_nonsyn is not None
        )

    @base_mus.setter
    def base_mus(self, value):
        """Set baseline mutation rates.

        Validates that the gene index matches contexts_by_gene.
        """

        # Allow None
        if value is None:
            self._base_mus = value
            self._prob_g_tau_tau_independent = None
            return

        # Check if contexts_by_gene is loaded
        if self.dataset._contexts_by_gene is None:
            logger.warning(
                "Cannot validate base_mus index: "
                "contexts_by_gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() to ensure gene indices match."
            )
            self._base_mus = value
            return

        # Validate index for DataFrames
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(
                self.dataset.contexts_by_gene.index
            ):
                logger.warning(
                    "base_mus index does not match "
                    "dataset.contexts_by_gene.index. "
                    "This may cause errors when computing mutation "
                    "rates. To fix: reindex base_mus to match "
                    "contexts_by_gene.index, or recompute base_mus "
                    "after loading contexts_by_gene."
                )

        # Validate index for dicts of DataFrames (signature-separated)
        elif isinstance(value, dict):
            for sig_name, df in value.items():
                if isinstance(
                    df, pd.DataFrame
                ) and not df.index.equals(
                    self.dataset.contexts_by_gene.index
                ):
                    logger.warning(
                        "base_mus['%s'] index does not match "
                        "dataset.contexts_by_gene.index. "
                        "This may cause errors when computing "
                        "mutation rates. To fix: reindex all "
                        "base_mus DataFrames to match "
                        "contexts_by_gene.index, or recompute "
                        "base_mus after loading contexts_by_gene.",
                        sig_name,
                    )
                    break  # Only warn once

        self._base_mus = value

    def has_base_mus(self):
        """Check if baseline mutation rates have been computed."""
        return self._base_mus is not None

    def has_sig_dependent_mus(self):
        """Check if base_mus are signature-dependent (dict).

        Returns
        -------
        bool
            True if base_mus is a dict (signature-separated),
            False if it's a DataFrame (signature-independent).
        """
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates not computed. "
                "Call compute_base_mus() first."
            )
        return isinstance(self._base_mus, dict)

    def has_cov_effects(self):
        """Check if covariate effects have been estimated."""
        return self.cov_effects is not None

    def has_cov_effects_posteriors(self):
        """Check if covariate-effect posterior samples exist."""
        return self.cov_effects_posteriors is not None

    def has_mu_gs(self):
        """Check if gene-level mutation rates have been computed."""
        return self._mu_gs is not None

    def has_mu_ms(self):
        """Check if variant-level mutation rates have been computed."""
        return self.mu_ms is not None

    @property
    def prob_g_tau_tau_independent(self):
        """Whether base_mus were computed with type-independent p(g)."""
        if self._prob_g_tau_tau_independent is None:
            raise ValueError(
                "prob_g_tau_tau_independent not set. "
                "Call compute_base_mus() first."
            )
        return self._prob_g_tau_tau_independent

    @prob_g_tau_tau_independent.setter
    def prob_g_tau_tau_independent(self, value):
        """Set the probability independence flag."""
        self._prob_g_tau_tau_independent = value

    def is_submodel_of(self, other):
        """Check if this model is a submodel of another model."""
        if not isinstance(other, Model):
            return False

        if self.dataset is not other.dataset:
            return False

        if self._base_mus is not other._base_mus:
            return False

        if self._mu_taus is not other._mu_taus:
            return False

        if self.cov_matrix is None or other.cov_matrix is None:
            return self.cov_matrix is other.cov_matrix

        return set(self.cov_matrix.columns).issubset(
            other.cov_matrix.columns
        )

    def is_supermodel_of(self, other):
        """Check if this model is a supermodel of another model."""
        if not isinstance(other, Model):
            return False
        return other.is_submodel_of(self)

    def with_covariates_removed(self, covariates):
        """Return a copy of the model without the specified covariates.

        Parameters
        ----------
        covariates : str | Iterable[str]
            Covariate name or collection of names to remove from the
            covariate matrix. Names must exist in ``self.covariate_names``.

        Returns
        -------
        Model
            A new Model instance sharing the same dataset, mu_taus,
            base_mus, and other non-covariate results, but with
            covariates removed from the covariate matrix.
        """
        if self.cov_matrix is None:
            raise ValueError(
                "Model has no covariates to remove. "
                "Assign a covariate matrix first."
            )

        if isinstance(covariates, str):
            covariate_set = {covariates}
        else:
            covariate_set = set(covariates)

        invalid = covariate_set - set(self.covariate_names)
        if invalid:
            raise ValueError(
                f"Covariate(s) not found in model: {sorted(invalid)}"
            )

        remaining_columns = [
            col
            for col in self.covariate_names
            if col not in covariate_set
        ]

        new_model = self.copy()
        if remaining_columns:
            new_model.cov_matrix = self.cov_matrix[remaining_columns]
        else:
            new_model.cov_matrix = None
        return new_model

    def estimate_gamma(
        self,
        item,
        level=None,
        upper_bound_prior=None,
        store=True,
        non_silent=True,
        excluded_samples=None,
    ):
        """Estimate selection coefficient for a variant or gene.

        Parameters
        ----------
        item : str
            Variant identifier (e.g., "ZZZ3 p.Y721H"), gene name
            (e.g., "BRAF"), or ensembl_gene_id
            (e.g., "ENSG00000157764").
        level : str or None, optional
            Type of item: 'variant', 'gene', or None (auto-detect).
            Default None.
        upper_bound_prior : float or None, optional
            Upper bound for gamma prior. If None (default), uses
            :func:`estimate_gammas.estimate_gamma_from_mus`'s own
            default -- that function also auto-expands the bound if
            the posterior turns out to be bound-limited, so an
            explicit value here is only needed to change the
            starting point.
        store : bool, optional
            Whether to store result in self.gammas. Default True.
        non_silent : bool, optional
            For genes, whether to use non-silent mutations only.
            Default True.
        excluded_samples : collection of str or None, default None
            Tumor barcodes to drop from both the "present" and
            "absent" masks before estimation. See
            `_estimate_gamma_variant`/`_estimate_gamma_gene` for
            details.

        Returns
        -------
        dict
            Estimation results for the item.

        Examples
        --------
        >>> # Auto-detect variant
        >>> model.estimate_gamma("ZZZ3 p.Y721H")
        >>> # Auto-detect gene by name
        >>> model.estimate_gamma("BRAF")
        >>> # Auto-detect gene by ensembl_gene_id
        >>> model.estimate_gamma("ENSG00000157764")
        >>> # Explicit level specification
        >>> model.estimate_gamma("BRAF", level='gene', non_silent=True)
        """

        # Auto-detect level if not specified
        if level is None:
            level = self._detect_item_level(item)

        if level == "variant":
            result = self._estimate_gamma_variant(
                item,
                upper_bound_prior=upper_bound_prior,
                store=store,
                excluded_samples=excluded_samples,
            )
        elif level == "gene":
            result = self._estimate_gamma_gene(
                item,
                upper_bound_prior=upper_bound_prior,
                store=store,
                non_silent=non_silent,
                excluded_samples=excluded_samples,
            )
        else:
            raise ValueError(
                f"Invalid level: {level!r}. "
                "Must be 'variant', 'gene', or None."
            )

        self._report_gamma_posterior(result)
        return result

    def _detect_item_level(self, item):
        """Detect whether item is a variant or gene.

        Parameters
        ----------
        item : str
            Item to detect.

        Returns
        -------
        str
            'variant' or 'gene'.

        Raises
        ------
        ValueError
            If item cannot be identified.
        """
        # Check if it's a variant
        if self.mu_ms is not None and item in self.mu_ms.index:
            return "variant"
        if (
            hasattr(self.dataset, "_variants_present")
            and self.dataset._variants_present is not None
            and item in self.dataset._variants_present.index
        ):
            return "variant"

        # Check if it's an ensembl_gene_id
        if self._mu_gs is not None and item in self.mu_gs.index:
            return "gene"
        if (
            hasattr(self.dataset, "_genes_present")
            and self.dataset._genes_present is not None
            and item in self.dataset._genes_present.index
        ):
            return "gene"

        # Check if it's a gene name in mutation database
        if hasattr(self.dataset, "mutation_db"):
            mapping = (
                self.dataset.mutation_db[["gene", "ensembl_gene_id"]]
                .drop_duplicates()
                .set_index("gene")["ensembl_gene_id"]
            )
            if item in mapping.index:
                return "gene"

        raise ValueError(
            f"Could not identify {item!r} as a variant or gene. "
            "Please specify 'level' parameter explicitly."
        )

    def _report_gamma_posterior(self, result):
        """Print posterior summary for gamma inference if available."""
        try:
            import arviz as az
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning(
                "arviz is not installed; cannot summarize gamma posterior."
            )
            return

        if not hasattr(result, "posterior"):
            return

        # This is purely a convenience print; any az.summary failure
        # should degrade to a warning, not break the caller.
        try:
            summary = az.summary(result, var_names=["gamma"])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to summarize gamma posterior: %s", exc
            )
            return

        print("Gamma posterior summary:")
        print(summary.to_string())

    def _estimate_gamma_variant(
        self,
        variant,
        upper_bound_prior=None,
        store=True,
        excluded_samples=None,
    ):
        """Estimate selection coefficient for a variant.

        Parameters
        ----------
        variant : str
            Variant identifier.
        upper_bound_prior : float or None, optional
            Upper bound for gamma prior. If None, uses
            :func:`estimate_gammas.estimate_gamma_from_mus`'s own
            default (which also auto-expands if bound-limited).
        store : bool, optional
            Whether to store result.
        excluded_samples : collection of str or None, optional
            Tumor sample barcodes to drop entirely (from both the
            present and absent sets) before estimating gamma, e.g.
            samples flagged by :func:`sample_qc.combine_sample_flags`.
            Inverse-variance downweighting is not implemented here --
            see `Model.estimate_cov_effects`'s docstring for why.

        Returns
        -------
        dict
            Estimation results.
        """
        from .estimate_gammas import estimate_gamma_from_mus

        if self.mu_ms is None:
            self.compute_mu_ms()

        if variant not in self.mu_ms.index:
            raise ValueError(
                f"Variant {variant!r} not found in mutation rates."
            )

        variants_present = self.dataset.variants_present
        present_mask = variants_present.loc[variant] == 1
        absent_mask = ~present_mask
        if excluded_samples is not None:
            not_excluded = ~present_mask.index.isin(excluded_samples)
            present_mask &= not_excluded
            absent_mask &= not_excluded

        extra = (
            {}
            if upper_bound_prior is None
            else {"upper_bound_prior": upper_bound_prior}
        )
        result = estimate_gamma_from_mus(
            self.mu_ms.loc[variant][present_mask],
            self.mu_ms.loc[variant][absent_mask],
            **extra,
        )

        if store:
            self.gammas[variant] = result

        return result

    def _estimate_gamma_gene(
        self,
        gene,
        upper_bound_prior=None,
        store=True,
        non_silent=True,
        excluded_samples=None,
    ):
        """Estimate selection coefficient for a gene.

        Parameters
        ----------
        gene : str
            Gene name or ensembl_gene_id.
        upper_bound_prior : float or None, optional
            Upper bound for gamma prior. If None, uses
            :func:`estimate_gammas.estimate_gamma_from_mus`'s own
            default (which also auto-expands if bound-limited).
        store : bool, optional
            Whether to store result.
        non_silent : bool, optional
            Whether to use non-silent mutations only.
        excluded_samples : collection of str or None, optional
            Tumor sample barcodes to drop entirely (from both the
            present and absent sets) before estimating gamma, e.g.
            samples flagged by :func:`sample_qc.combine_sample_flags`.
            Inverse-variance downweighting is not implemented here --
            see `Model.estimate_cov_effects`'s docstring for why.

        Returns
        -------
        dict
            Estimation results.
        """
        from .estimate_gammas import estimate_gamma_from_mus

        if self._mu_gs is None:
            self.compute_mu_gs()

        gene_presence = (
            self.dataset.genes_present_non_silent
            if non_silent
            else self.dataset.genes_present
        )

        # Try to get ensembl_gene_id if gene is a name
        if gene in self.mu_gs.index:
            gene_id = gene
        else:
            mapping = (
                self.dataset.mutation_db[["gene", "ensembl_gene_id"]]
                .drop_duplicates()
                .set_index("gene")["ensembl_gene_id"]
            )
            if gene not in mapping:
                raise ValueError(
                    f"Gene {gene!r} not found in mutation database."
                )
            gene_id = mapping[gene]

        if gene_id not in self.mu_gs.index:
            raise ValueError(
                f"Gene ID {gene_id!r} not found in mu_gs."
            )

        present_mask = gene_presence.loc[gene_id] == 1
        absent_mask = ~present_mask
        if excluded_samples is not None:
            not_excluded = ~present_mask.index.isin(excluded_samples)
            present_mask &= not_excluded
            absent_mask &= not_excluded

        extra = (
            {}
            if upper_bound_prior is None
            else {"upper_bound_prior": upper_bound_prior}
        )
        result = estimate_gamma_from_mus(
            self.mu_gs.loc[gene_id][present_mask],
            self.mu_gs.loc[gene_id][absent_mask],
            **extra,
        )

        if store:
            # Always store with ensembl_gene_id for consistency
            self.gammas[gene_id] = result

        return result

    def plot_gamma_results(
        self,
        keys=None,
        level=None,
        change_gene_ids_to_names=True,
        **kwargs,
    ):
        """Plot posterior vs counts for selection results.

        Parameters
        ----------
        keys : str, list of str, or None, optional
            Keys to select from self.gammas for plotting.
            - If str: plot single result for that key
            - If list: plot results for those keys
            - If None: plot all results in self.gammas
            Default None.
        level : {'variant', 'gene'} or None, optional
            Level of results being plotted. If None, auto-detects
            based on whether keys are in variants_present or
            genes_present. Default None.
        change_gene_ids_to_names : bool, default True
            If True and level is 'gene', convert ensembl_gene_ids
            to gene names in the plot legend. If False or level is
            'variant', use the original keys.
        **kwargs
            Additional keyword arguments passed to
            plot_posteriors_vs_counts (e.g., save, show,
            max_shift_x).

        Returns
        -------
        None

        Examples
        --------
        >>> # Plot all gamma results
        >>> model.plot_gamma_results()
        >>>
        >>> # Plot specific variants
        >>> model.plot_gamma_results(
        ...     keys=['KRAS p.G12D', 'BRAF p.V600E'],
        ...     level='variant',
        ...     save='variant_selection.png',
        ...     show=True)
        >>>
        >>> # Plot specific genes by ensembl_gene_id
        >>> model.plot_gamma_results(
        ...     keys=['ENSG00000133703', 'ENSG00000157764'],
        ...     level='gene',
        ...     max_shift_x=250)
        >>>
        >>> # Plot genes with ensembl IDs kept in legend
        >>> model.plot_gamma_results(
        ...     keys=['ENSG00000133703'],
        ...     level='gene',
        ...     change_gene_ids_to_names=False)
        """
        from .estimate_presence import filter_passenger_genes
        from .figures import plot_posteriors_vs_counts

        if not self.gammas:
            raise ValueError(
                "No gamma results to plot. "
                "Call estimate_gamma() first."
            )

        # Select results based on keys
        if keys is None:
            # Plot all results
            results = self.gammas
        elif isinstance(keys, str):
            # Single key
            if keys not in self.gammas:
                raise ValueError(
                    f"Key {keys!r} not found in gamma results."
                )
            results = {keys: self.gammas[keys]}
        else:
            # List of keys
            results = {}
            for key in keys:
                if key not in self.gammas:
                    raise ValueError(
                        f"Key {key!r} not found in gamma results."
                    )
                results[key] = self.gammas[key]

        # Auto-detect level if not specified
        if level is None:
            # Check first key to determine level
            first_key = next(iter(results.keys()))
            if (
                hasattr(self.dataset, "_variants_present")
                and self.dataset._variants_present is not None
                and first_key in self.dataset._variants_present.index
            ):
                level = "variant"
            elif (
                hasattr(self.dataset, "_genes_present")
                and self.dataset._genes_present is not None
                and first_key in self.dataset._genes_present.index
            ):
                level = "gene"
            else:
                raise ValueError(
                    f"Could not auto-detect level for key {first_key!r}. "
                    "Please specify level='variant' or level='gene'."
                )

        # Build counts dictionary from dataset
        if level == "variant":
            if self.dataset._variants_present is None:
                raise ValueError(
                    "Variant presence matrix not computed. "
                    "Call dataset.compute_variants_present() first."
                )
            variant_counts = self.dataset.variants_present.sum(
                axis=1
            ).astype(int)
            counts = {
                key: int(variant_counts.get(key, 0))
                for key in results
            }
            passenger_genes = set(
                filter_passenger_genes(self.dataset.mutation_db)
            )
            results_for_plot = results
        else:  # level == 'gene'
            if self.dataset._genes_present_non_silent is None:
                raise ValueError(
                    "Gene presence matrix not computed. "
                    "Call dataset.compute_gene_presence_non_silent() first."
                )
            gene_presence = self.dataset.genes_present_non_silent
            gene_counts = gene_presence.sum(axis=1).astype(int)

            mapping_df = (
                self.dataset.mutation_db[["ensembl_gene_id", "gene"]]
                .dropna()
                .drop_duplicates()
            )
            id_to_name = dict(
                zip(mapping_df["ensembl_gene_id"], mapping_df["gene"])
            )
            name_to_id = dict(
                zip(mapping_df["gene"], mapping_df["ensembl_gene_id"])
            )
            passenger_gene_names = set(
                filter_passenger_genes(self.dataset.mutation_db)
            )

            results_for_plot = {}
            counts = {}
            passenger_genes = set()

            for key, idata in results.items():
                gene_id = key
                if gene_id not in gene_counts.index:
                    gene_id = name_to_id.get(gene_id)
                if gene_id is None:
                    raise ValueError(
                        f"Gene key {key!r} not found in dataset. "
                        "Ensure gamma results were generated for valid "
                        "Ensembl IDs or gene symbols."
                    )

                if change_gene_ids_to_names:
                    base_label = id_to_name.get(gene_id, gene_id)
                    label = base_label
                    if label in results_for_plot:
                        label = f"{base_label} ({gene_id})"
                else:
                    label = gene_id

                results_for_plot[label] = idata
                counts[label] = int(gene_counts.get(gene_id, 0))

                gene_name = id_to_name.get(gene_id, gene_id)
                if gene_name in passenger_gene_names:
                    passenger_genes.add(label)

        plot_posteriors_vs_counts(
            results_for_plot,
            counts,
            passenger_genes,
            level=level,
            **kwargs,
        )

    def plot_signature_correlations(
        self,
        top_n=None,
        figsize=None,
        mutations_log_scale=False,
        save_path=None,
        show=True,
    ):
        """Plot signature correlations with covariates.

        Visualizes correlations between mutational signatures and
        provided covariates using bars and mutation counts.

        Parameters
        ----------
        top_n : int, optional
            If provided, only plot the top N signatures by
            mutation count. Default None (plot all).
        figsize : tuple, optional
            Figure size. Default None.
        mutations_log_scale : bool, optional
            If True, use log scale for mutations y-axis.
            Default False.
        save_path : str or Path, optional
            Path to save the figure. If None, doesn't save.
        show : bool, optional
            Whether to display the figure. Default True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If covariates are not set or required data is missing.

        Examples
        --------
        >>> # Plot all signature correlations
        >>> model.plot_signature_correlations(show=True)
        >>>
        >>> # Plot top 5 signatures only
        >>> model.plot_signature_correlations(
        ...     top_n=5,
        ...     save_path='sig_corr.png')
        >>>
        >>> # Use log scale for mutations
        >>> model.plot_signature_correlations(
        ...     mutations_log_scale=True,
        ...     figsize=(12, 8))
        """
        from pathlib import Path

        from .figures import plot_signature_correlations

        # Check that covariates are set
        if self.cov_matrix is None:
            raise ValueError(
                "No covariates set. "
                "Create model with cov_matrix parameter."
            )

        # Get L_low and L_high from auto kwargs (whatever compute_mu_taus
        # was actually run with -- None if no correction was applied)
        L_low = self._auto_mu_taus_kwargs.get("L_low")
        L_high = self._auto_mu_taus_kwargs.get("L_high")

        # Build signature matrix path
        location_sig_matrix_norm = (
            Path(self.dataset.location_maf_files)
            / "signature_decomposition"
            / self.dataset.signature_class
            / "Assignment_Solution"
            / "Signatures"
            / "Assignment_Solution_Signatures.txt"
        )

        # Convert cov_matrix DataFrame to dict of Series
        cov_matrix_for_corr = {
            col: self.cov_matrix[col]
            for col in self.cov_matrix.columns
        }

        # Call the plotting function
        plot_signature_correlations(
            db=self.dataset.mutation_db,
            assignments=self.dataset.sig_assignments,
            location_sig_matrix_norm=location_sig_matrix_norm,
            L_low=L_low,
            L_high=L_high,
            cov_matrix_for_corr=cov_matrix_for_corr,
            top_n=top_n,
            figsize=figsize,
            mutations_log_scale=mutations_log_scale,
            save_path=save_path,
            show=show,
        )

    def _compute_mu_g_taus(self, use_cov_effects=True):
        """Compute per-gene, per-type, per-sample mutation rates.

        Mirrors :meth:`compute_mu_gs`, but keeps mutation type τ as
        a separate axis instead of summing over it: this is
        ``mu_{g,tau}^{j}``, the quantity :meth:`compute_mu_ms`
        needs (and divides by ``n_{g,c(tau)}``) to derive
        variant-level rates. Using the gene *total* (summed over
        all 96 types, as ``mu_gs`` is) there would be wrong --- it
        would divide an all-type total by a single type's context
        count.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from mutation type τ (from
            ``constants.canonical_types_order``) to a Genes ×
            Tumors DataFrame of ``mu_{g,tau}^{j}``, covariate-scaled
            when ``use_cov_effects`` and covariate effects are
            available (same scaling ``compute_mu_gs`` applies, just
            not yet summed over τ).
        """
        from .constants import canonical_types_order
        from .estimate_mus import (
            compute_mu_g_per_tumor,
            compute_mus_per_gene_per_sample,
        )

        signature_separated = isinstance(self._mu_taus, dict)

        # self._base_mus may have been aggregated (e.g. 30 raw
        # signatures -> 5 signature_selection groups, via
        # aggregate_signatures), while self._mu_taus stays
        # un-aggregated -- cov_effects is fit against the aggregated
        # groups, so it must be applied to per-tau mu_taus grouped
        # the same way. Summation commutes with the later gene/type
        # mixing, so aggregating raw mu_taus first is equivalent to
        # aggregating the derived base_mus, as aggregate_signatures
        # does.
        mu_taus_for_g_taus = self._mu_taus
        if (
            signature_separated
            and isinstance(self._base_mus, dict)
            and set(self._mu_taus.keys())
            != set(self._base_mus.keys())
        ):
            # _auto_signature_selection is a construction-time
            # convenience field and isn't restored by
            # load_model(), so prefer it when available (it may
            # carry tuple/'+' groupings) but fall back to
            # reconstructing the grouping from base_mus' own
            # keys, splitting any "sigA+sigB" name back into a
            # tuple. This matches what aggregate_signatures
            # would have been called with, since base_mus'
            # keys *are* the resulting group names. The synthetic
            # "other" key (from aggregate_signatures'
            # include_other=True) has no corresponding raw
            # signature name to split back out of -- reconstruct
            # it via _aggregate_signature_dict's own
            # include_other mechanism instead of naming it
            # explicitly in the selection.
            if self._auto_signature_selection is not None:
                selection = self._auto_signature_selection
                include_other = self._auto_include_other
            else:
                selection = [
                    tuple(key.split("+")) if "+" in key else key
                    for key in self._base_mus
                    if key != "other"
                ]
                include_other = "other" in self._base_mus
            mu_taus_for_g_taus = _aggregate_signature_dict(
                self._mu_taus, selection, include_other=include_other
            )

        use_cov = (
            use_cov_effects
            and self.cov_effects is not None
            and self.cov_matrix is not None
        )

        # Computed one tau at a time (not separate_per_tau=True for
        # all 96 at once): a full genes x tumors x 96-types x
        # n_signatures tensor can be tens of GB for realistic gene
        # and sample counts, so each tau's baseline is built,
        # covariate-scaled, and freed before moving to the next.
        result = {}
        for tau in canonical_types_order:
            base_g_tau = compute_mu_g_per_tumor(
                mu_taus=mu_taus_for_g_taus,
                contexts_by_gene=self.dataset.contexts_by_gene,
                prob_g_tau_tau_independent=(
                    self.prob_g_tau_tau_independent
                ),
                separate_per_tau=[tau],
            )

            if signature_separated:
                baseline_tau = {
                    sigma: base_g_tau[sigma][tau]
                    for sigma in base_g_tau
                }
                baseline_total = sum(baseline_tau.values())
            else:
                baseline_tau = base_g_tau[tau]
                baseline_total = baseline_tau

            if not use_cov:
                result[tau] = baseline_total
                continue

            scaled = compute_mus_per_gene_per_sample(
                db=self.dataset.mutation_db,
                base_mus=baseline_tau,
                cov_effect=self.cov_effects,
                cov_matrix=self.cov_matrix,
            )

            # Genes without covariate coverage keep their baseline
            # rate, mirroring compute_mu_gs's
            # assign_base_mus_to_rest behavior.
            missing_genes = baseline_total.index.difference(
                scaled.index
            )
            if missing_genes.any():
                scaled = pd.concat(
                    [scaled, baseline_total.loc[missing_genes]]
                )

            result[tau] = scaled

        return result

    def compute_mu_ms(self, use_cov_effects=True, **kwargs):
        """Compute per-variant mutation rates per sample.

        Wraps :func:`estimate_mus.compute_mu_m_per_tumor` to convert
        gene-level mutation rates and variant annotations into
        variant-level expectations. Results are stored in `self.mu_ms`.

        .. note::
            This method is automatically called by
            :meth:`estimate_cov_effects` after estimating covariate
            effects. You typically do not need to call this manually
            unless you want to recompute variant rates with different
            parameters.

        Parameters
        ----------
        use_cov_effects : bool, optional
            If True (default) and covariate effects have been estimated,
            use mutation rates that include covariate effects (mu_gs).
            If False or if cov_effects is None, use baseline rates
            (base_mus) without covariate adjustments. Default True.
        **kwargs : dict
            Additional keyword arguments forwarded to
            :func:`estimate_mus.compute_mu_m_per_tumor`. Refer to that
            function for supported options (e.g., float_type).

        Returns
        -------
        pd.DataFrame
            Variants × tumors mutation rate matrix.

        See Also
        --------
        estimate_cov_effects : Automatically calls this method after
            estimating covariate effects
        compute_mu_gs : Computes gene-level mutation rates
        estimate_mus.compute_mu_m_per_tumor : Underlying computation
        """
        from .estimate_mus import compute_mu_m_per_tumor

        if self.dataset._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded in dataset. "
                "Call dataset.generate_mutation_db() or "
                "dataset.load_dataset() first."
            )

        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first."
            )

        if self.dataset._variant_db is None:
            raise ValueError(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first."
            )

        if self._prob_g_tau_tau_independent is None:
            raise ValueError(
                "prob_g_tau_tau_independent flag not set. "
                "Call compute_base_mus() first."
            )

        # Per-type gene rates: mu_ms needs the type-tau-specific share
        # of each gene's rate, not the gene total (self.mu_gs / self.
        # base_mus), so that dividing by n_{g,c(tau)} is correct.
        mu_g_tau_j = self._compute_mu_g_taus(
            use_cov_effects=use_cov_effects
        )

        self.mu_ms = compute_mu_m_per_tumor(
            variants_df=self.dataset.variant_db,
            mu_g_tau_j=mu_g_tau_j,
            contexts_by_gene=self.dataset.contexts_by_gene,
            prob_g_tau_tau_independent=self.prob_g_tau_tau_independent,
            **kwargs,
        )

        return self.mu_ms

    def save_model(self, directory, overwrite=False):
        """Persist this Model's results to disk.

        Saves all model components to the specified directory,
        creating a structured snapshot that can be reloaded with
        :meth:`load_model`.

        Parameters
        ----------
        directory : str or Path
            Directory path where model will be saved.
        overwrite : bool, default False
            If True, overwrite existing model at this location.
            If False, raises FileExistsError if directory exists.

        Directory Structure
        -------------------
        The saved model creates the following structure::

            directory/
            ├── model_manifest.json       # Metadata and file registry
            ├── cov_matrix.parquet        # Covariate matrix
            ├── mu_taus.parquet           # Per-type mutation rates
            │   └── (or mu_taus/*.parquet for multi-signature)
            ├── base_mus.parquet          # Baseline gene rates
            │   └── (or base_mus/*.parquet for multi-signature)
            ├── cov_effects.npy           # Coefficient estimates
            ├── cov_effects_posteriors.nc # MCMC posterior (if available)
            ├── mu_gs.parquet             # Gene rates with covariates
            ├── mu_ms.parquet             # Variant rates with covariates
            └── gammas/                   # Selection coefficients
                ├── gamma_{variant}.nc    # Per-variant posteriors
                └── gamma_{gene_id}.nc    # Per-gene posteriors

        Notes
        -----
        - Gamma results are saved as individual NetCDF files, one per
          variant or gene, in the ``gammas/`` subdirectory
        - File names for gammas use underscores instead of spaces
          (e.g., "BRAF p.V600E" → "gamma_BRAF_p.V600E.nc")
        - The manifest tracks all saved files and their locations
        - Multi-signature models save separate parquet files per
          signature in subdirectories

        See Also
        --------
        load_model : Reload a saved model from disk
        """
        from pathlib import Path

        directory = Path(directory)
        manifest_path = directory / "model_manifest.json"

        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Model directory {directory} already exists. "
                "Pass overwrite=True to replace it."
            )

        directory.mkdir(parents=True, exist_ok=True)

        def _save_dataframe(df, filename):
            path = directory / filename
            if df.attrs:
                df = df.copy()
                df.attrs = {}
            df.to_parquet(path)
            return filename

        def _save_dict_of_dataframes(data, folder):
            folder_path = directory / folder
            folder_path.mkdir(exist_ok=True)
            stored = {}
            for key, df in data.items():
                stored[key] = _save_dataframe(
                    df, f"{folder}/{key}.parquet"
                )
            return stored

        def _save_array(arr, filename):
            path = directory / filename
            np.save(path, arr)
            return filename

        def _json_safe(value):
            """Convert objects to JSON-serializable representations."""
            import numpy as np
            import pandas as pd

            # Handle None, bool, int, float, str - already JSON-safe
            if value is None or isinstance(
                value, (bool, int, float, str)
            ):
                return value

            # Handle numpy types first (most specific)
            if isinstance(value, np.integer):
                return int(value)
            if isinstance(value, np.floating):
                return float(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()

            # Handle pandas types
            if isinstance(value, (pd.Series, pd.Index)):
                return value.tolist()
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="list")

            # Handle collections recursively
            if isinstance(value, dict):
                return {
                    _json_safe(k): _json_safe(v)
                    for k, v in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if isinstance(value, set):
                return [_json_safe(v) for v in value]

            # Try common conversion methods. Each candidate method may
            # exist (hasattr) but still raise for this particular
            # value (e.g. an incompatible dtype); silently falling
            # through to the next candidate -- and ultimately to
            # str(value) below -- is the intended best-effort
            # behavior here, not an error to surface.
            if hasattr(value, "tolist"):
                try:
                    return value.tolist()
                except Exception:  # noqa: BLE001, S110
                    pass
            if hasattr(value, "to_dict"):
                try:
                    return _json_safe(value.to_dict())
                except Exception:  # noqa: BLE001, S110
                    pass
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:  # noqa: BLE001, S110
                    pass

            # Last resort: convert to string
            return str(value)

        files = {}

        if self.cov_matrix is not None:
            files["cov_matrix"] = _save_dataframe(
                self.cov_matrix, "cov_matrix.parquet"
            )

        if self._mu_taus is not None:
            if isinstance(self._mu_taus, dict):
                files["mu_taus"] = _save_dict_of_dataframes(
                    self._mu_taus, "mu_taus"
                )
            else:
                files["mu_taus"] = _save_dataframe(
                    self._mu_taus, "mu_taus.parquet"
                )

        if self._base_mus is not None:
            if isinstance(self._base_mus, dict):
                files["base_mus"] = _save_dict_of_dataframes(
                    self._base_mus, "base_mus"
                )
            else:
                files["base_mus"] = _save_dataframe(
                    self._base_mus, "base_mus.parquet"
                )

        if self.cov_effects is not None:
            files["cov_effects"] = _save_array(
                self.cov_effects, "cov_effects.npy"
            )

        if self.cov_effects_posteriors is not None:
            posterior = self.cov_effects_posteriors
            if hasattr(posterior, "to_netcdf"):
                filename = directory / "cov_effects_posteriors.nc"
                posterior.to_netcdf(filename)
                files["cov_effects_posteriors"] = (
                    "cov_effects_posteriors.nc"
                )

        if self._mu_gs is not None:
            files["mu_gs"] = _save_dataframe(
                self._mu_gs, "mu_gs.parquet"
            )

        if self.mu_ms is not None:
            files["mu_ms"] = _save_dataframe(
                self.mu_ms, "mu_ms.parquet"
            )

        if self.gammas:
            gammas_dir = directory / "gammas"
            gammas_dir.mkdir(exist_ok=True)
            gamma_files = {}

            for key, result in self.gammas.items():
                # Create safe filename by replacing problematic chars
                safe_key = (
                    str(key)
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("\\", "_")
                    .replace(":", "_")
                )
                filename = f"gamma_{safe_key}.nc"
                filepath = gammas_dir / filename

                if (
                    filepath.exists()
                    and self._gammas_loaded_from_disk.get(key)
                    is result
                ):
                    # This exact object is the one load_model() read
                    # from this file earlier this session -- content
                    # is guaranteed identical, and re-opening it for
                    # write can collide with a lingering cached read
                    # handle from xarray's file manager and raise a
                    # spurious PermissionError. A file merely
                    # *existing* at this path is not enough: if the
                    # in-memory result was recomputed (e.g. the key
                    # was deleted from self.gammas and re-estimated),
                    # it must overwrite the stale file, not skip it.
                    gamma_files[key] = f"gammas/{filename}"
                    continue

                # Save as NetCDF if possible. Write to a temporary
                # path and rename into place rather than writing
                # `filepath` directly: if this key was previously
                # loaded from this exact path via load_model() and
                # then recomputed, xarray's file manager may still
                # hold a cached read handle open on `filepath`, and
                # writing to it directly can raise a spurious
                # PermissionError (netCDF4/HDF5). Writing a fresh
                # path and swapping it in avoids that collision
                # regardless of any lingering handle.
                if hasattr(result, "to_netcdf"):
                    tmp_filepath = filepath.with_suffix(".nc.tmp")
                    result.to_netcdf(tmp_filepath)
                    tmp_filepath.replace(filepath)
                    gamma_files[key] = f"gammas/{filename}"
                else:
                    logger.warning(
                        f"Gamma result for {key!r} does not have "
                        f"to_netcdf method. Skipping save."
                    )

            files["gamma_files"] = gamma_files

        # The r_g fit is not recoverable from the saved rate
        # matrices: theta, the per-channel intercept and the
        # sufficient statistics all live outside them. Without
        # persisting them a reloaded model silently drops the
        # channel corrections instead of failing, so they are saved
        # whenever a fit is present.
        if self._rg_theta is not None:
            if self._base_mus_syn is not None:
                files["base_mus_syn"] = _save_dataframe(
                    self._base_mus_syn, "base_mus_syn.parquet"
                )
            if self._base_mus_nonsyn is not None:
                files["base_mus_nonsyn"] = _save_dataframe(
                    self._base_mus_nonsyn, "base_mus_nonsyn.parquet"
                )
            if self._channel_cov_effects is not None:
                files["channel_cov_effects"] = _save_array(
                    self._channel_cov_effects,
                    "channel_cov_effects.npy",
                )
            if self._rg_statistics is not None:
                stats = self._rg_statistics
                frame = pd.DataFrame(
                    {
                        "counts_silent": stats["counts_silent"],
                        "counts_non_silent": stats[
                            "counts_non_silent"
                        ],
                        "baseline_silent": stats["baseline_silent"],
                        "baseline_non_silent": stats[
                            "baseline_non_silent"
                        ],
                    }
                )
                frame["in_non_silent"] = frame.index.isin(
                    stats["in_non_silent"]
                )
                files["rg_statistics"] = _save_dataframe(
                    frame, "rg_statistics.parquet"
                )

        dataset_snapshot = getattr(
            self.dataset, "dataset_directory", None
        )
        if dataset_snapshot is not None:
            dataset_snapshot = str(Path(dataset_snapshot).resolve())
        else:
            logger.warning(
                "Dataset does not have an associated saved directory. "
                "Model snapshots will be unable to reload the dataset. "
                "Call MutationDataset.save_dataset() and reload it "
                "before saving models."
            )

        manifest = {
            "version": 1,
            "dataset_snapshot": dataset_snapshot,
            "dataset_location": getattr(
                self.dataset, "location_maf_files", None
            ),
            "covariate_names": self.covariate_names,
            "mu_taus_separate": isinstance(self._mu_taus, dict),
            "rg_theta": self._rg_theta,
            "rg_delta_intercept": self._rg_delta_intercept,
            "rg_separate_c": self._rg_separate_c,
            "prob_g_tau_tau_independent": (
                self._prob_g_tau_tau_independent
            ),
            "files": files,
        }

        manifest_path.write_text(
            json.dumps(_json_safe(manifest), indent=2)
        )
        self._saved_location = str(directory.resolve())

    @classmethod
    def load_model(cls, directory):
        """Load a Model from a directory created by save_model()."""
        from pathlib import Path

        directory = Path(directory)
        manifest_path = directory / "model_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found at {manifest_path}."
            )

        manifest = json.loads(manifest_path.read_text())

        dataset_snapshot = manifest.get("dataset_snapshot")
        if not dataset_snapshot:
            raise ValueError(
                "Model manifest does not include a dataset_snapshot "
                "entry. Recreate the model by loading the appropriate "
                "MutationDataset and re-saving the model snapshot."
            )

        snapshot_path = Path(dataset_snapshot)
        if not snapshot_path.is_absolute():
            candidate = (directory / snapshot_path).resolve()
            if candidate.exists():
                snapshot_path = candidate

        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Dataset snapshot not found at {snapshot_path}. "
                "Ensure the dataset directory still exists or "
                "recreate it with MutationDataset.save_dataset()."
            )

        model = cls(
            dataset=MutationDataset.load_dataset(snapshot_path),
            cov_matrix=None,
        )

        files = manifest.get("files", {})

        def _load_dataframe(filename):
            return pd.read_parquet(directory / filename)

        if "cov_matrix" in files:
            model.cov_matrix = _load_dataframe(files["cov_matrix"])

        if "mu_taus" in files:
            mu_info = files["mu_taus"]
            if isinstance(mu_info, dict):
                model._mu_taus = {
                    sig: _load_dataframe(path)
                    for sig, path in mu_info.items()
                }
            else:
                model._mu_taus = _load_dataframe(mu_info)

        if "base_mus" in files:
            base_info = files["base_mus"]
            if isinstance(base_info, dict):
                model._base_mus = {
                    sig: _load_dataframe(path)
                    for sig, path in base_info.items()
                }
            else:
                model._base_mus = _load_dataframe(base_info)

        if "cov_effects" in files:
            model.cov_effects = np.load(
                directory / files["cov_effects"]
            )

        if "mu_gs" in files:
            model._mu_gs = _load_dataframe(files["mu_gs"])

        if "mu_ms" in files:
            model.mu_ms = _load_dataframe(files["mu_ms"])

        # Load gamma results (new format: individual .nc files)
        if "gamma_files" in files:
            import arviz as az

            model.gammas = {}
            for key, filepath in files["gamma_files"].items():
                full_path = directory / filepath
                if full_path.exists():
                    try:
                        model.gammas[key] = az.from_netcdf(full_path)
                        model._gammas_loaded_from_disk[key] = (
                            model.gammas[key]
                        )
                    # One corrupt/unreadable gamma file shouldn't
                    # abort loading the rest of the model -- skip it
                    # and warn, same pattern as MAF batch processing.
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            f"Failed to load gamma result for {key!r} "
                            f"from {filepath}: {e}"
                        )
                else:
                    logger.warning(
                        f"Gamma file not found: {filepath}"
                    )

        # Backward compatibility: load old JSON format
        elif "gammas" in files:
            path = directory / files["gammas"]
            model.gammas = json.loads(path.read_text())
        elif "gamma_ms" in files or "gamma_gs" in files:
            model.gammas = {}
            if "gamma_ms" in files:
                path = directory / files["gamma_ms"]
                model.gammas.update(json.loads(path.read_text()))
            if "gamma_gs" in files:
                path = directory / files["gamma_gs"]
                model.gammas.update(json.loads(path.read_text()))

        model._rg_theta = manifest.get("rg_theta")
        model._rg_delta_intercept = manifest.get("rg_delta_intercept")
        model._rg_separate_c = manifest.get("rg_separate_c", False)
        if "base_mus_syn" in files:
            model._base_mus_syn = _load_dataframe(
                files["base_mus_syn"]
            )
        if "base_mus_nonsyn" in files:
            model._base_mus_nonsyn = _load_dataframe(
                files["base_mus_nonsyn"]
            )
        if "channel_cov_effects" in files:
            model._channel_cov_effects = np.load(
                directory / files["channel_cov_effects"]
            )
        if "rg_statistics" in files:
            frame = _load_dataframe(files["rg_statistics"])
            model._rg_statistics = {
                "genes": frame.index,
                "counts_silent": frame["counts_silent"],
                "counts_non_silent": frame["counts_non_silent"],
                "baseline_silent": frame["baseline_silent"],
                "baseline_non_silent": frame["baseline_non_silent"],
                "in_non_silent": frame.index[frame["in_non_silent"]],
            }

        model._prob_g_tau_tau_independent = manifest.get(
            "prob_g_tau_tau_independent"
        )
        model._saved_location = str(directory.resolve())

        return model

    @property
    def saved_location(self):
        """Directory where the model snapshot is stored."""
        if self._saved_location is None:
            raise ValueError(
                "Model has not been saved yet. "
                "Call save_model() or load_model() first."
            )
        return self._saved_location

    @property
    def mu_gs(self):
        """Per-gene, per-sample mutation rates (lazy loaded)."""
        if self._mu_gs is None:
            raise ValueError(
                "Mutation rates not computed. "
                "Call compute_mu_gs() first."
            )
        return self._mu_gs

    @mu_gs.setter
    def mu_gs(self, value):
        """Set per-gene, per-sample mutation rates.

        Validates that the gene index matches contexts_by_gene
        and cov_matrix (if present).
        """

        # Allow None
        if value is None:
            self._mu_gs = value
            return

        # Check if contexts_by_gene is loaded
        if self.dataset._contexts_by_gene is None:
            logger.warning(
                "Cannot validate mu_gs index: "
                "contexts_by_gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() to ensure gene indices match."
            )
            self._mu_gs = value
            return

        # Validate index for DataFrames
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(
                self.dataset.contexts_by_gene.index
            ):
                logger.warning(
                    "mu_gs index does not match "
                    "dataset.contexts_by_gene.index. "
                    "This may cause errors in downstream analysis. "
                    "To fix: reindex mu_gs to match "
                    "contexts_by_gene.index, or recompute mu_gs "
                    "after loading contexts_by_gene."
                )

            # Also check against cov_matrix if present
            if (
                self.cov_matrix is not None
                and not value.index.equals(self.cov_matrix.index)
            ):
                logger.warning(
                    "mu_gs index does not match cov_matrix.index. "
                    "This may cause errors in downstream analysis. "
                    "To fix: call model.assign_cov_matrix() again "
                    "to reindex cov_matrix to match "
                    "contexts_by_gene.index, then recompute mu_gs."
                )

        # Validate index for dicts of DataFrames (signature-separated)
        elif isinstance(value, dict):
            for sig_name, df in value.items():
                if isinstance(df, pd.DataFrame):
                    if not df.index.equals(
                        self.dataset.contexts_by_gene.index
                    ):
                        logger.warning(
                            "mu_gs['%s'] index does not match "
                            "dataset.contexts_by_gene.index. "
                            "This may cause errors in downstream "
                            "analysis. To fix: reindex all mu_gs "
                            "DataFrames to match contexts_by_gene.index, "
                            "or recompute mu_gs after loading "
                            "contexts_by_gene.",
                            sig_name,
                        )
                        break  # Only warn once for contexts_by_gene

                    # Also check against cov_matrix if present
                    if (
                        self.cov_matrix is not None
                        and not df.index.equals(self.cov_matrix.index)
                    ):
                        logger.warning(
                            "mu_gs['%s'] index does not "
                            "match cov_matrix.index. "
                            "This may cause errors in downstream "
                            "analysis. To fix: call "
                            "model.assign_cov_matrix() again to "
                            "reindex cov_matrix to match "
                            "contexts_by_gene.index, then recompute "
                            "mu_gs.",
                            sig_name,
                        )
                        break  # Only warn once

        self._mu_gs = value

    def has_mu_taus(self):
        """Check if mutation burdens have been computed."""
        return self._mu_taus is not None

    def copy(self):
        """Create a copy of the model with shared dataset and base results.

        This method creates a new Model instance that shares the dataset
        reference and large computed base results (_base_mus, _mu_taus)
        with the original model, while creating independent copies of
        configuration attributes and resetting model-specific results.

        This is useful for creating multiple models with different
        covariate matrices from a common base model.

        Returns
        -------
        Model
            New Model instance with:
            - Shared: dataset, _base_mus, _mu_taus (memory efficient)
            - Copied: cov_matrix, cov_effects_kwargs
            - Reset: cov_effects, passenger_genes_r2,
              cov_effects_posteriors, mu_gs, mu_ms (model-specific
              results)

        Notes
        -----
        **Memory management:**

        The copy is shallow for large objects that are dataset-dependent
        and can be safely shared:
        - dataset: MutationDataset reference (not copied)
        - _base_mus: Baseline mutation rates (shared, can be very large)
        - _mu_taus: Mutation burdens (shared, can be very large)

        The copy is deep for small configuration objects:
        - cov_matrix: Covariate matrix (copied if not None)
        - cov_effects_kwargs: MAP estimation parameters (deep copied)

        Model-specific results are reset to None since they depend on
        the covariate matrix and need to be recomputed for the new model:
        - cov_effects, passenger_genes_r2, cov_effects_posteriors
        - mus (mu_gs), mu_ms

        **Typical workflow:**

        1. Create and populate a base model with shared computations:
            >>> base_model = Model(dataset)
            >>> base_model.compute_mu_taus()
            >>> base_model.compute_base_mus()

        2. Create copies with different covariate matrices:
            >>> model_mrt = base_model.copy()
            >>> model_mrt.cov_matrix = cov_matrix[['mrt']]
            >>>
            >>> model_gexp = base_model.copy()
            >>> model_gexp.cov_matrix = cov_matrix[['log1p_gexp']]
            >>>
            >>> model_full = base_model.copy()
            >>> model_full.cov_matrix = cov_matrix[['mrt', 'log1p_gexp']]

        3. Each model can then independently estimate covariate effects:
            >>> model_mrt.estimate_cov_effects()
            >>> model_gexp.estimate_cov_effects()
            >>> model_full.estimate_cov_effects()

        Examples
        --------
        >>> # Create base model with shared computations
        >>> dataset = MutationDataset(location_maf_files,
                                      signature_class="SBS")
        >>> dataset.generate_mutation_db()
        >>> dataset.run_signature_decomposition()
        >>> dataset.generate_contexts_by_gene(fastas)
        >>>
        >>> base_model = Model(dataset)
        >>> base_model.compute_mu_taus()
        >>> base_model.compute_base_mus()
        >>>
        >>> # Create multiple models with different covariates
        >>> model1 = base_model.copy()
        >>> model1.cov_matrix = cov_matrix_full[['mrt']]
        >>>
        >>> model2 = base_model.copy()
        >>> model2.cov_matrix = cov_matrix_full[['log1p_gexp', 'log1p_atac']]
        >>>
        >>> # Models share base_mus and mu_taus (memory efficient)
        >>> assert model1._base_mus is base_model._base_mus
        >>> assert model1._mu_taus is base_model._mu_taus
        >>> assert model1.dataset is base_model.dataset
        >>>
        >>> # But have independent covariate matrices
        >>> assert model1.cov_matrix is not model2.cov_matrix

        See Also
        --------
        compute_mu_taus : Compute mutation burdens (shared across copies)
        compute_base_mus : Compute baseline mutation rates (shared)

        """
        import copy as copy_module

        # Create new Model instance with shared dataset
        new_model = Model(
            dataset=self.dataset,
            cov_matrix=(
                self.cov_matrix.copy()
                if self.cov_matrix is not None
                else None
            ),
            cov_effects_kwargs=copy_module.deepcopy(
                self.cov_effects_kwargs
            ),
        )

        # Share large base results (memory efficient)
        new_model._base_mus = self._base_mus  # Share, don't copy
        new_model._mu_taus = self._mu_taus  # Share, don't copy
        new_model._prob_g_tau_tau_independent = (
            self._prob_g_tau_tau_independent
        )

        # Model-specific results are left as None (default)
        # These will be recomputed for the new covariate matrix:
        # - cov_effects, passenger_genes_r2, cov_effects_posteriors
        # - mu_gs, mu_ms

        return new_model

    @property
    def mu_taus(self):
        """Mutation burden per tumor (lazy loaded)."""
        if self._mu_taus is None:
            raise ValueError(
                "Mutation burdens not computed. "
                "Call compute_mu_taus() first."
            )
        return self._mu_taus

    @mu_taus.setter
    def mu_taus(self, value):
        """Set mutation burdens per tumor."""
        self._mu_taus = value

    def compute_mu_taus(self, separate_per_sigma=False, **kwargs):
        """Compute mutation burden (total mutations) per tumor.

        This method estimates the baseline mutation rate per tumor
        per mutation type (μ_τ^(j)), incorporating signature
        exposures and mutation burden estimates. It wraps
        :func:`estimate_mus.compute_mu_tau_per_tumor`.

        The mutation burden represents the expected total number
        of mutations of each type in each tumor, without considering
        gene-specific covariate effects.

        The normalized signature matrix is automatically loaded from
        the signature decomposition results.

        Parameters
        ----------
        separate_per_sigma : bool, default False
            Whether to return signature-separated mutation burdens:
            - False: Returns single DataFrame with total mutation
              burden summed across all signatures
            - True: Returns dict mapping each signature to its
              contribution to the mutation burden

            When True, the model will have signature-dependent
            covariate effects (one set of effects per signature).
            When False, covariate effects are signature-independent.
        **kwargs : dict
            Additional arguments passed to
            :func:`estimate_mus.compute_mu_tau_per_tumor`:
            - L_low : float or None, optional (default None)
                Lower burden threshold for correcting low-burden
                samples. None (the default) applies no correction --
                each sample's own raw alpha/burden is used as-is.
            - L_high : float or None, optional (default None)
                Upper burden threshold for intermediate-burden
                correction. No effect if L_low is None.
            - cut_at_L_low : bool, default False
                Whether to hard clip burden estimates at L_low

        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
            When separate_per_sigma=False:
                DataFrame with tumor samples as index, mutation
                types as columns, and total mutation burden values.

            When separate_per_sigma=True:
                Dictionary mapping signature names to DataFrames.
                Each DataFrame has the same structure (tumors ×
                mutation types) but contains only that signature's
                contribution.

        Raises
        ------
        ValueError
            If mutation database is not loaded in the dataset.
        ValueError
            If signature decomposition has not been run. Call
            dataset.run_signature_decomposition() first.

        Notes
        -----
        **Signature-dependent vs. signature-independent models:**

        The `separate_per_sigma` parameter determines whether
        covariate effects will be estimated separately for each
        signature:

        - separate_per_sigma=False: Signature-independent model
            - Single set of covariate effects for all mutation types
            - Faster computation, fewer parameters
            - Assumes covariate effects are the same across
              signatures

        - separate_per_sigma=True: Signature-dependent model
            - Separate covariate effects for each signature
            - More flexible, can capture signature-specific
              covariate relationships
            - Requires more data and computation time

        The mutation database must be loaded before calling this
        method (e.g., via dataset.generate_mutation_db() or
        dataset.load_dataset()).

        Signature decomposition must also be run (via
        dataset.run_signature_decomposition()), which will load
        both the signature assignments and the normalized signature
        matrix.

        Examples
        --------
        >>> # Signature-independent model
        >>> model.compute_mu_taus(separate_per_sigma=False)
        >>> print(model.mu_taus.shape)  # (n_tumors, n_types)
        >>>
        >>> # Signature-dependent model
        >>> model.compute_mu_taus(separate_per_sigma=True)
        >>> print(type(model.mu_taus))  # dict
        >>> for sig_name, mu_df in model.mu_taus.items():
        ...     print(f"{sig_name}: {mu_df.shape}")
        >>>
        >>> # With burden correction for low-count samples
        >>> model.compute_mu_taus(
        ...     separate_per_sigma=False,
        ...     L_low=50,
        ...     L_high=200)

        See Also
        --------
        estimate_mus.compute_mu_tau_per_tumor : Core computation
        """
        import tempfile
        from pathlib import Path

        from .estimate_mus import compute_mu_tau_per_tumor

        # Ensure mutation_db is loaded
        if self.dataset._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded in dataset. "
                "Call dataset.generate_mutation_db() or "
                "dataset.load_dataset() first."
            )

        # Ensure signature decomposition has been run
        if self.dataset._sig_assignments is None:
            raise ValueError(
                "Signature decomposition not run. "
                "Call dataset.run_signature_decomposition() first."
            )

        if self.dataset._signature_matrix is None:
            raise ValueError(
                "Signature matrix not loaded. "
                "Call dataset.run_signature_decomposition() first."
            )

        # Write signature matrix to temporary file for compute_mu_tau_per_tumor
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp_file:
            self.dataset.signature_matrix.to_csv(
                tmp_file.name, sep="\t"
            )
            tmp_path = tmp_file.name

        try:
            l_low = kwargs.get("L_low")
            l_high = kwargs.get("L_high")

            if "L_low" not in kwargs:
                logger.warning(
                    "L_low was not provided; no low-burden correction "
                    "will be applied (each sample's own raw alpha/"
                    "burden is used as-is). Pass L_low=<value> to "
                    "blend low-burden samples' estimates toward a "
                    "population average instead."
                )

            compute_kwargs = kwargs.copy()
            compute_kwargs["L_low"] = l_low
            compute_kwargs["L_high"] = l_high

            # Store L_low and L_high for use by other methods
            self._auto_mu_taus_kwargs["L_low"] = l_low
            self._auto_mu_taus_kwargs["L_high"] = l_high

            # Compute mutation burdens
            self._mu_taus = compute_mu_tau_per_tumor(
                db=self.dataset.mutation_db,
                location_signature_matrix=tmp_path,
                assignments=self.dataset.sig_assignments,
                separate_per_sigma=separate_per_sigma,
                **compute_kwargs,
            )
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)

        return self._mu_taus

    def compute_base_mus(self, prob_g_tau_tau_independent=False):
        """Compute baseline mutation rates per gene per tumor.

        This method computes the expected mutation rate for each gene
        in each tumor by combining per-tumor mutation burdens (mu_taus)
        with gene-level trinucleotide context opportunities. It wraps
        :func:`estimate_mus.compute_mu_g_per_tumor`.

        The baseline mutation rates represent the expected number of
        mutations per gene per tumor WITHOUT considering covariate
        effects. These rates serve as the starting point for models
        that incorporate gene-level covariates.

        Parameters
        ----------
        prob_g_tau_tau_independent : bool, default False
            Determines how gene probabilities are calculated:

            - False (default): **Type-dependent gene probabilities**
                Uses trinucleotide context-specific opportunities to
                compute p(g | τ) separately for each mutation type τ.
                This accounts for the fact that different genes have
                different trinucleotide compositions, which affects
                their susceptibility to different mutation types.

                For each gene g and tumor j:
                    μ_g^(j) = Σ_τ μ_τ^(j) × p(g | τ)

                where p(g | τ) is the gene's share of opportunities
                for the trinucleotide context underlying mutation
                type τ.

                **Use this when:** You want to model how different
                mutational processes (signatures) affect genes
                differently based on their sequence composition.
                This is the more accurate and commonly used option.

            - True: **Type-independent gene probabilities**
                Uses total gene opportunities to compute a single
                p(g) that applies to all mutation types equally.
                This simplifies the model by assuming that a gene's
                mutation probability doesn't depend on the mutation
                type.

                For each gene g and tumor j:
                    μ_g^(j) = p(g) × Σ_τ μ_τ^(j)

                where p(g) = total_opportunities_g /
                             Σ_g' total_opportunities_g'

                **Use this when:** You want a simpler model or when
                dealing with non-SBS mutation types where
                context-specific opportunities aren't well-defined
                (e.g., indels, structural variants).

        Returns
        -------
        pd.DataFrame or dict[str, pd.DataFrame]
            When mu_taus is a DataFrame (signature-independent):
                Single DataFrame with genes as index, tumors as
                columns, and baseline mutation rates as values.
                Shape: (n_genes, n_tumors)

            When mu_taus is a dict (signature-separated):
                Dictionary mapping signature names to DataFrames.
                Each DataFrame has the same structure (genes × tumors)
                but contains only that signature's contribution to
                the baseline mutation rate.

        Raises
        ------
        ValueError
            If mu_taus have not been computed. Call
            compute_mu_taus() first.
        ValueError
            If contexts_by_gene are not loaded in the dataset. Call
            dataset.generate_contexts_by_gene() or load_dataset() first.

        Notes
        -----
        **Understanding prob_g_tau_tau_independent:**

        The choice of this parameter fundamentally affects how the
        model distributes mutation burden across genes:

        1. **Type-dependent (False, default):**
           - More biologically accurate for SBS mutations
           - Accounts for sequence composition effects
           - Different signatures affect genes differently
           - Example: A gene rich in C>T contexts will have higher
             baseline rates for signatures that predominantly cause
             C>T mutations

        2. **Type-independent (True):**
           - Simpler computation
           - Assumes uniform mutation susceptibility across types
           - May be more appropriate for non-SBS mutations
           - Example: All genes mutate proportionally to their
             total length/opportunity, regardless of mutation type

        **Workflow:**

        This method should be called after computing mu_taus:
            1. model.compute_mu_taus()
            2. model.compute_base_mus(prob_g_tau_tau_independent=False)
            3. Proceed with covariate effect estimation

        The mutation database and contexts_by_gene must be loaded
        in the dataset before calling this method.

        Examples
        --------
        >>> # Standard workflow with type-dependent gene probs
        >>> model = Model(dataset, cov_matrix)
        >>> model.compute_mu_taus()
        >>> model.compute_base_mus(prob_g_tau_tau_independent=False)
        >>> print(model.base_mus.shape)  # (n_genes, n_tumors)
        >>>
        >>> # Type-independent gene probabilities
        >>> model.compute_base_mus(prob_g_tau_tau_independent=True)
        >>>
        >>> # Signature-separated model
        >>> model_sig = Model(dataset, cov_matrix)
        >>> model_sig.compute_mu_taus(separate_per_sigma=True)
        >>> model_sig.compute_base_mus()
        >>> print(type(model_sig.base_mus))  # dict
        >>> for sig_name, mu_df in model_sig.base_mus.items():
        ...     print(f"{sig_name}: {mu_df.shape}")

        See Also
        --------
        estimate_mus.compute_mu_g_per_tumor : Core computation
        compute_mu_taus : Compute mutation burdens first
        """
        from .estimate_mus import compute_mu_g_per_tumor

        # Ensure mu_taus have been computed
        if self._mu_taus is None:
            raise ValueError(
                "Mutation burdens (mu_taus) not computed. "
                "Call compute_mu_taus() first."
            )

        # Ensure contexts_by_gene are loaded
        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first."
            )

        # Compute baseline mutation rates per gene
        self._base_mus = compute_mu_g_per_tumor(
            mu_taus=self._mu_taus,
            contexts_by_gene=self.dataset.contexts_by_gene,
            prob_g_tau_tau_independent=prob_g_tau_tau_independent,
        )

        self._prob_g_tau_tau_independent = prob_g_tau_tau_independent
        return self._base_mus

    def compute_channel_base_mus(
        self, prob_g_tau_tau_independent=None
    ):
        """Split baseline rates into synonymous/non-synonymous channels.

        Computes ``μ̄_g^(syn,j)`` and ``μ̄_g^(nonsyn,j)`` from the
        dataset's consequence-split opportunity tables (see
        :meth:`MutationDataset.generate_consequence_contexts_by_gene`)
        and stores them in ``self.base_mus_syn`` /
        ``self.base_mus_nonsyn``. ``compute_base_mus()``'s merged
        ``base_mus`` is left untouched, and the two channels sum back
        to it exactly.

        Parameters
        ----------
        prob_g_tau_tau_independent : bool or None, default None
            Which ``p_gτ`` variant to use. ``None`` (default) reuses
            whatever :meth:`compute_base_mus` last used, so the
            channels always match the merged baseline; pass an
            explicit value only if you deliberately want them to
            differ. Note that a τ-**independent** merged baseline
            propagates here and flattens the split: the resulting
            syn/non-syn ratio is then the same for every sample
            regardless of its spectrum. Prefer building the merged
            baseline τ-dependently when the channels are the point.

        Returns
        -------
        (pd.DataFrame, pd.DataFrame) or (dict, dict)
            ``(base_mus_syn, base_mus_nonsyn)``, in the same
            signature-independent or signature-separated shape as
            ``base_mus``.

        Raises
        ------
        ValueError
            If ``mu_taus``, the merged ``base_mus`` (when
            *prob_g_tau_tau_independent* is left at None), or the
            dataset's split opportunity tables are missing.
        """
        from .estimate_mus import compute_mu_g_channel_per_tumor

        if self._mu_taus is None:
            raise ValueError(
                "Mutation burdens (mu_taus) not computed. "
                "Call compute_mu_taus() first."
            )

        if not self.dataset.has_consequence_contexts_by_gene():
            raise ValueError(
                "Consequence-split opportunity counts not available "
                "in dataset. Call "
                "dataset.generate_consequence_contexts_by_gene() or "
                "load_dataset() first."
            )

        if prob_g_tau_tau_independent is None:
            if self._prob_g_tau_tau_independent is None:
                raise ValueError(
                    "prob_g_tau_tau_independent is None and no "
                    "merged base_mus has been computed to inherit it "
                    "from. Call compute_base_mus() first, or pass the "
                    "value explicitly."
                )
            prob_g_tau_tau_independent = (
                self._prob_g_tau_tau_independent
            )

        contexts = self.dataset.contexts_by_gene
        channels = {}
        for name, channel_contexts in (
            ("syn", self.dataset.contexts_by_gene_syn),
            ("nonsyn", self.dataset.contexts_by_gene_nonsyn),
        ):
            channels[name] = compute_mu_g_channel_per_tumor(
                mu_taus=self._mu_taus,
                channel_contexts_by_gene=channel_contexts,
                contexts_by_gene=contexts,
                prob_g_tau_tau_independent=prob_g_tau_tau_independent,
            )

        self._base_mus_syn = channels["syn"]
        self._base_mus_nonsyn = channels["nonsyn"]

        return self._base_mus_syn, self._base_mus_nonsyn

    def compute_channel_mu_gs(self, channel):
        """Covariate-scaled per-gene rates for one consequence channel.

        The channel counterpart of :meth:`compute_mu_gs`: applies the
        same (shared) ``cov_effects`` to that channel's baseline
        rates. Nothing is stored -- the result is returned, since
        these are cheap to recompute and only evaluation code needs
        them.

        After a ``separate_c="intercept"`` fit, the non-synonymous
        channel additionally carries that fit's own intercept
        (``exp(delta)``), which lives outside ``cov_effects``.

        Parameters
        ----------
        channel : {"syn", "nonsyn"}
            Which channel's rates to scale.

        Returns
        -------
        pd.DataFrame
            Genes × samples rates for that channel.
        """
        from .estimate_mus import compute_mus_per_gene_per_sample

        if channel == "syn":
            base = self._base_mus_syn
        elif channel == "nonsyn":
            base = self._base_mus_nonsyn
        else:
            raise ValueError(
                f"Unknown channel {channel!r}; expected 'syn' or "
                "'nonsyn'."
            )

        if base is None:
            raise ValueError(
                f"Baseline rates for the {channel!r} channel not "
                "computed. Call compute_channel_base_mus() first."
            )

        rates = compute_mus_per_gene_per_sample(
            db=self.dataset.mutation_db,
            base_mus=base,
            cov_effect=self.cov_effects,
            cov_matrix=self.cov_matrix,
        )

        # A `separate_c="intercept"` fit puts the non-synonymous
        # channel's own intercept in `_rg_delta_intercept` rather than
        # in `cov_effects` (which stays shared and 1-D). Applying it
        # here is not optional: without it the channel rates silently
        # omit the very offset that was fitted, and the offset is
        # large -- 13-18% on real cohorts.
        if (
            channel == "nonsyn"
            and self._rg_delta_intercept is not None
        ):
            rates = rates * np.exp(self._rg_delta_intercept)

        return rates

    def compute_mu_gs(self, assign_base_mus_to_rest=True, **kwargs):
        """Compute per-gene, per-sample mutation rates.

        This method computes the expected mutation rate for each gene
        in each sample by scaling the baseline mutation rates
        (base_mus) with covariate effects. It wraps
        :func:`estimate_mus.compute_mus_per_gene_per_sample`.

        The result is stored in `self.mu_gs`.

        If covariate effects have been estimated (cov_effects is not
        None), the baseline rates are scaled by exp(X @ beta) where
        X is the covariate matrix and beta are the covariate effects.
        Otherwise, returns the baseline mutation rates.

        Parameters
        ----------
        assign_base_mus_to_rest : bool, default True
            If True, assign baseline rates to any gene that did not
            receive a covariate-adjusted rate so that ``mu_gs``
            always includes every gene present in ``base_mus``.
        **kwargs : dict
            Additional keyword arguments passed to
            :func:`estimate_mus.compute_mus_per_gene_per_sample`:
            - restrict_to_passenger : bool, default False
                If True, restrict to passenger genes only
            - separate_mus_per_model : bool, default False
                If True and cov_effect is a dict of multiple models,
                return separate results per model

        Returns
        -------
        pd.DataFrame
            DataFrame with genes as index, samples as columns,
            and mutation rates as values. Shape: (n_genes, n_samples).

            Note: Even when base_mus is signature-separated (dict),
            the result is a combined DataFrame with total mutation
            rates per gene per sample.

        Raises
        ------
        ValueError
            If base_mus have not been computed. Call
            compute_base_mus() first.
        ValueError
            If model has a covariate matrix but covariate effects
            have not been estimated yet. Call estimate_cov_effects()
            first.
        ValueError
            If covariate effects are used but cov_matrix is None.

        Notes
        -----
        **With covariate effects:**

        When cov_effects is not None, the mutation rates incorporate
        the impact of covariates:
            μ_g^(j) = base_μ_g^(j) × exp(X_g @ β)

        where:
        - base_μ_g^(j) is the baseline mutation rate from base_mus
        - X_g is the covariate vector for gene g (from cov_matrix)
        - β are the covariate effects (from cov_effects)

        **Without covariate effects:**

        When cov_effects is None, returns the baseline mutation rates
        from base_mus, optionally filtered to passenger genes if
        restrict_to_passenger=True.

        **Workflow:**

        For baseline models (no covariates):
            1. model.compute_mu_taus()
            2. model.compute_base_mus()
            3. model.compute_mu_gs()

        For models with covariates (REQUIRED):
            1. model.compute_mu_taus()
            2. model.compute_base_mus()
            3. model.estimate_cov_effects()  # MUST call this first
            4. model.compute_mu_gs()  # Or automatic via estimate_cov_effects()

        The mutation database and base_mus must be loaded before
        calling this method. For models with covariates,
        estimate_cov_effects() MUST be called before this method.

        Examples
        --------
        >>> # Compute mutation rates without covariate effects
        >>> model = Model(dataset, None)
        >>> model.compute_mu_taus()
        >>> model.compute_base_mus()
        >>> model.compute_mu_gs()
        >>> print(model.mu_gs.shape)  # (n_genes, n_samples)
        >>>
        >>> # Compute mutation rates with covariate effects
        >>> model = Model(dataset)
        >>> model.assign_cov_matrix(cov_matrix)
        >>> model.compute_mu_taus()
        >>> model.compute_base_mus()
        >>> model.estimate_cov_effects()  # Calls compute_mu_gs() automatically
        >>> print(model.mu_gs.shape)  # Already computed
        >>>
        >>> # Or call compute_mu_gs() explicitly after estimate_cov_effects()
        >>> model.compute_mu_gs()  # Can be called after estimate_cov_effects()
        >>> print(model.mu_gs.shape)
        >>>
        >>> # Restrict to passenger genes only
        >>> model.compute_mu_gs(restrict_to_passenger=True)
        >>> mus_passenger = model.mu_gs
        >>>
        >>> # Copy mutation rates to another model
        >>> new_model.mu_gs = model.mu_gs

        See Also
        --------
        estimate_mus.compute_mus_per_gene_per_sample : Core computation
        compute_base_mus : Compute baseline mutation rates first
        """
        from .estimate_mus import compute_mus_per_gene_per_sample

        # Ensure mutation_db is loaded
        if self.dataset._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded in dataset. "
                "Call dataset.generate_mutation_db() or "
                "dataset.load_dataset() first."
            )

        # Ensure base_mus have been computed
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first."
            )

        # If model has covariates but effects not estimated yet
        if self.cov_matrix is not None and self.cov_effects is None:
            raise ValueError(
                "Model has a covariate matrix but covariate effects "
                "have not been estimated yet. "
                "Call estimate_cov_effects() first to estimate how "
                "covariates affect mutation rates."
            )

        # If using covariate effects, ensure cov_matrix is provided
        if self.cov_effects is not None and self.cov_matrix is None:
            raise ValueError(
                "cov_matrix must be provided when using covariate "
                "effects. This should not happen if the Model was "
                "created properly."
            )

        # Compute per-gene, per-sample mutation rates
        result = compute_mus_per_gene_per_sample(
            db=self.dataset.mutation_db,
            base_mus=self.base_mus,
            cov_effect=self.cov_effects,
            cov_matrix=self.cov_matrix,
            **kwargs,
        )

        # Set _mu_gs
        if assign_base_mus_to_rest:
            base = self.base_mus
            if isinstance(base, dict):
                base = sum(base.values())
            missing_genes = base.index.difference(result.index)
            if missing_genes.any():
                result = pd.concat([result, base.loc[missing_genes]])
        self._mu_gs = result

        return self._mu_gs

    def estimate_cov_effects(
        self,
        sample="MAP",
        chains=4,
        burn=1000,
        tol=0.05,
        excluded_samples=None,
    ):
        """Estimate covariate effect coefficients via MAP or MCMC.

        This method estimates the effect of covariates on mutation
        rates by fitting a Bernoulli model with per-gene linear
        predictors. The coefficients quantify how each covariate
        scales the baseline mutation rates.

        The estimation can use either MAP (Maximum A Posteriori)
        optimization for a point estimate, or MCMC sampling for the
        full posterior distribution.

        The result is stored in `self.cov_effects` (point estimate),
        and for MCMC the full posterior is stored in
        `self.cov_effects_posteriors`. Mutation rates (`self.mu_gs`)
        are automatically recomputed with the estimated covariate
        effects applied.

        Parameters
        ----------
        sample : {"MAP", "full"} | int, default "MAP"
            Sampling mode:
            - "MAP" (case-insensitive): Run MAP only (draws=1).
              Fast optimization for point estimate.
            - "full": MCMC with 4000 draws total (split across
              chains). Returns full posterior distribution.
            - int N: MCMC with 4000 draws, but randomly subsample N
              passenger genes to keep runtime manageable. Useful for
              quick posterior estimates with many genes.
        chains : int, default 4
            Number of MCMC chains to run in parallel. Only used when
            sample is "full" or an integer.
        burn : int, default 1000
            Number of tuning (warm-up) steps per chain. Only used
            when sample is "full" or an integer.
        tol : float, default 0.05
            Tolerance (in absolute coefficient space) used to warn
            when estimates or HDI bounds fall too close to the
            configured parameter bounds. Increase this if you want
            a looser check.
        excluded_samples : collection of str or None, default None
            Tumor sample barcodes to drop entirely before fitting
            (e.g. samples flagged by
            :func:`sample_qc.combine_sample_flags`). Applied to both
            `base_mus` and `genes_present` before the model sees them.
            Inverse-variance downweighting instead of dropping is not
            implemented -- PyMC's `Bernoulli` likelihood here has no
            native per-observation weight argument; see the L_low
            low-burden-correction plan for why this was deferred.

        Returns
        -------
        np.ndarray | arviz.InferenceData
            **MAP mode (sample="MAP")**:
                Returns np.ndarray with covariate effect coefficients:
                - (n_covariates + 1,) for signature-independent
                - (n_signatures, n_covariates + 1) for
                  signature-separated

            **MCMC mode (sample="full" or int)**:
                Returns arviz.InferenceData with posterior samples.
                Variable `c` has shape:
                - (chain, draw, n_covariates + 1) for
                  signature-independent
                - (chain, draw, n_signatures, n_covariates + 1) for
                  signature-separated

            The first coefficient (index 0 or [:, :, :, 0]) is the
            intercept, and remaining coefficients correspond to
            covariates in the order they appear in cov_matrix.columns.

        Raises
        ------
        ValueError
            If base_mus have not been computed. Call
            compute_base_mus() first.
        ValueError
            If genes_present has not been computed in the dataset.
            Call dataset.compute_gene_presence() first.
        ValueError
            If cov_matrix is None. This model needs covariates to
            estimate their effects.

        Notes
        -----
        **Model:**

        For each gene g, the baseline mutation rate is scaled by:
            μ_g^(j) = base_μ_g^(j) × exp(η_g)

        where η_g is the linear predictor:
            η_g = c_0 + c_1 × cov_1(g) + ... + c_K × cov_K(g)

        The coefficients c = [c_0, c_1, ..., c_K] are estimated by
        maximizing the likelihood (MAP) or sampling the posterior
        (MCMC) of observed gene presence data under a Bernoulli model:
            P(gene g present in tumor j) = 1 - exp(-μ_g^(j))

        **Interpretation:**

        - c_0 (intercept): Overall scaling of mutation rates
        - c_k > 0: Covariate k increases mutation rates
        - c_k < 0: Covariate k decreases mutation rates
        - c_k = 0: Covariate k has no effect

        **Gene filtering:**

        The estimation uses only passenger genes (not in Cancer Gene
        Census) with complete covariate data:
        1. Passenger genes are identified using Cancer Gene Census
        2. Genes with any NaN values in covariates are excluded
        3. Only genes with complete data are used for estimation

        When `sample` is an integer N, a random subset of N genes
        is drawn from the filtered passenger genes (using
        `constants.random_seed` for reproducibility).

        This filtering ensures unbiased coefficient estimates by:
        - Using neutral selection genes (passengers) to avoid
          confounding from positive/negative selection
        - Excluding genes with missing covariates that would bias
          the likelihood

        The number of genes used is stored in
        `self.n_in_cov_effects_estimation` and can be accessed
        after calling this method.

        **Posterior mean vs MAP:**

        For MCMC modes ("full" or int), this method uses the
        posterior mean (average across MCMC samples) to set
        `self.cov_effects`, rather than the MAP estimate. The
        posterior mean is generally preferred as a point estimate
        from MCMC output, as it:
        - Accounts for posterior uncertainty
        - Is less sensitive to optimization convergence issues
        - Provides a Bayes estimate under squared error loss

        **Complete Process:**

        This method performs the following steps in order:

        1. **Configuration and Validation:**
           - Extract bounds configuration from `cov_effects_kwargs`
           - Validate prerequisites: `base_mus`, `genes_present`,
             `cov_matrix` must be available
           - Determine sampling mode (MAP vs MCMC) based on `sample`
             parameter

        2. **Gene Filtering:**
           - Identify passenger genes using Cancer Gene Census
           - Filter to genes with complete covariate data (no NaN)
           - Optionally subsample N genes if `sample` is an integer
           - Store final gene count in
             `self._n_in_cov_effects_estimation`

        3. **Data Preparation:**
           - Detect signature-dependent mode (if `base_mus` is dict)
           - Filter and transpose `base_mus` to selected passenger
             genes
           - Filter `genes_present` matrix to selected genes
           - Filter `cov_matrix` to selected genes
           - Convert to numpy arrays for estimation

        4. **Estimation:**
           - **MAP mode (sample="MAP"):**
             - Run MAP optimization with draws=1
             - Store point estimate in `self.cov_effects`
             - Check if estimates are near bounds, warn if needed
           - **MCMC mode (sample="full" or int):**
             - Run MCMC sampling with 4000 draws across chains
             - Store full posterior in `self.cov_effects_posteriors`
             - Extract posterior mean and store in `self.cov_effects`
             - Print posterior summary table
             - Check if HDI bounds are near parameter bounds, warn
               if needed

        5. **Automatic Recomputation:**
           - Call `compute_mu_gs()` to recompute gene-level mutation
             rates with the estimated effects applied
           - Call `compute_mu_ms()` to recompute variant-level
             mutation rates with the estimated effects applied
           - Call `estimate_passenger_genes_r2()` to evaluate model
             performance on passenger genes

        6. **Return:**
           - Return `cov_effects_posteriors` (InferenceData) for MCMC
           - Return `cov_effects` (ndarray) for MAP

        After this method completes, the following attributes are
        populated and ready to use:
        - `self.cov_effects` - coefficient estimates
        - `self.cov_effects_posteriors` - full posterior (MCMC only)
        - `self._n_in_cov_effects_estimation` - number of genes used
        - `self._mu_gs` - gene-level rates with covariate effects
        - `self.mu_ms` - variant-level rates with covariate effects
        - `self._passenger_genes_r2` - model performance metric

        You do not need to manually call `compute_mu_gs()`,
        `compute_mu_ms()`, or `estimate_passenger_genes_r2()` after
        this method.

        **Configuration:**

        Additional parameters for the estimation can be passed via
        `cov_effects_kwargs` when creating the Model:
            - lower_bounds_c : float or array, default -1
            - upper_bounds_c : float or array, default 2
            - save_path : str or Path, optional absolute path prefix
              (without extension) where estimation results should be
              saved

        **Typical workflow:**

        1. Compute baseline mutation rates:
            >>> model.compute_mu_taus()
            >>> model.compute_base_mus()

        2. Estimate covariate effects (mu_gs, mu_ms, and R² computed
           automatically):
            >>> # MAP estimation (fast)
            >>> model.estimate_cov_effects(sample="MAP")
            >>> print(f"Used {model.n_in_cov_effects_estimation} "
            ...       f"genes")
            >>> print(f"Passenger genes R²: "
            ...       f"{model.passenger_genes_r2:.4f}")
            >>> print(f"Intercept: {model.cov_effects[0]:.4f}")
            >>>
            >>> # Full posterior (slower)
            >>> posterior = model.estimate_cov_effects(sample="full")
            >>> import arviz as az
            >>> az.summary(posterior, var_names=['c'])

        Examples
        --------
        >>> # MAP estimation (default)
        >>> model = Model(dataset)
        >>> model.assign_cov_matrix(cov_matrix_full[['mrt']])
        >>> model.compute_mu_taus()
        >>> model.compute_base_mus()
        >>> model.estimate_cov_effects()  # sample="MAP" by default
        >>> print(f"Used {model.n_in_cov_effects_estimation} genes")
        >>> print(f"MRT effect: {model.cov_effects[1]:.4f}")
        >>>
        >>> # Full posterior estimation
        >>> posterior = model.estimate_cov_effects(sample="full")
        >>> import arviz as az
        >>> az.plot_posterior(posterior, var_names=['c'])
        >>> # Posterior mean stored in cov_effects
        >>> print(f"Intercept (mean): {model.cov_effects[0]:.4f}")
        >>>
        >>> # Subsampled MCMC (faster, for exploration)
        >>> posterior = model.estimate_cov_effects(
        ...     sample=1000, chains=4, burn=500)
        >>> print(f"Used {model.n_in_cov_effects_estimation} genes "
        ...       f"(subsampled from all passenger genes)")

        See Also
        --------
        estimate_covariates_effect.estimate_covariates_effect : Core
            computation
        compute_base_mus : Must be called first
        compute_mu_gs : Called automatically to compute gene-level
            rates
        compute_mu_ms : Called automatically to compute variant-level
            rates
        estimate_passenger_genes_r2 : Called automatically to
            evaluate model performance
        n_in_cov_effects_estimation : Property showing gene count
            used
        """
        from .constants import random_seed
        from .estimate_covariates_effect import (
            estimate_covariates_effect,
        )
        from .estimate_presence import filter_passenger_genes_ensembl

        # Step 1: Configuration - Extract bounds from kwargs
        cov_effects_kwargs = dict(self.cov_effects_kwargs)
        signature = inspect.signature(estimate_covariates_effect)
        default_lower_bound = signature.parameters[
            "lower_bounds_c"
        ].default
        if default_lower_bound is inspect._empty:
            default_lower_bound = None
        default_upper_bound = signature.parameters[
            "upper_bounds_c"
        ].default
        if default_upper_bound is inspect._empty:
            default_upper_bound = None
        lower_bounds_value = cov_effects_kwargs.get(
            "lower_bounds_c", default_lower_bound
        )
        upper_bounds_value = cov_effects_kwargs.get(
            "upper_bounds_c", default_upper_bound
        )

        # Step 1: Validation - Ensure prerequisites are available
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first."
            )

        if self.dataset._genes_present is None:
            raise ValueError(
                "Gene presence matrix not computed in dataset. "
                "Call dataset.compute_gene_presence() first."
            )

        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix is None. Cannot estimate covariate "
                "effects without covariates. Create model with "
                "cov_matrix or use assign_cov_matrix()."
            )

        # Step 1: Determine draws and sampling mode
        if isinstance(sample, int) or (
            isinstance(sample, str) and sample.lower() == "full"
        ):
            draws = 4000
            is_mcmc = True
        elif isinstance(sample, str) and sample.lower() == "map":
            draws = 1
            is_mcmc = False
        else:
            raise ValueError(
                f"sample must be 'MAP', 'full', or an integer, "
                f"got {sample}"
            )

        # Step 2: Gene Filtering - Identify passenger genes
        passenger_gene_ids = filter_passenger_genes_ensembl(
            self.cov_matrix.index
        )

        # Step 2: Filter to genes with complete covariate data (no NaN)
        passenger_cov = self.cov_matrix.loc[passenger_gene_ids]
        complete_mask = ~passenger_cov.isna().any(axis=1)
        passenger_genes_complete = passenger_gene_ids[complete_mask]

        # Step 2: Optionally subsample genes if integer provided
        if draws > 1 and isinstance(sample, int):
            logger.info(
                f"Subsampling {sample} genes from "
                f"{len(passenger_genes_complete)} passenger genes "
                f"with complete covariates"
            )
            passenger_genes_complete = pd.Index(
                passenger_genes_complete.to_series().sample(
                    sample, random_state=random_seed
                )
            )

        # Step 2: Store gene count for later access
        self._n_in_cov_effects_estimation = len(
            passenger_genes_complete
        )

        # Step 3: Data Preparation - Detect signature-dependent mode
        is_signature_dependent = isinstance(self._base_mus, dict)

        # Step 3: Log estimation details
        if is_mcmc:
            logger.info(
                f"Estimating covariate effects posteriors for "
                f"{self._n_in_cov_effects_estimation} passenger genes "
                f"with {self.cov_matrix.shape[1]} covariate(s)"
            )
            logger.info(
                f"MCMC parameters: {draws} draws, {chains} chains, "
                f"{burn} tuning steps"
            )
        else:
            logger.info(
                f"Estimating covariate effects for "
                f"{self._n_in_cov_effects_estimation} passenger genes "
                f"with {self.cov_matrix.shape[1]} covariate(s)"
            )

        if is_signature_dependent:
            n_sigs = len(self._base_mus)
            logger.info(
                f"Using signature-dependent mode "
                f"({n_sigs} signatures)"
            )
        else:
            logger.info("Using signature-independent mode")

        # Step 3: drop excluded samples (tumor axis) before anything
        # gets transposed to a bare numpy array -- past this point
        # there's no longer a labeled axis to filter by barcode.
        genes_present_source = self.dataset.genes_present
        if excluded_samples is not None:
            kept_samples = genes_present_source.columns.difference(
                excluded_samples
            )
            genes_present_source = genes_present_source[kept_samples]
            if isinstance(self._base_mus, dict):
                base_mus_source = {
                    sig: df[kept_samples]
                    for sig, df in self._base_mus.items()
                }
            else:
                base_mus_source = self._base_mus[kept_samples]
        else:
            base_mus_source = self._base_mus

        # Step 3: Filter and transpose base_mus
        if isinstance(base_mus_source, dict):
            # Signature-separated: filter and transpose each DataFrame
            mus_transposed = {
                sig: df.loc[passenger_genes_complete].T.values
                for sig, df in base_mus_source.items()
            }
        else:
            # Signature-independent: filter and transpose DataFrame
            mus_transposed = base_mus_source.loc[
                passenger_genes_complete
            ].T.values

        # Step 3: Filter genes_present matrix. genes_present's
        # crosstab only contains genes observed as mutated in this
        # cohort, so under gene_universe="wes_target",
        # passenger_genes_complete (drawn from cov_matrix's index,
        # which can include genes never mutated here but with real
        # covariate data) may include IDs absent from genes_present --
        # by construction, never-mutated, so 0 in every sample.
        presence_matrix = genes_present_source.reindex(
            passenger_genes_complete, fill_value=0
        ).T.values

        # Step 3: Filter cov_matrix and convert to array
        cov_matrix_array = self.cov_matrix.loc[
            passenger_genes_complete
        ].values

        # Step 4: Estimation - Run MAP or MCMC
        if is_mcmc:
            logger.info("Running MCMC sampling...")
            result = estimate_covariates_effect(
                mus=mus_transposed,
                presence_matrix=presence_matrix,
                cov_matrix=cov_matrix_array,
                draws=draws,
                chains=chains,
                burn=burn,
                **cov_effects_kwargs,
            )

            # Store full posterior
            self.cov_effects_posteriors = result
            logger.info("MCMC sampling completed")

            # Extract posterior mean for use in subsequent
            # calculations
            import arviz as az

            posterior_mean = (
                az.extract(result, var_names=["c"])
                .mean(dim="sample")
                .values
            )
            self.cov_effects = posterior_mean
            logger.info(
                "Extracted posterior mean for subsequent "
                "calculations"
            )

            lower_bounds_arr, upper_bounds_arr = (
                self._resolve_covariate_bounds(
                    self.cov_effects.shape,
                    lower_bounds_value,
                    upper_bounds_value,
                )
            )

            summary = az.summary(result, var_names=["c"])
            logger.info("Posterior summary:\n%s", summary.to_string())

            if {"hdi_3%", "hdi_97%"} <= set(summary.columns):
                hdi_lower = summary["hdi_3%"].to_numpy()
                hdi_upper = summary["hdi_97%"].to_numpy()
                self._warn_if_near_bounds(
                    lower_candidate=hdi_lower,
                    upper_candidate=hdi_upper,
                    lower_bounds=lower_bounds_arr,
                    upper_bounds=upper_bounds_arr,
                    tol=tol,
                    mode_desc="Posterior HDI",
                )
            else:
                logger.warning(
                    "Posterior summary missing HDI columns; "
                    "skipping bounds proximity check for posterior."
                )
        else:
            logger.info("Running MAP estimation...")
            result = estimate_covariates_effect(
                mus=mus_transposed,
                presence_matrix=presence_matrix,
                cov_matrix=cov_matrix_array,
                draws=1,
                **cov_effects_kwargs,
            )

            # Extract MAP estimate from result dict
            self.cov_effects = result["c"]
            logger.info("MAP estimation completed")

            lower_bounds_arr, upper_bounds_arr = (
                self._resolve_covariate_bounds(
                    self.cov_effects.shape,
                    lower_bounds_value,
                    upper_bounds_value,
                )
            )
            self._warn_if_near_bounds(
                lower_candidate=self.cov_effects,
                upper_candidate=self.cov_effects,
                lower_bounds=lower_bounds_arr,
                upper_bounds=upper_bounds_arr,
                tol=tol,
                mode_desc="MAP estimates",
            )

        # Step 5: Automatic Recomputation - gene-level rates
        self.compute_mu_gs()

        # Step 5: Automatic Recomputation - variant-level rates
        self.compute_mu_ms()

        # Step 5: Automatic Recomputation - model performance
        self.estimate_passenger_genes_r2()

        # Step 6: Return results
        if is_mcmc:
            return self.cov_effects_posteriors
        else:
            return self.cov_effects

    def estimate_channel_cov_effects(
        self,
        sample="MAP",
        chains=4,
        burn=1000,
        tol=0.05,
        excluded_samples=None,
        include_drivers=True,
        likelihood="bernoulli",
    ):
        """Fit one shared ``c`` against both consequence channels.

        The channel-split sibling of :meth:`estimate_cov_effects`.
        Where that method fits a single Bernoulli likelihood over
        "gene mutated at all", restricted to passenger genes, this one
        fits **two** Bernoulli likelihoods sharing a single
        coefficient vector:

        * **silent channel** over gene set ``G`` -- all genes with
          complete covariates, *drivers included*, since a synonymous
          mutation is selection-free in any gene;
        * **non-silent channel** over gene set ``P`` -- passenger
          genes only, exactly as today's fit.

        Splitting the two also splits the baseline rate: each channel
        is scored against its own ``μ̄_g^(syn/nonsyn,j)`` from
        :meth:`compute_channel_base_mus`, not against the merged
        ``base_mus``.

        This is **not** a re-parameterisation of the merged fit.
        ``1[silent or non-silent]`` is an OR of two events, and
        observing the pair jointly is strictly more informative than
        the OR alone, so results differ even with
        ``include_drivers=False``. Expect the drivers-off number to
        land *close* to the merged baseline; a large gap there is a
        signal to debug the split, not a result.

        Parameters
        ----------
        sample : {"MAP", "full"} | int, default "MAP"
            As in :meth:`estimate_cov_effects`.
        chains, burn, tol, excluded_samples
            As in :meth:`estimate_cov_effects`.
        include_drivers : bool, default True
            If True, the silent channel runs over every gene with
            complete covariates (``G``). If False, it is restricted to
            the same passenger set as the non-silent channel (``P``),
            isolating the value of the finer-grained observation from
            the value of the driver genes it lets in.
        likelihood : {"bernoulli", "poisson"}, default "bernoulli"
            Observation model. ``"bernoulli"`` fits each channel's
            0/1 presence matrix; ``"poisson"`` fits its mutation
            *counts* (``genes_counts_silent`` /
            ``genes_counts_non_silent`` -- call
            :meth:`MutationDataset.compute_gene_counts_channels`
            first). Counts are the generative quantity; presence
            censors them, and worst exactly where the mutation mass
            is. Score a Poisson fit with
            ``estimate_passenger_genes_r2(target="non_silent_counts")``
            rather than the presence targets.

        Returns
        -------
        np.ndarray | arviz.InferenceData
            As in :meth:`estimate_cov_effects`; ``c`` is a single
            shared vector, not one per channel.

        Raises
        ------
        NotImplementedError
            If ``base_mus`` is signature-separated. The two-channel
            likelihood has no multi-signature mode yet.
        ValueError
            If the channel baselines, either observation matrix for
            the chosen ``likelihood``, or the covariate matrix are
            missing.

        Notes
        -----
        The gene sets are deliberately allowed to differ between
        channels, which is the whole point: a driver's non-silent
        counts are selection-contaminated by construction, but its
        silent counts are not, so it can inform the shared covariate
        effects through the silent channel only.

        As in :meth:`estimate_cov_effects`, ``mu_gs``, ``mu_ms`` and
        the passenger-gene R² are recomputed afterwards, so the merged
        ``mu_gs`` stays consistent with the newly fitted ``c`` (which
        is shared, so it scales both channels and hence their sum).
        """
        from .constants import random_seed
        from .estimate_covariates_effect import (
            estimate_channel_covariates_effect,
        )
        from .estimate_presence import filter_passenger_genes_ensembl

        cov_effects_kwargs = dict(self.cov_effects_kwargs)
        signature = inspect.signature(
            estimate_channel_covariates_effect
        )
        default_lower_bound = signature.parameters[
            "lower_bounds_c"
        ].default
        default_upper_bound = signature.parameters[
            "upper_bounds_c"
        ].default
        lower_bounds_value = cov_effects_kwargs.get(
            "lower_bounds_c", default_lower_bound
        )
        upper_bounds_value = cov_effects_kwargs.get(
            "upper_bounds_c", default_upper_bound
        )

        if not self.has_channel_base_mus():
            raise ValueError(
                "Channel baseline rates not computed. Call "
                "compute_channel_base_mus() first."
            )

        if isinstance(self._base_mus_syn, dict):
            raise NotImplementedError(
                "estimate_channel_cov_effects has no "
                "signature-separated mode; build the model without "
                "cov_effects_per_sigma/signature_selection."
            )

        if likelihood == "bernoulli":
            if self.dataset._genes_present_silent is None:
                raise ValueError(
                    "Silent gene presence matrix not computed in "
                    "dataset. Call "
                    "dataset.compute_gene_presence_silent() first."
                )
            if self.dataset._genes_present_non_silent is None:
                raise ValueError(
                    "Non-silent gene presence matrix not computed "
                    "in dataset. Call "
                    "dataset.compute_gene_presence_non_silent() "
                    "first."
                )
            observed_silent = self.dataset.genes_present_silent
            observed_non_silent = (
                self.dataset.genes_present_non_silent
            )
        elif likelihood == "poisson":
            if not self.dataset.has_channel_counts():
                raise ValueError(
                    "Channel count matrices not computed in "
                    "dataset. Call "
                    "dataset.compute_gene_counts_channels() first."
                )
            observed_silent = self.dataset.genes_counts_silent
            observed_non_silent = self.dataset.genes_counts_non_silent
        else:
            raise ValueError(
                f"Unknown likelihood {likelihood!r}; expected "
                "'bernoulli' or 'poisson'."
            )

        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix is None. Cannot estimate covariate "
                "effects without covariates. Create model with "
                "cov_matrix or use assign_cov_matrix()."
            )

        if isinstance(sample, int) or (
            isinstance(sample, str) and sample.lower() == "full"
        ):
            draws = 4000
            is_mcmc = True
        elif isinstance(sample, str) and sample.lower() == "map":
            draws = 1
            is_mcmc = False
        else:
            raise ValueError(
                f"sample must be 'MAP', 'full', or an integer, "
                f"got {sample}"
            )

        # Gene sets: complete covariates for both, passenger-only for
        # the non-silent channel, everything (or the same passenger
        # set) for the silent one.
        complete_genes = self.cov_matrix.index[
            ~self.cov_matrix.isna().any(axis=1)
        ]
        passenger_genes = pd.Index(
            filter_passenger_genes_ensembl(complete_genes)
        )
        silent_genes = (
            complete_genes if include_drivers else passenger_genes
        )

        if draws > 1 and isinstance(sample, int):
            logger.info(
                f"Subsampling {sample} genes per channel from "
                f"{len(silent_genes)} silent / "
                f"{len(passenger_genes)} non-silent genes"
            )
            silent_genes = pd.Index(
                silent_genes.to_series().sample(
                    min(sample, len(silent_genes)),
                    random_state=random_seed,
                )
            )
            passenger_genes = pd.Index(
                passenger_genes.to_series().sample(
                    min(sample, len(passenger_genes)),
                    random_state=random_seed,
                )
            )

        # Reported as the non-silent channel's gene count -- the
        # directly comparable number to estimate_cov_effects's.
        self._n_in_cov_effects_estimation = len(passenger_genes)

        logger.info(
            f"Estimating shared covariate effects ({likelihood}) "
            f"across two channels: {len(silent_genes)} genes in the "
            f"silent channel (drivers "
            f"{'included' if include_drivers else 'excluded'}), "
            f"{len(passenger_genes)} in the non-silent channel, "
            f"{self.cov_matrix.shape[1]} covariate(s)"
        )

        channel_inputs = {}
        for name, genes, base_mus, presence_source in (
            (
                "silent",
                silent_genes,
                self._base_mus_syn,
                observed_silent,
            ),
            (
                "non_silent",
                passenger_genes,
                self._base_mus_nonsyn,
                observed_non_silent,
            ),
        ):
            mus = base_mus.loc[genes]
            if excluded_samples is not None:
                mus = mus[mus.columns.difference(excluded_samples)]
            # Align the observations onto the rate matrix's own
            # sample axis:
            # both come from the same dataset, but they are built by
            # different routes (mu_taus' index vs a crosstab), and
            # past the `.T.values` below there is no labelled axis
            # left to catch a mismatch. Genes or samples absent from
            # the crosstab were never observed mutated in this
            # channel, i.e. 0, not missing.
            presence = presence_source.reindex(
                index=genes, columns=mus.columns, fill_value=0
            )
            channel_inputs[name] = (
                mus.T.values,
                presence.T.values,
                self.cov_matrix.loc[genes].values,
            )

        result = estimate_channel_covariates_effect(
            mus_silent=channel_inputs["silent"][0],
            observed_silent=channel_inputs["silent"][1],
            cov_matrix_silent=channel_inputs["silent"][2],
            mus_non_silent=channel_inputs["non_silent"][0],
            observed_non_silent=channel_inputs["non_silent"][1],
            cov_matrix_non_silent=channel_inputs["non_silent"][2],
            likelihood=likelihood,
            draws=draws,
            chains=chains,
            burn=burn,
            **cov_effects_kwargs,
        )

        if is_mcmc:
            import arviz as az

            self.cov_effects_posteriors = result
            self.cov_effects = (
                az.extract(result, var_names=["c"])
                .mean(dim="sample")
                .values
            )
            summary = az.summary(result, var_names=["c"])
            logger.info("Posterior summary:\n%s", summary.to_string())
            candidates = (
                (
                    summary["hdi_3%"].to_numpy(),
                    summary["hdi_97%"].to_numpy(),
                )
                if {"hdi_3%", "hdi_97%"} <= set(summary.columns)
                else None
            )
            mode_desc = "Posterior HDI"
        else:
            self.cov_effects = result["c"]
            candidates = (self.cov_effects, self.cov_effects)
            mode_desc = "MAP estimates"

        lower_bounds_arr, upper_bounds_arr = (
            self._resolve_covariate_bounds(
                self.cov_effects.shape,
                lower_bounds_value,
                upper_bounds_value,
            )
        )
        if candidates is None:
            logger.warning(
                "Posterior summary missing HDI columns; skipping "
                "bounds proximity check for posterior."
            )
        else:
            self._warn_if_near_bounds(
                lower_candidate=candidates[0],
                upper_candidate=candidates[1],
                lower_bounds=lower_bounds_arr,
                upper_bounds=upper_bounds_arr,
                tol=tol,
                mode_desc=mode_desc,
            )

        self.compute_mu_gs()
        # Variant-level rates are not needed by this stage's
        # channel-holdout evaluation, and a dataset can legitimately
        # have none (the split fit only needs gene-level presence),
        # so skip rather than fail -- loudly, since any previously
        # computed mu_ms is now stale with respect to the new `c`.
        if self.dataset.has_variants():
            self.compute_mu_ms()
        elif self.mu_ms is not None:
            logger.warning(
                "No variant database: mu_ms was NOT recomputed and is "
                "now stale with respect to the newly fitted "
                "coefficients."
            )
        self.estimate_passenger_genes_r2()

        if is_mcmc:
            return self.cov_effects_posteriors
        else:
            return self.cov_effects

    def _channel_gene_statistics(
        self, include_drivers=True, excluded_samples=None
    ):
        """Per-gene sufficient statistics for the ``r_g`` likelihood.

        With ``r_g`` marginalized out (see :mod:`sigmutsel.estimate_rg`),
        each channel's Poisson terms depend on the data only through
        per-gene totals, so the genes × samples matrices collapse to
        four aligned vectors.

        Returns
        -------
        dict
            ``genes`` (the silent channel's gene set ``G``, the index
            everything else is aligned to), ``counts_silent``,
            ``counts_non_silent``, ``baseline_silent``,
            ``baseline_non_silent`` (all ``pd.Series`` over ``genes``,
            with zeros for genes outside the non-synonymous set), and
            ``in_non_silent`` (the passenger gene index).
        """
        from .estimate_presence import filter_passenger_genes_ensembl

        complete_genes = self.cov_matrix.index[
            ~self.cov_matrix.isna().any(axis=1)
        ]
        passenger_genes = pd.Index(
            filter_passenger_genes_ensembl(complete_genes)
        )
        silent_genes = (
            complete_genes if include_drivers else passenger_genes
        )

        def _sums(frame, genes):
            frame = frame.reindex(
                index=genes,
                columns=self._base_mus_syn.columns,
                fill_value=0,
            )
            if excluded_samples is not None:
                frame = frame[
                    frame.columns.difference(excluded_samples)
                ]
            return frame.sum(axis=1)

        def _baseline(frame, genes):
            frame = frame.loc[genes]
            if excluded_samples is not None:
                frame = frame[
                    frame.columns.difference(excluded_samples)
                ]
            return frame.sum(axis=1)

        zeros = pd.Series(0.0, index=silent_genes)
        counts_non_silent = zeros.add(
            _sums(
                self.dataset.genes_counts_non_silent, passenger_genes
            ),
            fill_value=0.0,
        ).loc[silent_genes]
        baseline_non_silent = zeros.add(
            _baseline(self._base_mus_nonsyn, passenger_genes),
            fill_value=0.0,
        ).loc[silent_genes]

        return {
            "genes": silent_genes,
            "counts_silent": _sums(
                self.dataset.genes_counts_silent, silent_genes
            ),
            "counts_non_silent": counts_non_silent,
            "baseline_silent": _baseline(
                self._base_mus_syn, silent_genes
            ),
            "baseline_non_silent": baseline_non_silent,
            "in_non_silent": passenger_genes,
        }

    def estimate_channel_rg_cov_effects(
        self,
        sample="MAP",
        chains=4,
        burn=1000,
        tol=0.05,
        excluded_samples=None,
        include_drivers=True,
        separate_c="intercept",
    ):
        """Fit the full unified model: shared ``c``, ``θ`` and ``r_g``.

        Stage 4 of the channel-split model. Adds a per-gene rate
        correction ``r_g ~ Gamma(θ, 1/θ)``, shared across both
        consequence channels, on top of the two-channel Poisson
        likelihood. ``r_g`` is integrated out analytically, so the fit
        is over ``c`` and ``θ`` only -- see
        :mod:`sigmutsel.estimate_rg` for the closed form and why it
        matters.

        Afterwards, ``self.rg_theta`` holds the fitted θ and the two
        ``r_g`` variants are available through
        :meth:`compute_r_g_production` and
        :meth:`compute_r_g_for_evaluation`. Neither is applied to
        ``mu_gs`` automatically: which one a number was computed with
        is exactly the thing that must never be ambiguous, so applying
        ``r_g`` is always an explicit act at the call site.

        Parameters
        ----------
        sample, chains, burn, tol, excluded_samples, include_drivers
            As in :meth:`estimate_channel_cov_effects`. The same
            ``excluded_samples`` must be used here and in any later
            ``r_g`` or R² call, since the per-gene statistics are sums
            over whichever samples were kept.
        separate_c : bool | str, default "intercept"
            ``False`` (shared), ``"intercept"`` (shared slopes, own
            non-synonymous intercept) or ``True`` (own vector per
            channel); ``r_g`` and θ stay shared in all three. The
            intercept absorbs a calibration offset between channels,
            so testing ``True`` against ``False`` conflates that
            with the slopes -- go through ``"intercept"`` to
            separate them. It is the default because a shared
            intercept leaves the channels miscalibrated against each
            other by 13--18% in every cohort measured; see
            :mod:`sigmutsel.estimate_rg` for the numbers and for the
            caveat that correcting it is not a uniform win.
            The result lands in :attr:`channel_cov_effects` with shape
            ``(2, n_coeffs)`` -- row 0 synonymous, row 1
            non-synonymous -- and **not** in ``cov_effects``, because
            a single ``mu_gs`` is no longer defined by one ``c``. For
            the same reason the usual downstream recomputation
            (``mu_gs``/``mu_ms``/R²) is skipped in this mode; use
            :meth:`channel_rg_log_likelihood_at_fit` to compare the
            two models.

        Returns
        -------
        np.ndarray | arviz.InferenceData
            The shared coefficient vector (MAP) or its posterior.

        Raises
        ------
        ValueError
            If the channel baselines, the channel count matrices, or
            the covariate matrix are missing.
        NotImplementedError
            If ``base_mus`` is signature-separated.
        """
        from .estimate_rg import estimate_channel_rg_effect

        cov_effects_kwargs = dict(self.cov_effects_kwargs)
        signature = inspect.signature(estimate_channel_rg_effect)
        lower_bounds_value = cov_effects_kwargs.get(
            "lower_bounds_c",
            signature.parameters["lower_bounds_c"].default,
        )
        upper_bounds_value = cov_effects_kwargs.get(
            "upper_bounds_c",
            signature.parameters["upper_bounds_c"].default,
        )

        if not self.has_channel_base_mus():
            raise ValueError(
                "Channel baseline rates not computed. Call "
                "compute_channel_base_mus() first."
            )
        if isinstance(self._base_mus_syn, dict):
            raise NotImplementedError(
                "estimate_channel_rg_cov_effects has no "
                "signature-separated mode."
            )
        if not self.dataset.has_channel_counts():
            raise ValueError(
                "Channel count matrices not computed in dataset. "
                "Call dataset.compute_gene_counts_channels() first."
            )
        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix is None. Cannot estimate covariate "
                "effects without covariates."
            )

        if isinstance(sample, int) or (
            isinstance(sample, str) and sample.lower() == "full"
        ):
            draws = 4000
            is_mcmc = True
        elif isinstance(sample, str) and sample.lower() == "map":
            draws = 1
            is_mcmc = False
        else:
            raise ValueError(
                f"sample must be 'MAP', 'full', or an integer, "
                f"got {sample}"
            )

        stats = self._channel_gene_statistics(
            include_drivers=include_drivers,
            excluded_samples=excluded_samples,
        )
        self._rg_statistics = stats
        self._n_in_cov_effects_estimation = len(
            stats["in_non_silent"]
        )

        logger.info(
            f"Fitting shared c and theta over {len(stats['genes'])} "
            f"genes (drivers "
            f"{'included' if include_drivers else 'excluded'}), "
            f"{len(stats['in_non_silent'])} of them in the "
            "non-silent channel"
        )

        result = estimate_channel_rg_effect(
            counts_silent=stats["counts_silent"].values,
            baseline_silent=stats["baseline_silent"].values,
            counts_non_silent=stats["counts_non_silent"].values,
            baseline_non_silent=stats["baseline_non_silent"].values,
            cov_matrix=self.cov_matrix.loc[stats["genes"]].values,
            separate_c=separate_c,
            draws=draws,
            chains=chains,
            burn=burn,
            **cov_effects_kwargs,
        )
        self._rg_separate_c = separate_c

        self._rg_delta_intercept = None
        if is_mcmc:
            import arviz as az

            self.cov_effects_posteriors = result
            self.cov_effects = (
                az.extract(result, var_names=["c"])
                .mean(dim="sample")
                .values
            )
            self._rg_theta = float(
                np.exp(
                    az.extract(result, var_names=["log_theta"])
                    .mean(dim="sample")
                    .values
                )
            )
            if separate_c == "intercept":
                self._rg_delta_intercept = float(
                    az.extract(result, var_names=["delta_intercept"])
                    .mean(dim="sample")
                    .values
                )
            summary = az.summary(result, var_names=["c", "log_theta"])
            logger.info("Posterior summary:\n%s", summary.to_string())
            candidates = (
                (
                    summary["hdi_3%"].to_numpy()[
                        : len(self.cov_effects)
                    ],
                    summary["hdi_97%"].to_numpy()[
                        : len(self.cov_effects)
                    ],
                )
                if {"hdi_3%", "hdi_97%"} <= set(summary.columns)
                else None
            )
            mode_desc = "Posterior HDI"
        else:
            self.cov_effects = result["c"]
            self._rg_theta = float(np.exp(result["log_theta"]))
            self._rg_delta_intercept = (
                float(result["delta_intercept"])
                if "delta_intercept" in result
                else None
            )
            candidates = (self.cov_effects, self.cov_effects)
            mode_desc = "MAP estimates"

        logger.info(f"Fitted theta = {self._rg_theta:.4g}")

        lower_bounds_arr, upper_bounds_arr = (
            self._resolve_covariate_bounds(
                self.cov_effects.shape,
                lower_bounds_value,
                upper_bounds_value,
            )
        )
        if candidates is not None:
            self._warn_if_near_bounds(
                lower_candidate=candidates[0],
                upper_candidate=candidates[1],
                lower_bounds=lower_bounds_arr,
                upper_bounds=upper_bounds_arr,
                tol=tol,
                mode_desc=mode_desc,
            )

        if separate_c is True:
            # A single mu_gs is not defined by one `c` here, so the
            # usual recomputation would be meaningless. Move the
            # coefficients out of `cov_effects` entirely so nothing
            # downstream can read a (2, n) array as the shared vector.
            self._channel_cov_effects = np.asarray(self.cov_effects)
            self.cov_effects = None
        else:
            self._channel_cov_effects = None
            self.compute_mu_gs()
            if self.dataset.has_variants():
                self.compute_mu_ms()
            self.estimate_passenger_genes_r2()

        if is_mcmc:
            return self.cov_effects_posteriors
        elif separate_c is True:
            return self._channel_cov_effects
        else:
            return self.cov_effects

    @property
    def channel_cov_effects(self):
        """Per-channel coefficients from a ``separate_c`` fit.

        Shape ``(2, n_coeffs)``: row 0 synonymous, row 1
        non-synonymous. Deliberately separate from
        :attr:`cov_effects`, which always means the shared vector.
        """
        if self._channel_cov_effects is None:
            raise ValueError(
                "No separate-c fit available. Call "
                "estimate_channel_rg_cov_effects(separate_c=True) "
                "first."
            )
        return self._channel_cov_effects

    def channel_rg_log_likelihood_at_fit(self):
        """Marginal log-likelihood at the fitted ``c`` and ``θ``.

        Evaluates the same objective the fit maximised, so a shared
        and a separate fit on identical data can be compared
        directly: the shared model is nested inside the separate one
        at ``c^(syn) = c^(nonsyn)``, so
        ``2 * (ll_separate - ll_shared)`` is a likelihood-ratio
        statistic on ``n_coeffs`` degrees of freedom.

        The constant Poisson term dropped by the likelihood is the
        same for both models, so it cancels in the difference.
        """
        from .estimate_rg import channel_rg_log_likelihood

        if self._rg_statistics is None or self._rg_theta is None:
            raise ValueError(
                "No r_g fit available. Call "
                "estimate_channel_rg_cov_effects() first."
            )

        stats = self._rg_statistics
        cov = self.cov_matrix.loc[stats["genes"]].values
        cov_ext = np.concatenate(
            [np.ones((cov.shape[0], 1)), cov], axis=1
        )

        if self._rg_separate_c is True:
            coeffs = self.channel_cov_effects
            eta_silent = cov_ext @ coeffs[0]
            eta_non_silent = cov_ext @ coeffs[1]
        else:
            eta_silent = cov_ext @ np.asarray(self.cov_effects)
            eta_non_silent = (
                eta_silent + self._rg_delta_intercept
                if self._rg_delta_intercept is not None
                else None
            )

        return float(
            channel_rg_log_likelihood(
                eta_silent=eta_silent,
                eta_non_silent=eta_non_silent,
                theta=self.rg_theta,
                counts_silent=stats["counts_silent"].values,
                counts_non_silent=stats["counts_non_silent"].values,
                baseline_silent=stats["baseline_silent"].values,
                baseline_non_silent=stats[
                    "baseline_non_silent"
                ].values,
            )
        )

    @property
    def rg_theta(self):
        """Fitted Gamma shape θ for ``r_g`` (lazy)."""
        if self._rg_theta is None:
            raise ValueError(
                "theta not fitted. Call "
                "estimate_channel_rg_cov_effects() first."
            )
        return self._rg_theta

    def _rg_expectations(self):
        """Covariate-scaled per-gene expectations for both channels."""
        if self._rg_statistics is None or self._rg_theta is None:
            raise ValueError(
                "No r_g fit available. Call "
                "estimate_channel_rg_cov_effects() first."
            )
        stats = self._rg_statistics
        eta = pd.Series(
            self.cov_matrix.loc[stats["genes"]].values
            @ np.asarray(self.cov_effects)[1:]
            + np.asarray(self.cov_effects)[0],
            index=stats["genes"],
        )
        scale = np.exp(eta)
        return (
            stats["baseline_silent"] * scale,
            stats["baseline_non_silent"] * scale,
        )

    def compute_r_g_production(self):
        """Per-gene ``r_g`` from **both** channels -- the paper number.

        Wraps :func:`estimate_rg.r_g_production`. Use this for
        published rates; never for a reported R², which must use
        :meth:`compute_r_g_for_evaluation` instead.
        """
        from .estimate_rg import r_g_production

        expected_silent, expected_non_silent = self._rg_expectations()
        return r_g_production(
            counts_silent=self._rg_statistics["counts_silent"],
            counts_non_silent=self._rg_statistics[
                "counts_non_silent"
            ],
            expected_silent=expected_silent,
            expected_non_silent=expected_non_silent,
            theta=self.rg_theta,
        )

    def compute_r_g_for_evaluation(self):
        """Per-gene ``r_g`` from the **silent channel only**.

        Wraps :func:`estimate_rg.r_g_silent_only_for_evaluation`,
        which has no argument through which non-silent data could
        reach it. Scale ``μ^(nonsyn)`` by this before scoring against
        non-silent counts, and the score is honest by construction.
        """
        from .estimate_rg import r_g_silent_only_for_evaluation

        expected_silent, _ = self._rg_expectations()
        return r_g_silent_only_for_evaluation(
            counts_silent=self._rg_statistics["counts_silent"],
            expected_silent=expected_silent,
            theta=self.rg_theta,
        )

    def _resolve_covariate_bounds(
        self, coeffs_shape, lower_value, upper_value
    ):
        """Broadcast configured bounds to match coefficient shape."""
        upper_array = None
        if upper_value is not None:
            upper_array = self._broadcast_bounds_value(
                upper_value, coeffs_shape
            )
        else:
            logger.warning(
                "upper_bounds_c was None; skipping bounds proximity "
                "checks."
            )
            return None, None

        if lower_value is None:
            lower_array = (
                -upper_array if upper_array is not None else None
            )
        else:
            lower_array = self._broadcast_bounds_value(
                lower_value, coeffs_shape
            )

        if lower_array is None or upper_array is None:
            logger.warning(
                "Unable to broadcast coefficient bounds to shape %s; "
                "skipping boundary proximity checks.",
                coeffs_shape,
            )
        return lower_array, upper_array

    def _broadcast_bounds_value(self, value, shape):
        """Broadcast a bounds value to the coefficient shape."""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.to_numpy()
        arr = np.asarray(value, dtype=float)
        try:
            return np.broadcast_to(arr, shape)
        except ValueError:
            logger.warning(
                "Could not broadcast bounds %s to shape %s",
                arr,
                shape,
            )
            return None

    def _warn_if_near_bounds(
        self,
        lower_candidate,
        upper_candidate,
        lower_bounds,
        upper_bounds,
        tol,
        mode_desc,
    ):
        """Warn if coefficients or HDIs are close to parameter bounds."""
        if lower_bounds is None or upper_bounds is None:
            return

        expected_size = int(np.prod(lower_bounds.shape))
        flat_labels = self._coefficient_labels(lower_bounds.shape)

        lower_candidate = np.asarray(lower_candidate, dtype=float)
        upper_candidate = np.asarray(upper_candidate, dtype=float)

        if lower_candidate.size != expected_size:
            logger.warning(
                "Expected %d lower-side values for bounds check but "
                "got %d; skipping.",
                expected_size,
                lower_candidate.size,
            )
            return
        if upper_candidate.size != expected_size:
            logger.warning(
                "Expected %d upper-side values for bounds check but "
                "got %d; skipping.",
                expected_size,
                upper_candidate.size,
            )
            return

        lower_candidate = lower_candidate.reshape(lower_bounds.shape)
        upper_candidate = upper_candidate.reshape(upper_bounds.shape)

        lower_mask = (
            np.isfinite(lower_candidate)
            & np.isfinite(lower_bounds)
            & ((lower_candidate - lower_bounds) <= tol)
        )
        upper_mask = (
            np.isfinite(upper_candidate)
            & np.isfinite(upper_bounds)
            & ((upper_bounds - upper_candidate) <= tol)
        )

        near_lower = np.flatnonzero(lower_mask.ravel())
        near_upper = np.flatnonzero(upper_mask.ravel())

        messages = []
        if near_lower.size:
            labels = ", ".join(flat_labels[i] for i in near_lower)
            messages.append(f"lower bounds ({labels})")
        if near_upper.size:
            labels = ", ".join(flat_labels[i] for i in near_upper)
            messages.append(f"upper bounds ({labels})")

        if messages:
            warn_msg = (
                f"{mode_desc} are within {tol:.3g} of the parameter "
                f"bounds for {', '.join(messages)}. "
                "Consider rerunning estimate_cov_effects() with "
                "adjusted lower_bounds_c/upper_bounds_c."
            )
            log_fn = getattr(logger, "warming", logger.warning)
            log_fn(warn_msg)

    def _coefficient_labels(self, shape):
        """Return human-readable labels for coefficients."""
        covariate_labels = ["intercept"] + list(self.covariate_names)

        if len(shape) == 1:
            labels = []
            for idx in range(shape[0]):
                if idx < len(covariate_labels):
                    labels.append(covariate_labels[idx])
                else:
                    labels.append(f"c[{idx}]")
            return labels

        if len(shape) == 2:
            signature_labels = self._signature_labels(shape[0])
            labels = []
            for sig_idx, sig_label in enumerate(signature_labels):
                for coef_idx in range(shape[1]):
                    if coef_idx < len(covariate_labels):
                        coef_label = covariate_labels[coef_idx]
                    else:
                        coef_label = f"c[{coef_idx}]"
                    labels.append(f"{sig_label}:{coef_label}")
            return labels

        total = int(np.prod(shape))
        return [f"c[{i}]" for i in range(total)]

    def _signature_labels(self, n_signatures):
        """Return signature labels if base_mus are signature-specific."""
        if isinstance(self._base_mus, dict):
            signature_names = list(self._base_mus.keys())
            if len(signature_names) == n_signatures:
                return signature_names
        return [f"signature_{i}" for i in range(n_signatures)]

    def estimate_passenger_genes_r2(
        self,
        sample_weights=None,
        excluded_samples=None,
        target="any",
        gene_scaling=None,
    ):
        """Estimate R² for passenger gene mutation frequency predictions.

        This method evaluates model performance on passenger genes
        by comparing predicted mutation frequency (number of samples
        with mutations per gene) with observed frequency. Passenger
        genes are those not in the Cancer Gene Census, which are
        assumed to be under neutral selection.

        The R² metric quantifies how well the model's predicted
        mutation rates explain the observed mutation frequency across
        passenger genes (gene-level evaluation).

        If `mu_gs` have not been computed yet, this method
        automatically calls `compute_mu_gs()` first. For models
        without covariates, an info message is logged to indicate
        that baseline mutation rates are being used.

        The result is stored in `self.passenger_genes_r2`.

        Parameters
        ----------
        sample_weights : pd.Series or None, default None
            Optional per-sample weight (indexed by
            ``Tumor_Sample_Barcode``, matching `genes_present`'s
            columns), applied to both the observed and expected sums
            before computing R². Default (None) weights every sample
            equally (today's behavior). Intended for samples flagged
            by :mod:`sample_qc` as lower-confidence, rather than
            dropping them outright -- see `excluded_samples` for that.
            Samples missing from `sample_weights` get weight 1.
        excluded_samples : collection of str or None, default None
            Sample barcodes to exclude entirely from both sums before
            computing R² (e.g. samples flagged by
            :func:`sample_qc.combine_sample_flags`, when dropping
            rather than downweighting). Applied before
            `sample_weights`.
        target : {"any", "non_silent", "non_silent_counts"}, default "any"
            Which observation to score against.

            - ``"any"`` (default, today's behavior): "gene mutated at
              all", i.e. `dataset.genes_present`, predicted from the
              merged `mu_gs`.
            - ``"non_silent"``: only non-silent mutations, i.e.
              `dataset.genes_present_non_silent`, predicted from the
              **non-synonymous channel's** rates
              (:meth:`compute_channel_mu_gs`) rather than the merged
              ones -- predicting a non-silent target from a total rate
              would be biased high by construction.

            - ``"non_silent_counts"``: the same channel, but scored
              on mutation *counts* rather than presence --
              `dataset.genes_counts_non_silent` against
              ``Σ_j μ_g^(nonsyn,j)``, with **no** ``1 - exp(-μ)``
              step, since a count target has no censoring. This is
              the headline for a Poisson (count) fit; using the
              presence formula on it would cap every gene's
              prediction at the number of samples and bias exactly
              the high-rate genes the target exists to measure.

            The presence targets are not interchangeable: ≈25% of
            presence events in the "any" target involve a silent
            mutation, which is exactly the part any silent-driven
            per-gene correction can leak into. Report the non-silent
            number whenever the model being scored has seen silent
            counts, and match the target to the likelihood the model
            was fit with.
        gene_scaling : pd.Series or None, default None
            Optional per-gene multiplier applied to the rates before
            scoring -- in practice, a ``r_g``. Genes absent from it
            get 1.0.

            There is deliberately no "use r_g" flag: the caller passes
            the ``r_g`` it means, and the two constructors are named
            so the choice is legible at the call site --
            ``gene_scaling=model.compute_r_g_for_evaluation()`` for a
            reportable R², ``compute_r_g_production()`` for the
            published rates. A scaled result is **not stored** in any
            attribute, only returned, so it can never be mistaken for
            the model's own unscaled R².

        Returns
        -------
        float
            R² score (coefficient of determination) for passenger
            genes. Values range from -∞ to 1, where:
            - 1.0: Perfect predictions
            - 0.0: Model performs as well as predicting the mean
            - < 0: Model performs worse than predicting the mean

        Raises
        ------
        ValueError
            If base_mus have not been computed (needed to compute
            mu_gs if not already available). Call compute_base_mus()
            first.
        ValueError
            If genes_present has not been computed in the dataset.
            Call dataset.compute_gene_presence() first.

        Notes
        -----
        **Method:**

        1. Identify passenger genes using Cancer Gene Census
        2. Restrict mu_gs and genes_present to passenger genes only
        3. For each passenger gene, sum observed presence across all
           samples:
           present_sum_g = Σ_j I[gene g mutated in sample j]
        4. For each passenger gene, compute expected number of samples
           where the gene is mutated:
           expected_g = Σ_j (1 - exp(-μ_g^(j)))
        5. Compute R² between expected_g and present_sum_g across all
           passenger genes

        **Interpretation:**

        High R² (close to 1) indicates the model's predicted
        mutation rates accurately capture the observed mutation
        frequency (number of samples with mutations) for individual
        passenger genes. Low or negative R² suggests poor model fit.

        This gene-level evaluation assesses whether the model
        correctly predicts which genes are mutated more frequently
        across samples. Since passenger genes are assumed to be
        under neutral selection (not positively or negatively
        selected), they provide a clean test set for evaluating the
        mutation rate model without confounding selection effects.

        **Typical workflow:**

        1. Compute baseline mutation rates:
            >>> model.compute_mu_taus()
            >>> model.compute_base_mus()

        2. (Optional) Estimate covariate effects:
            >>> model.estimate_cov_effects()  # Also computes R²

        3. Evaluate model performance (if not done by estimate_cov_effects):
            >>> r2 = model.estimate_passenger_genes_r2()
            >>> print(f"Passenger genes R²: {r2:.4f}")

        Note: You don't need to explicitly call `compute_mu_gs()`
        before this method, as it will be called automatically if
        needed.

        Examples
        --------
        >>> # Evaluate baseline model (no covariates)
        >>> model_no_cov = Model(dataset, None)
        >>> model_no_cov.compute_mu_taus()
        >>> model_no_cov.compute_base_mus()
        >>> # No need to call compute_mu_gs(), it's automatic
        >>> r2_baseline = model_no_cov.estimate_passenger_genes_r2()
        >>> print(f"Baseline R²: {r2_baseline:.4f}")
        >>>
        >>> # Evaluate model with covariates
        >>> model_with_cov = Model(dataset)
        >>> model_with_cov.assign_cov_matrix(cov_matrix)
        >>> model_with_cov.compute_mu_taus()
        >>> model_with_cov.compute_base_mus()
        >>> # estimate_cov_effects() automatically computes R²
        >>> model_with_cov.estimate_cov_effects()
        >>> r2_with_cov = model_with_cov.passenger_genes_r2  # Already set
        >>> print(f"With covariates R²: {r2_with_cov:.4f}")
        >>> print(f"Improvement: {r2_with_cov - r2_baseline:.4f}")
        >>>
        >>> # Compare multiple models
        >>> models = {
        ...     'baseline': model_no_cov,
        ...     'mrt': model_mrt,
        ...     'mrt+gexp': model_mrt_gexp}
        >>> for name, model in models.items():
        ...     r2 = model.estimate_passenger_genes_r2()
        ...     print(f"{name}: R² = {r2:.4f}")

        See Also
        --------
        compute_mu_gs : Must be called first to compute mutation rates
        estimate_presence.filter_passenger_genes_ensembl : Identifies
            passenger genes
        """
        from sklearn.metrics import r2_score

        from .estimate_presence import filter_passenger_genes_ensembl

        if target not in (
            "any",
            "non_silent",
            "non_silent_counts",
        ):
            raise ValueError(
                f"Unknown target {target!r}; expected 'any', "
                "'non_silent' or 'non_silent_counts'."
            )

        # Check if mu_gs need to be computed
        if self._mu_gs is None:
            # Check if this is a model without covariates
            if self.cov_matrix is None:
                logger.info(
                    "Model has no covariate matrix. Computing mu_gs "
                    "with baseline mutation rates only (no covariate "
                    "effects)."
                )

            # Ensure base_mus are available
            if self._base_mus is None:
                raise ValueError(
                    "Baseline mutation rates (base_mus) not computed. "
                    "Call compute_base_mus() first."
                )

            # Compute mu_gs
            self.compute_mu_gs()

        # Ensure the observation matrix this target scores against
        # has been computed
        if target == "any":
            if self.dataset._genes_present is None:
                raise ValueError(
                    "Gene presence matrix not computed in dataset. "
                    "Call dataset.compute_gene_presence() first."
                )
            observed_source = self.dataset.genes_present
            rates = self._mu_gs
        elif target == "non_silent":
            if self.dataset._genes_present_non_silent is None:
                raise ValueError(
                    "Non-silent gene presence matrix not computed in "
                    "dataset. Call "
                    "dataset.compute_gene_presence_non_silent() "
                    "first."
                )
            observed_source = self.dataset.genes_present_non_silent
            # Score the non-silent target against the non-silent
            # channel's own rate, not the merged one.
            rates = self.compute_channel_mu_gs("nonsyn")
        else:  # non_silent_counts
            if self.dataset._genes_counts_non_silent is None:
                raise ValueError(
                    "Non-silent gene count matrix not computed in "
                    "dataset. Call "
                    "dataset.compute_gene_counts_channels() first."
                )
            observed_source = self.dataset.genes_counts_non_silent
            rates = self.compute_channel_mu_gs("nonsyn")

        if gene_scaling is not None:
            rates = rates.mul(
                gene_scaling.reindex(rates.index, fill_value=1.0),
                axis=0,
            )

        # Identify passenger genes
        passenger_gene_ids = filter_passenger_genes_ensembl(
            rates.index
        )

        # Restrict to passenger genes. genes_present's crosstab is
        # built only from genes that appear at least once in the
        # mutation database, so under gene_universe="wes_target"
        # mu_gs.index can include genes absent from genes_present --
        # by construction, genes never observed as mutated in this
        # cohort (not an unknown/missing state), so their observed
        # presence is 0 in every sample, not a lookup error.
        mu_gs_passenger = rates.loc[passenger_gene_ids]
        genes_present_passenger = observed_source.reindex(
            passenger_gene_ids, fill_value=0
        )

        # A NaN rate is always a bug (most often a cov_matrix with NaN
        # rows for genes that have no covariates, whose exp(η) then
        # poisons mu_gs), and it fails *silently* below: pandas'
        # `.sum()` skips NaN, so such a gene is scored as expecting 0
        # mutations rather than raising. Caught the hard way while
        # reproducing a saved COAD fit locally -- it cost 0.08 of R²
        # with no error anywhere.
        n_nan = int(mu_gs_passenger.isna().values.sum())
        if n_nan:
            n_genes = int(mu_gs_passenger.isna().any(axis=1).sum())
            raise ValueError(
                f"{n_nan} NaN rate(s) across {n_genes} passenger "
                "gene(s) in the matrix being scored. Summing would "
                "silently treat those genes as expecting 0 mutations "
                "and report a too-low R². The usual cause is a "
                "cov_matrix carrying NaN rows for genes without "
                "covariate data -- drop them before fitting."
            )

        if excluded_samples is not None:
            keep = mu_gs_passenger.columns.difference(
                excluded_samples
            )
            mu_gs_passenger = mu_gs_passenger[keep]
            genes_present_passenger = genes_present_passenger[keep]

        if sample_weights is not None:
            weights = sample_weights.reindex(
                mu_gs_passenger.columns, fill_value=1.0
            )
        else:
            weights = pd.Series(1.0, index=mu_gs_passenger.columns)

        # Sum observed across all samples for each gene -- presence
        # events for the presence targets, mutation counts for the
        # counts target.
        present_sum = genes_present_passenger.mul(
            weights, axis=1
        ).sum(
            axis=1
        )  # Sum over genes

        if target == "non_silent_counts":
            # No censoring in a count target, so no 1 - exp(-μ): the
            # expected number of mutations is the rate itself. Using
            # the presence formula here would cap every gene's
            # prediction at the sample count and bias exactly the
            # high-rate genes the count target exists to measure.
            expected = mu_gs_passenger.mul(weights, axis=1).sum(
                axis=1
            )
        else:
            # Convert mutation rates to presence probabilities and sum
            # across all samples for each gene
            expected = (
                (1 - np.exp(-mu_gs_passenger))
                .mul(weights, axis=1)
                .sum(axis=1)
            )

        # Compute R² between expected and observed
        r2 = r2_score(present_sum, expected)

        # Store result. The targets are kept in separate attributes on
        # purpose: `passenger_genes_r2` is the number every existing
        # caller (and every saved model) already means by "the R²",
        # and silently redefining it depending on the last call's
        # `target` is exactly the kind of ambiguity this evaluation is
        # supposed to remove.
        # A gene_scaling'd number is a different quantity (it
        # depends on which r_g the caller chose), so it is returned
        # and never stored -- nothing downstream can then read it as
        # the model's own R².
        if gene_scaling is None:
            if target == "any":
                self._passenger_genes_r2 = r2
            elif target == "non_silent":
                self._passenger_genes_r2_non_silent = r2
            else:
                self._passenger_genes_r2_non_silent_counts = r2

        return r2

    def aggregate_signatures(
        self, signature_selection, include_other=False
    ):
        """Aggregate signature-separated base_mus into chosen signatures.

        This method allows you to combine multiple related signatures
        (e.g., SBS10a, SBS10b, SBS10c) into aggregate signatures
        (e.g., SBS10), and optionally group all remaining signatures
        into an "other" category.

        The aggregated result replaces self._base_mus. Equivalent to
        passing ``signature_selection``/``include_other`` directly to
        :meth:`Model.__init__`, which calls this method automatically
        as part of construction.

        Parameters
        ----------
        signature_selection : list
            List of signatures to keep/aggregate. Each element can be:

            - **Individual signature** (str): Keep as-is if exact match
              exists, e.g., 'SBS1'

            - **Aggregation pattern** (str): Aggregate all signatures
              starting with this prefix, e.g., 'SBS10' will aggregate
              'SBS10a', 'SBS10b', 'SBS10c', etc.

            - **Tuple of signatures** (tuple or list): Aggregate
              multiple specific signatures into one, e.g.,
              ('SBS1', 'SBS5') or ['SBS1', 'SBS5']
              The aggregated signature will be named by joining with
              '+', e.g., 'SBS1+SBS5'

        include_other : bool, default False
            If True, create an 'other' category containing the sum
            of all signatures not included in signature_selection.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping aggregated signature names to
            DataFrames (genes × samples). This is also stored in
            self._base_mus.

        Raises
        ------
        ValueError
            If base_mus is not signature-dependent (not a dict).
        ValueError
            If base_mus have not been computed.

        Notes
        -----
        **Aggregation logic:**

        1. **Exact matches**: If a signature in signature_selection
           exactly matches a key in base_mus, it's kept as-is.

        2. **Prefix matching**: If a signature doesn't have an exact
           match, all signatures starting with that prefix are
           aggregated. For example, 'SBS10' will aggregate 'SBS10a',
           'SBS10b', 'SBS10c', 'SBS10d', etc.

        3. **Explicit grouping**: Tuples/lists in signature_selection
           explicitly specify which signatures to aggregate together.

        4. **Other category**: If include_other=True, all signatures
           not matched by signature_selection are summed into 'other'.

        **Memory management:**

        This operation modifies self._base_mus in place. The original
        signature-separated base_mus are replaced with the aggregated
        version. If you need to preserve the original, use model.copy()
        first.

        Examples
        --------
        >>> # Basic aggregation
        >>> model.aggregate_signatures(['SBS1', 'SBS5', 'SBS10'])
        >>> # This keeps SBS1, SBS5, and aggregates SBS10a+SBS10b+SBS10c->SBS10
        >>>
        >>> # Explicit grouping of signatures
        >>> model.aggregate_signatures([
        ...     'SBS1',
        ...     'SBS10',
        ...     ('SBS5', 'SBS44')])
        >>> # Result: SBS1, SBS10 (agg), SBS5+SBS44 (agg)
        >>>
        >>> # Include all other signatures
        >>> model.aggregate_signatures(
        ...     ['SBS1', 'SBS5', 'SBS10'],
        ...     include_other=True)
        >>> # Result: SBS1, SBS5, SBS10 (agg), other (all remaining)
        >>>
        >>> # Complex example from main.py
        >>> sig_selection = [
        ...     'SBS5', 'SBS1', 'SBS44', 'SBS10a', 'SBS10b', 'SBS15']
        >>> model.aggregate_signatures(sig_selection)
        >>> model.aggregate_signatures(['SBS10'])  # Further aggregate SBS10a+SBS10b
        >>> # Or in one step:
        >>> sig_selection = ['SBS5', 'SBS1', 'SBS44', 'SBS10', 'SBS15']
        >>> model.aggregate_signatures(sig_selection, include_other=True)

        See Also
        --------
        compute_base_mus : Must be called first with separate_per_sigma=True
        copy : Create a copy before aggregating to preserve original
        """
        # Ensure base_mus have been computed
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first."
            )

        # Ensure base_mus are signature-dependent
        if not isinstance(self._base_mus, dict):
            raise TypeError(
                "base_mus must be signature-dependent (dict) to "
                "aggregate signatures. Current base_mus is a "
                "DataFrame (signature-independent). "
                "To create signature-separated base_mus, use "
                "model.compute_mu_taus(separate_per_sigma=True) "
                "before compute_base_mus()."
            )

        # Replace base_mus with aggregated version
        self._base_mus = _aggregate_signature_dict(
            self._base_mus,
            signature_selection,
            include_other=include_other,
        )

        return self._base_mus
