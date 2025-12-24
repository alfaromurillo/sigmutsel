"""Data models for mutation rate analysis."""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import logging
import inspect
import json
from pathlib import Path


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
        in a gene's coding sequence. Automatically restricted to genes
        present in the mutation database. Lazy-loaded via
        generate_contexts_by_gene(). These values are needed
        to compute later the probability that a mutation of a certain
        type lands on a gene. Thus, gene mutation rates can only be
        obtained for genes with this information, and so the index of
        contexts_by_gene is the maximal scope of the analysis.
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

    # Lazy-loaded attributes
    _mutation_db: pd.DataFrame = None
    _genes_present: pd.DataFrame = None
    _genes_present_non_silent: pd.DataFrame = None
    _variant_db: pd.DataFrame = None
    _variants_present: pd.DataFrame = None
    _sig_assignments: pd.DataFrame = None
    _signature_matrix: pd.DataFrame = None
    _signature_reference_genome: str | None = None
    _signature_exome: bool | None = None
    _signature_cosmic_version: float | None = None
    _signature_genome_build: str | None = None
    _contexts_by_gene: pd.DataFrame = None
    dataset_directory: str | None = field(
        default=None, init=False, repr=False)

    def __repr__(self):
        """Show loaded status of lazy attributes (custom repr)."""
        loaded = []
        if self._mutation_db is not None:
            loaded.append(
                f"mutation_db: {self._mutation_db.shape[0]} rows")
        if self._genes_present is not None:
            loaded.append(
                f"genes_present: {self._genes_present.shape}")
        if self._genes_present_non_silent is not None:
            loaded.append(
                f"genes_present_non_silent: "
                f"{self._genes_present_non_silent.shape}")
        if self._variant_db is not None:
            loaded.append(
                f"variant_db: {self._variant_db.shape[0]} variants")
        if self._variants_present is not None:
            loaded.append(
                f"variants_present: {self._variants_present.shape}")
        if self._sig_assignments is not None:
            loaded.append(
                f"sig_assignments: {self._sig_assignments.shape}")
        if self._signature_matrix is not None:
            loaded.append(
                f"signature_matrix: {self._signature_matrix.shape}")
        if self._contexts_by_gene is not None:
            loaded.append(
                f"contexts_by_gene: {self._contexts_by_gene.shape}")

        loaded_str = "\n  ".join(loaded) if loaded else "None"

        return (
            f"MutationDataset(\n"
            f"  signature_class={self.signature_class!r}\n"
            f"  location_maf_files={self.location_maf_files!r}\n"
            f"  loaded_data:\n  {loaded_str}\n"
            f")")

    def save_dataset(self, directory):
        """Persist loaded dataset artifacts to a directory."""
        import json

        directory = Path(directory)
        manifest_path = directory / "dataset_manifest.json"

        if manifest_path.exists():
            response = input(
                f"Dataset already exists at {directory}. "
                "Overwrite? [y/N]: ").strip().lower()
            if response not in {"y", "yes"}:
                raise FileExistsError(
                    f"Dataset directory {directory} already exists.")

        directory.mkdir(parents=True, exist_ok=True)

        data_specs = [
            ('mutation_db',
             '_mutation_db',
             'mutation_db.parquet',
             'parquet'),
            ('genes_present',
             '_genes_present',
             'genes_present.parquet',
             'parquet'),
            ('genes_present_non_silent',
             '_genes_present_non_silent',
             'genes_present_non_silent.parquet',
             'parquet'),
            ('variant_db',
             '_variant_db',
             'variant_db.parquet',
             'parquet'),
            ('variants_present',
             '_variants_present',
             'variants_present.parquet',
             'parquet'),
            ('sig_assignments',
             '_sig_assignments',
             'sig_assignments.parquet',
             'parquet'),
            ('signature_matrix',
             '_signature_matrix',
             'signature_matrix.parquet',
             'parquet'),
            ('contexts_by_gene',
             '_contexts_by_gene',
             'contexts_by_gene.csv',
             'csv'),
        ]

        saved_files = {}

        for public_name, private_name, filename, fmt in data_specs:
            value = getattr(self, private_name)
            if value is None:
                continue

            file_path = directory / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if fmt == 'parquet':
                # Convert list columns to JSON strings for parquet compatibility
                value_to_save = value.copy()
                for col in value_to_save.columns:
                    if value_to_save[col].apply(lambda x: isinstance(x, list)).any():
                        value_to_save[col] = value_to_save[col].apply(
                            lambda x: json.dumps(x) if isinstance(x, list) else x)
                value_to_save.to_parquet(file_path)
            elif fmt == 'csv':
                value.to_csv(file_path)
            else:
                raise ValueError(f"Unsupported format {fmt} for {public_name}")

            saved_files[public_name] = {
                "filename": filename,
                "format": fmt,
            }

        manifest = {
            "version": 1,
            "signature_class": self.signature_class,
            "location_maf_files": str(self.location_maf_files),
            "files": saved_files,
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
        import json

        directory = Path(directory)
        manifest_path = directory / "dataset_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at {manifest_path}. "
                "Ensure save_dataset() was run first.")

        manifest = json.loads(manifest_path.read_text())
        dataset = cls(
            location_maf_files=manifest.get("location_maf_files"),
            signature_class=manifest.get("signature_class", "SBS"))

        for attr_name, info in manifest.get("files", {}).items():
            filename = info["filename"]
            fmt = info.get("format", Path(filename).suffix.lstrip("."))
            file_path = directory / filename

            if fmt == 'parquet':
                value = pd.read_parquet(file_path)
                # Convert JSON strings back to lists for columns like mut_types
                for col in value.columns:
                    # Check if column contains JSON array strings
                    sample = value[col].dropna().iloc[0] if not value[col].dropna().empty else None
                    if isinstance(sample, str) and sample.startswith('['):
                        try:
                            value[col] = value[col].apply(
                                lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x)
                        except (json.JSONDecodeError, TypeError):
                            # If it's not valid JSON, leave it as is
                            pass
            elif fmt == 'csv':
                value = pd.read_csv(file_path, index_col=0)
            else:
                raise ValueError(
                    f"Unsupported file format {fmt} for {attr_name}")

            setattr(dataset, attr_name, value)

        signature_params = manifest.get("signature_parameters", {})
        dataset._signature_reference_genome = (
            signature_params.get("reference_genome"))
        dataset._signature_exome = signature_params.get("exome")
        dataset._signature_cosmic_version = (
            signature_params.get("cosmic_version"))
        dataset._signature_genome_build = (
            signature_params.get("genome_build"))

        dataset.dataset_directory = str(directory)
        return dataset

    @property
    def mutation_db(self):
        """Mutation database (lazy loaded)."""
        if self._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded. "
                "Call generate_mutation_db() or load_dataset() first.")
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
                "loaded. Call generate_mutation_db() or load_dataset() first.")
        return self._mutation_db['Tumor_Sample_Barcode'].nunique()

    @property
    def n_genes(self):
        """Number of genes in the dataset."""
        if self._genes_present is not None:
            return self._genes_present.shape[0]
        # Fallback: compute from mutation_db
        if self._mutation_db is None:
            raise ValueError(
                "Cannot compute n_genes: mutation database not "
                "loaded. Call generate_mutation_db() or load_dataset() first.")
        return self._mutation_db['ensembl_gene_id'].nunique()

    @property
    def genes_present(self):
        """Gene presence matrix (lazy loaded)."""
        if self._genes_present is None:
            raise ValueError(
                "Gene presence matrix not computed. "
                "Call compute_gene_presence() first.")
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
                "Call compute_gene_presence_non_silent() first.")
        return self._genes_present_non_silent

    @genes_present_non_silent.setter
    def genes_present_non_silent(self, value):
        """Set non-silent gene presence matrix."""
        self._genes_present_non_silent = value

    @property
    def variant_db(self):
        """Variant database (lazy loaded)."""
        if self._variant_db is None:
            logger.warning(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first.")
            raise ValueError(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first.")
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
                "Call compute_variants_present() first.")
        return self._variants_present

    @variants_present.setter
    def variants_present(self, value):
        """Set variant presence matrix."""
        self._variants_present = value

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
                "load_dataset() first.")
        return self._mutation_db['variant'].nunique()

    @property
    def variant_counts(self):
        """Number of tumors each variant appears in.

        Returns
        -------
        pd.Series
            Variant counts sorted descending by frequency.
        """
        return (
            self.mutation_db
            .groupby("variant")["Tumor_Sample_Barcode"]
            .nunique()
            .sort_values(ascending=False))

    @property
    def gene_counts(self):
        """Number of tumors each gene is mutated in.

        Returns
        -------
        pd.Series
            Gene counts sorted descending by frequency.
        """
        return (
            self.mutation_db
            .groupby("gene")["Tumor_Sample_Barcode"]
            .nunique()
            .sort_values(ascending=False))

    @property
    def variant_type_counts(self):
        """Number of different types each variant has.

        Returns
        -------
        pd.Series
            Variant type counts sorted descending.
        """
        return (
            self.mutation_db
            .groupby("variant")["type"]
            .nunique()
            .sort_values(ascending=False))

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
            self.mutation_db
            .groupby('variant')
            .agg(num_types=('type', 'nunique'),
                 num_tumors=('Tumor_Sample_Barcode', 'nunique'))
            .reset_index()
            .sort_values(by=['num_types', 'num_tumors'],
                         ascending=[False, False])
            .set_index('variant'))

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
        if (self.signature_class == 'ID' and
                'seqinfo_dir' not in kwargs):
            kwargs['seqinfo_dir'] = (
                Path(self.location_maf_files) / "output" /
                "vcf_files" / "ID")

        # Use generate_compact_db and store result
        self._mutation_db = generate_compact_db(
            self.location_maf_files,
            signature_class=self.signature_class,
            location_gene_set=location_gene_set,
            **kwargs)

    def has_gene_presence(self):
        """Check if gene presence matrix has been computed."""
        return self._genes_present is not None

    def has_non_silent_presence(self):
        """Check if non-silent gene presence has been computed."""
        return self._genes_present_non_silent is not None

    def has_variants(self):
        """Check if variants have been loaded."""
        return self._variant_db is not None

    def generate_variant_db(
            self,
            position_tolerance=3):
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
            extract_variants_from_db,
            annotate_variants_with_types)

        if self._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded. "
                "Call generate_mutation_db() or load_dataset() first.")

        variants = extract_variants_from_db(
            self.mutation_db,
            position_tolerance=position_tolerance)
        variants = annotate_variants_with_types(
            variants,
            self.mutation_db)

        self._variant_db = variants
        return variants

    def compute_gene_presence(self):
        """Compute gene presence matrix from mutation database.

        Calls compute_genes_present() to create a binary matrix
        indicating which genes are mutated in which samples.
        """
        from .estimate_presence import compute_genes_present

        self._genes_present = compute_genes_present(
            self.mutation_db)

    def compute_gene_presence_non_silent(self):
        """Compute non-silent gene presence matrix.

        Calls compute_genes_present() with scope='non-silent'
        to create a binary matrix for non-silent mutations only.
        """
        from .estimate_presence import compute_genes_present

        self._genes_present_non_silent = (
            compute_genes_present(
                self.mutation_db,
                scope='non-silent'))

    def compute_variants_present(self):
        """Compute variant presence matrix.

        Requires variant_db to be loaded first (via
        generate_variant_db() or load_dataset()).
        """
        from .estimate_presence import compute_variants_present

        if self._variant_db is None:
            raise ValueError(
                "Variants database not loaded. "
                "Call generate_variant_db() or load_dataset() first.")

        self._variants_present = compute_variants_present(
            self.mutation_db,
            self._variant_db)

    @property
    def sig_assignments(self):
        """Signature assignments (lazy loaded)."""
        if self._sig_assignments is None:
            raise ValueError(
                "Signature assignments not loaded. "
                "Call run_signature_decomposition() first.")
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
                "Call run_signature_decomposition() first.")
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
                "Call run_signature_decomposition() first.")
        return self._signature_reference_genome

    @property
    def signature_exome(self):
        """Whether exome-normalized signatures were used."""
        if self._signature_exome is None:
            raise ValueError(
                "Exome flag not recorded. "
                "Call run_signature_decomposition() first.")
        return self._signature_exome

    @property
    def signature_cosmic_version(self):
        """COSMIC version used for signature decomposition."""
        if self._signature_cosmic_version is None:
            raise ValueError(
                "COSMIC version not recorded. "
                "Call run_signature_decomposition() first.")
        return self._signature_cosmic_version

    @property
    def signature_genome_build(self):
        """Genome build used for signature decomposition."""
        if self._signature_genome_build is None:
            raise ValueError(
                "Genome build not recorded. "
                "Call run_signature_decomposition() first.")
        return self._signature_genome_build

    @property
    def contexts_by_gene(self):
        """Trinucleotide context counts by gene (lazy loaded)."""
        if self._contexts_by_gene is None:
            raise ValueError(
                "Contexts by gene not loaded. "
                "Call generate_contexts_by_gene() or load_dataset() first.")
        return self._contexts_by_gene

    @contexts_by_gene.setter
    def contexts_by_gene(self, value):
        """Set trinucleotide context counts by gene."""
        self._contexts_by_gene = value

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
            reference_genome='GRCh38',
            force_generation=False,
            **kwargs):
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
            mutational_matrices_generation)

        return mutational_matrices_generation(
            path_to_input_files=self.location_maf_files,
            reference_genome=reference_genome,
            force_generation=force_generation,
            **kwargs)

    def run_signature_decomposition(
            self,
            force_generation=False,
            exome=None,
            cosmic_version=None,
            genome_build='GRCh38',
            **kwargs):
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
        print("="*len(title))
        print(title)
        print("="*len(title))

        import logging
        from pathlib import Path
        from .signature_decomposition import (
            signature_decomposition as run_sig_decomp)

        logger = logging.getLogger(__name__)

        if exome is None:
            exome = kwargs.pop('exome', True)
        else:
            kwargs.pop('exome', None)

        if cosmic_version is None:
            cosmic_version = kwargs.pop('cosmic_version', 3.4)
        else:
            kwargs.pop('cosmic_version', None)

        if genome_build is None:
            genome_build = kwargs.pop('genome_build', 'GRCh38')
        else:
            kwargs.pop('genome_build', None)

        if not self.has_mutational_matrices():
            logger.info(
                "Mutational matrices not found. "
                "Generating them before signature decomposition...")
            matrix_kwargs = {}
            if exome is not None:
                matrix_kwargs['exome'] = exome
            self.generate_mutational_matrices(
                reference_genome=genome_build,
                force_generation=False,
                **matrix_kwargs)
            logger.info("...done.")
            print("")

        if not self.has_mutational_matrices():
            raise FileNotFoundError(
                f"Mutational matrices not found at "
                f"{self.location_maf_files}/output/. "
                "Automatic generation failed. "
                "Run generate_mutational_matrices() manually for "
                "more details.")

        # Determine which matrix to use based on signature_class
        matrix_map = {
            'SBS': 'SBS96',
            'DBS': 'DBS78',
            'ID': 'ID83',
        }

        # Provide helpful error for common mistakes
        if self.signature_class not in matrix_map:
            # Suggest correction for old aliases
            if self.signature_class == 'SNP':
                raise ValueError(
                    "signature_class='SNP' is not supported. "
                    "Use signature_class='SBS' instead. "
                    "Create a new dataset with: "
                    "MutationDataset(location_maf_files=..., "
                    "signature_class='SBS')")
            elif self.signature_class == 'INDEL':
                raise ValueError(
                    "signature_class='INDEL' is not supported. "
                    "Use signature_class='ID' instead. "
                    "Create a new dataset with: "
                    "MutationDataset(location_maf_files=..., "
                    "signature_class='ID')")
            else:
                raise ValueError(
                    f"Signature decomposition not supported for "
                    f"signature_class='{self.signature_class}'. "
                    f"Supported classes: {list(matrix_map.keys())}")

        matrix_resolution = matrix_map[self.signature_class]
        matrix_filename = (
            f"mutational_matrix.{matrix_resolution}.exome")

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
                "Run generate_mutational_matrices() first.")

        # Create results directory in standard location
        sig_decomp_dir = (
            Path(self.location_maf_files) / "signature_decomposition")
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
                f"force_generation=True.")
        else:
            logger.info(
                f"Running signature decomposition for "
                f"{self.signature_class} using matrix: {matrix_path}")

        # Set collapse_to_SBS96=False for ID and DBS
        collapse_to_SBS96 = (
            self.signature_class == 'SBS'
            if 'collapse_to_SBS96' not in kwargs
            else kwargs.pop('collapse_to_SBS96'))

        # Run signature decomposition
        self._sig_assignments = run_sig_decomp(
            results_dir=str(results_dir),
            input_data=str(matrix_path),
            input_type='matrix',
            collapse_to_SBS96=collapse_to_SBS96,
            force_generation=force_generation,
            exome=exome,
            cosmic_version=cosmic_version,
            genome_build=genome_build,
            **kwargs)

        if not results_exist:
            logger.info("... done with signature decomposition.")
            print("")

        # Load the normalized signature matrix
        sig_matrix_path = (
            solution_dir / "Signatures" /
            "Assignment_Solution_Signatures.txt")

        if sig_matrix_path.exists():
            self._signature_matrix = pd.read_csv(
                sig_matrix_path, sep='\t', index_col=0)
            logger.info(
                f"Loaded normalized signature matrix from "
                f"{sig_matrix_path}")
            print("")
        else:
            logger.warning(
                f"Signature matrix not found at {sig_matrix_path}")
            self._signature_matrix = None

        self._signature_reference_genome = genome_build
        self._signature_exome = exome
        self._signature_cosmic_version = cosmic_version
        self._signature_genome_build = genome_build

        print("")
        return self._sig_assignments

    def generate_contexts_by_gene(
            self,
            fastas=None):
        """Generate trinucleotide context counts by gene.

        Computes trinucleotide context counts from FASTA files and
        stores them in ``self._contexts_by_gene``. The computation is
        automatically restricted to genes present in the mutation
        database to keep runtime manageable.

        Parameters
        ----------
        fastas : str, Path, list, or None, default None
            Path to FASTA file(s) containing gene sequences. Can be:
            - Single FASTA file path (str or Path)
            - List of FASTA file paths
            - None: automatically uses locations.location_cds_fasta

        Returns
        -------
        pd.DataFrame
            DataFrame with genes as index and trinucleotide contexts
            as columns. Each cell contains the count of that context
            in that gene's sequence. Restricted to genes present in
            the mutation database.

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

        # Ensure mutation_db is loaded
        if self._mutation_db is None:
            raise ValueError(
                "Mutation database must be loaded before computing "
                "contexts. Call generate_mutation_db() or "
                "load_dataset() first.")

        # Compute contexts, restricting to genes in mutation_db
        self._contexts_by_gene = compute_contexts_by_gene(
            fastas,
            restrict_to_db=self.mutation_db)

        return self._contexts_by_gene

    def build_full_dataset(self, fastas=None):
        """Run the full data-generation pipeline for this dataset."""
        title = "Mutation data: building compact mutation database."
        print("="*len(title))
        print(title)
        print("="*len(title))
        self.generate_mutation_db()
        print("")

        title = "Gene presence: computing matrices."
        print("="*len(title))
        print(title)
        print("="*len(title))
        self.compute_gene_presence()
        self.compute_gene_presence_non_silent()
        print("")

        title = "Contexts by gene: computing opportunities."
        print("="*len(title))
        print(title)
        print("="*len(title))
        self.generate_contexts_by_gene(fastas=fastas)
        print("")

        title = "Variant data: generating annotations and presence."
        print("="*len(title))
        print(title)
        print("="*len(title))
        self.generate_variant_db()
        self.compute_variants_present()
        print("")


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
        ``prob_g_tau_tau_independent``, and ``signature_selection``
        can be provided at initialization to automatically run
        the corresponding setup steps (mutation burdens, baseline
        rates, and optional signature aggregation).
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
    cov_effects: np.ndarray = None
    _n_in_cov_effects_estimation: int = None
    _passenger_genes_r2: float = None
    cov_effects_posteriors: object = None
    _mu_gs: pd.DataFrame = None
    mu_ms: pd.DataFrame = None
    _mu_taus: pd.DataFrame | dict = None
    _prob_g_tau_tau_independent: bool | None = None
    gammas: dict = field(default_factory=dict, init=False, repr=False)
    _saved_location: str | None = field(default=None, init=False, repr=False)
    _auto_mu_taus_kwargs: dict = field(default_factory=dict, init=False, repr=False)
    _auto_cov_effects_per_sigma: bool | None = field(default=None, init=False, repr=False)
    _auto_prob_g_tau_tau_independent: bool | None = field(default=None, init=False, repr=False)
    _auto_signature_selection: list | None = field(default=None, init=False, repr=False)

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
            signature_selection: list | tuple | None = None):
        self.dataset = dataset
        self.cov_matrix = cov_matrix
        self.cov_effects_kwargs = (
            cov_effects_kwargs.copy() if cov_effects_kwargs else {})

        self._base_mus = None
        self.cov_effects = None
        self._n_in_cov_effects_estimation = None
        self._passenger_genes_r2 = None
        self.cov_effects_posteriors = None
        self._mu_gs = None
        self.mu_ms = None
        self._mu_taus = None
        self._prob_g_tau_tau_independent = None
        self.gammas = {}

        self._auto_mu_taus_kwargs = {
            "L_low": L_low,
            "L_high": L_high,
            "cut_at_L_low": cut_at_L_low}
        self._auto_cov_effects_per_sigma = cov_effects_per_sigma
        self._auto_prob_g_tau_tau_independent = (
            prob_g_tau_tau_independent)
        self._auto_signature_selection = (
            list(signature_selection)
            if signature_selection is not None else None)
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
            key: value for key, value in self._auto_mu_taus_kwargs.items()
            if value is not None}

        need_mu_taus = (
            bool(auto_mu_kwargs) or
            self._auto_cov_effects_per_sigma is not None or
            self._auto_prob_g_tau_tau_independent is not None or
            self._auto_signature_selection is not None)

        if need_mu_taus and self._mu_taus is None:
            separate = bool(self._auto_cov_effects_per_sigma)
            self.compute_mu_taus(
                separate_per_sigma=separate,
                **auto_mu_kwargs)

        if (self._auto_prob_g_tau_tau_independent is not None and
                self._base_mus is None):
            self.compute_base_mus(
                prob_g_tau_tau_independent=(
                    self._auto_prob_g_tau_tau_independent))

        if (self._auto_signature_selection and
                self._base_mus is None):
            self.compute_base_mus(
                prob_g_tau_tau_independent=False)

        if self._auto_signature_selection:
            self.aggregate_signatures(self._auto_signature_selection)

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
                    f"base_mus: dict with {len(self._base_mus)} signatures")
            else:
                loaded.append(f"base_mus: {self._base_mus.shape}")
        if self._mu_taus is not None:
            if isinstance(self._mu_taus, dict):
                loaded.append(
                    f"mu_taus: dict with {len(self._mu_taus)} signatures")
            else:
                loaded.append(f"mu_taus: {self._mu_taus.shape}")
        if self.cov_effects is not None:
            loaded.append(f"cov_effects: {self.cov_effects.shape}")
        if self._passenger_genes_r2 is not None:
            loaded.append(f"R²={self._passenger_genes_r2:.4f}")
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
                "n_in_cov_effects_estimation.")

        # If already set, return it
        if self._n_in_cov_effects_estimation is not None:
            return self._n_in_cov_effects_estimation

        # Otherwise compute it and warn
        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix not assigned. "
                "Call assign_cov_matrix() first.")

        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first.")

        # Identify passenger genes with complete covariate data
        passenger_gene_ids = filter_passenger_genes_ensembl(
            self.cov_matrix.index)

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
            n_complete)

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
                "estimate_passenger_genes_r2() now.")
            self.estimate_passenger_genes_r2()
            return self._passenger_genes_r2

        logger.info(
            "Passenger genes R² unavailable: "
            "estimate_cov_effects() has not been run yet.")

        # For models with covariates but no R² yet, return None (user
        # should call estimate_cov_effects or
        # estimate_passenger_genes_r2)
        return None

    def assign_cov_matrix(self, cov_matrix, run_pca=False,
                          pca_kwargs=None):
        """Assign covariate matrix, restricting to dataset genes.

        This method assigns a covariate matrix to the model after
        reindexing it to match the genes in contexts_by_gene.
        Optionally runs PCA on the covariates to reduce dimensionality.

        Note: This method is automatically called during Model
        initialization if a cov_matrix is provided to the constructor
        (e.g., Model(dataset, cov_matrix)).

        This ensures that:
        1. Only genes with context information are included
        2. Gene order matches dataset.contexts_by_gene.index
        3. Missing genes are handled appropriately
        4. (Optional) Covariates are transformed to principal components

        Parameters
        ----------
        cov_matrix : pd.DataFrame
            Covariate matrix with genes as index and covariates as
            columns. Index should be Ensembl gene IDs.
        run_pca : bool, default False
            If True, run PCA on the reindexed covariate matrix to
            reduce dimensionality. The resulting principal components
            replace the original covariates.
        pca_kwargs : dict or None, default None
            Keyword arguments passed to
            :func:`covariates_utilities.run_pca_on_covariates`.
            Common options:
            - n_components : int, number of PCs to keep
            - columns : list[str], subset of columns to include in PCA
            - standardize : bool, default True, z-score before PCA
            - dropna : str, default 'any', how to handle NaNs
            - **kwargs for sklearn.decomposition.PCA

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
        2. Run PCA using :func:`covariates_utilities.run_pca_on_covariates`
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
        covariates_utilities.run_pca_on_covariates : PCA implementation
        """
        # Ensure contexts_by_gene has been loaded
        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first.")

        # Reindex to match contexts_by_gene
        reindexed = cov_matrix.reindex(
            self.dataset.contexts_by_gene.index)

        # Optionally run PCA
        if run_pca:
            from .utils import run_pca_on_covariates

            if pca_kwargs is None:
                pca_kwargs = {}

            self.cov_matrix = run_pca_on_covariates(
                reindexed, **pca_kwargs)
        else:
            self.cov_matrix = reindexed

        return self.cov_matrix

    @property
    def base_mus(self):
        """Baseline mutation rates per gene per tumor (lazy loaded)."""
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates not computed. "
                "Call compute_base_mus() first.")
        return self._base_mus

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
                "load_dataset() to ensure gene indices match.")
            self._base_mus = value
            return

        # Validate index for DataFrames
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(
                    self.dataset.contexts_by_gene.index):
                logger.warning(
                    "base_mus index does not match "
                    "dataset.contexts_by_gene.index. "
                    "This may cause errors when computing mutation "
                    "rates. To fix: reindex base_mus to match "
                    "contexts_by_gene.index, or recompute base_mus "
                    "after loading contexts_by_gene.")

        # Validate index for dicts of DataFrames (signature-separated)
        elif isinstance(value, dict):
            for sig_name, df in value.items():
                if isinstance(df, pd.DataFrame):
                    if not df.index.equals(
                            self.dataset.contexts_by_gene.index):
                        logger.warning(
                            "base_mus['%s'] index does not match "
                            "dataset.contexts_by_gene.index. "
                            "This may cause errors when computing "
                            "mutation rates. To fix: reindex all "
                            "base_mus DataFrames to match "
                            "contexts_by_gene.index, or recompute "
                            "base_mus after loading contexts_by_gene.",
                            sig_name)
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
                "Call compute_base_mus() first.")
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
                "Call compute_base_mus() first.")
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
            other.cov_matrix.columns)

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
                "Assign a covariate matrix first.")

        if isinstance(covariates, str):
            covariate_set = {covariates}
        else:
            covariate_set = set(covariates)

        invalid = covariate_set - set(self.covariate_names)
        if invalid:
            raise ValueError(
                f"Covariate(s) not found in model: {sorted(invalid)}")

        remaining_columns = [
            col for col in self.covariate_names if col not in covariate_set]

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
            upper_bound_prior=0.5 * 10**3,
            store=True,
            non_silent=True):
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
        upper_bound_prior : float, optional
            Upper bound for gamma prior. Default 0.5 * 10**3.
        store : bool, optional
            Whether to store result in self.gammas. Default True.
        non_silent : bool, optional
            For genes, whether to use non-silent mutations only.
            Default True.

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
        from .estimate_gammas import estimate_gamma_from_mus

        # Auto-detect level if not specified
        if level is None:
            level = self._detect_item_level(item)

        if level == 'variant':
            result = self._estimate_gamma_variant(
                item,
                upper_bound_prior=upper_bound_prior,
                store=store)
        elif level == 'gene':
            result = self._estimate_gamma_gene(
                item,
                upper_bound_prior=upper_bound_prior,
                store=store,
                non_silent=non_silent)
        else:
            raise ValueError(
                f"Invalid level: {level!r}. "
                "Must be 'variant', 'gene', or None.")

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
            return 'variant'
        if (hasattr(self.dataset, '_variants_present') and
                self.dataset._variants_present is not None and
                item in self.dataset._variants_present.index):
            return 'variant'

        # Check if it's an ensembl_gene_id
        if self._mu_gs is not None and item in self.mu_gs.index:
            return 'gene'
        if (hasattr(self.dataset, '_genes_present') and
                self.dataset._genes_present is not None and
                item in self.dataset._genes_present.index):
            return 'gene'

        # Check if it's a gene name in mutation database
        if hasattr(self.dataset, 'mutation_db'):
            mapping = (
                self.dataset.mutation_db[['gene', 'ensembl_gene_id']]
                .drop_duplicates()
                .set_index('gene')['ensembl_gene_id'])
            if item in mapping.index:
                return 'gene'

        raise ValueError(
            f"Could not identify {item!r} as a variant or gene. "
            "Please specify 'level' parameter explicitly.")

    def _report_gamma_posterior(self, result):
        """Print posterior summary for gamma inference if available."""
        try:
            import arviz as az
        except ImportError:  # pragma: no cover - optional dependency
            logger.warning(
                "arviz is not installed; cannot summarize gamma posterior.")
            return

        if not hasattr(result, "posterior"):
            return

        try:
            summary = az.summary(result, var_names=['gamma'])
        except Exception as exc:  # pragma: no cover - logging path
            logger.warning(
                "Failed to summarize gamma posterior: %s", exc)
            return

        print("Gamma posterior summary:")
        print(summary.to_string())

    def _estimate_gamma_variant(
            self,
            variant,
            upper_bound_prior=0.5 * 10**3,
            store=True):
        """Estimate selection coefficient for a variant.

        Parameters
        ----------
        variant : str
            Variant identifier.
        upper_bound_prior : float, optional
            Upper bound for gamma prior.
        store : bool, optional
            Whether to store result.

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
                f"Variant {variant!r} not found in mutation rates.")

        variants_present = self.dataset.variants_present
        present_mask = variants_present.loc[variant] == 1
        absent_mask = ~present_mask

        result = estimate_gamma_from_mus(
            self.mu_ms.loc[variant][present_mask],
            self.mu_ms.loc[variant][absent_mask],
            upper_bound_prior=upper_bound_prior)

        if store:
            self.gammas[variant] = result

        return result

    def _estimate_gamma_gene(
            self,
            gene,
            upper_bound_prior=0.5 * 10**3,
            store=True,
            non_silent=True):
        """Estimate selection coefficient for a gene.

        Parameters
        ----------
        gene : str
            Gene name or ensembl_gene_id.
        upper_bound_prior : float, optional
            Upper bound for gamma prior.
        store : bool, optional
            Whether to store result.
        non_silent : bool, optional
            Whether to use non-silent mutations only.

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
            if non_silent else self.dataset.genes_present)

        # Try to get ensembl_gene_id if gene is a name
        if gene in self.mu_gs.index:
            gene_id = gene
        else:
            mapping = (
                self.dataset.mutation_db[['gene', 'ensembl_gene_id']]
                .drop_duplicates()
                .set_index('gene')['ensembl_gene_id'])
            if gene not in mapping:
                raise ValueError(
                    f"Gene {gene!r} not found in mutation database.")
            gene_id = mapping[gene]

        if gene_id not in self.mu_gs.index:
            raise ValueError(
                f"Gene ID {gene_id!r} not found in mu_gs.")

        present_mask = gene_presence.loc[gene_id] == 1
        absent_mask = ~present_mask

        result = estimate_gamma_from_mus(
            self.mu_gs.loc[gene_id][present_mask],
            self.mu_gs.loc[gene_id][absent_mask],
            upper_bound_prior=upper_bound_prior)

        if store:
            # Always store with ensembl_gene_id for consistency
            self.gammas[gene_id] = result

        return result

    def plot_gamma_results(
            self,
            keys=None,
            level=None,
            change_gene_ids_to_names=True,
            **kwargs):
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
        from .figures import plot_posteriors_vs_counts
        from .estimate_presence import filter_passenger_genes

        if not self.gammas:
            raise ValueError(
                "No gamma results to plot. "
                "Call estimate_gamma() first.")

        # Select results based on keys
        if keys is None:
            # Plot all results
            results = self.gammas
        elif isinstance(keys, str):
            # Single key
            if keys not in self.gammas:
                raise ValueError(
                    f"Key {keys!r} not found in gamma results.")
            results = {keys: self.gammas[keys]}
        else:
            # List of keys
            results = {}
            for key in keys:
                if key not in self.gammas:
                    raise ValueError(
                        f"Key {key!r} not found in gamma results.")
                results[key] = self.gammas[key]

        # Auto-detect level if not specified
        if level is None:
            # Check first key to determine level
            first_key = next(iter(results.keys()))
            if (hasattr(self.dataset, '_variants_present') and
                    self.dataset._variants_present is not None and
                    first_key in self.dataset._variants_present.index):
                level = 'variant'
            elif (hasattr(self.dataset, '_genes_present') and
                    self.dataset._genes_present is not None and
                    first_key in self.dataset._genes_present.index):
                level = 'gene'
            else:
                raise ValueError(
                    f"Could not auto-detect level for key {first_key!r}. "
                    "Please specify level='variant' or level='gene'.")

        # Build counts dictionary from dataset
        if level == 'variant':
            if self.dataset._variants_present is None:
                raise ValueError(
                    "Variant presence matrix not computed. "
                    "Call dataset.compute_variants_present() first.")
            variant_counts = (
                self.dataset.variants_present.sum(axis=1).astype(int))
            counts = {
                key: int(variant_counts.get(key, 0))
                for key in results.keys()}
            passenger_genes = set(
                filter_passenger_genes(self.dataset.mutation_db))
            results_for_plot = results
        else:  # level == 'gene'
            if self.dataset._genes_present_non_silent is None:
                raise ValueError(
                    "Gene presence matrix not computed. "
                    "Call dataset.compute_gene_presence_non_silent() first.")
            gene_presence = self.dataset.genes_present_non_silent
            gene_counts = gene_presence.sum(axis=1).astype(int)

            mapping_df = (
                self.dataset.mutation_db[['ensembl_gene_id', 'gene']]
                .dropna()
                .drop_duplicates())
            id_to_name = dict(
                zip(mapping_df['ensembl_gene_id'],
                    mapping_df['gene']))
            name_to_id = dict(
                zip(mapping_df['gene'],
                    mapping_df['ensembl_gene_id']))
            passenger_gene_names = set(
                filter_passenger_genes(self.dataset.mutation_db))

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
                        "Ensembl IDs or gene symbols.")

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
            **kwargs)

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
                "dataset.load_dataset() first.")

        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first.")

        if self.dataset._variant_db is None:
            raise ValueError(
                "Variant database not loaded. "
                "Call generate_variant_db() or load_dataset() first.")

        if self._prob_g_tau_tau_independent is None:
            raise ValueError(
                "prob_g_tau_tau_independent flag not set. "
                "Call compute_base_mus() first.")

        # Choose which mutation rates to use
        if use_cov_effects and self._mu_gs is not None:
            mu_g_j = self._mu_gs
        else:
            mu_g_j = self.base_mus

        self.mu_ms = compute_mu_m_per_tumor(
            variants_df=self.dataset.variant_db,
            mu_g_j=mu_g_j,
            contexts_by_gene=self.dataset.contexts_by_gene,
            prob_g_tau_tau_independent=self.prob_g_tau_tau_independent,
            **kwargs)

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
        import json
        from pathlib import Path

        directory = Path(directory)
        manifest_path = directory / "model_manifest.json"

        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                f"Model directory {directory} already exists. "
                "Pass overwrite=True to replace it.")

        directory.mkdir(parents=True, exist_ok=True)

        def _save_dataframe(df, filename):
            path = directory / filename
            df.to_parquet(path)
            return filename

        def _save_dict_of_dataframes(data, folder):
            folder_path = directory / folder
            folder_path.mkdir(exist_ok=True)
            stored = {}
            for key, df in data.items():
                stored[key] = _save_dataframe(df, f"{folder}/{key}.parquet")
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
                    value, (bool, int, float, str)):
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
                return {_json_safe(k): _json_safe(v)
                        for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_json_safe(v) for v in value]
            if isinstance(value, set):
                return [_json_safe(v) for v in value]

            # Try common conversion methods
            if hasattr(value, "tolist"):
                try:
                    return value.tolist()
                except Exception:
                    pass
            if hasattr(value, "to_dict"):
                try:
                    return _json_safe(value.to_dict())
                except Exception:
                    pass
            if hasattr(value, "item"):
                try:
                    return value.item()
                except Exception:
                    pass

            # Last resort: convert to string
            return str(value)

        files = {}

        if self.cov_matrix is not None:
            files['cov_matrix'] = _save_dataframe(
                self.cov_matrix, "cov_matrix.parquet")

        if self._mu_taus is not None:
            if isinstance(self._mu_taus, dict):
                files['mu_taus'] = _save_dict_of_dataframes(
                    self._mu_taus,
                    "mu_taus")
            else:
                files['mu_taus'] = _save_dataframe(
                    self._mu_taus, "mu_taus.parquet")

        if self._base_mus is not None:
            if isinstance(self._base_mus, dict):
                files['base_mus'] = _save_dict_of_dataframes(
                    self._base_mus,
                    "base_mus")
            else:
                files['base_mus'] = _save_dataframe(
                    self._base_mus, "base_mus.parquet")

        if self.cov_effects is not None:
            files['cov_effects'] = _save_array(
                self.cov_effects,
                "cov_effects.npy")

        if self.cov_effects_posteriors is not None:
            posterior = self.cov_effects_posteriors
            if hasattr(posterior, "to_netcdf"):
                filename = directory / "cov_effects_posteriors.nc"
                posterior.to_netcdf(filename)
                files['cov_effects_posteriors'] = (
                    "cov_effects_posteriors.nc")

        if self._mu_gs is not None:
            files['mu_gs'] = _save_dataframe(
                self._mu_gs, "mu_gs.parquet")

        if self.mu_ms is not None:
            files['mu_ms'] = _save_dataframe(
                self.mu_ms, "mu_ms.parquet")

        if self.gammas:
            gammas_dir = directory / "gammas"
            gammas_dir.mkdir(exist_ok=True)
            gamma_files = {}

            for key, result in self.gammas.items():
                # Create safe filename by replacing problematic chars
                safe_key = (str(key)
                           .replace(" ", "_")
                           .replace("/", "_")
                           .replace("\\", "_")
                           .replace(":", "_"))
                filename = f"gamma_{safe_key}.nc"
                filepath = gammas_dir / filename

                # Save as NetCDF if possible
                if hasattr(result, "to_netcdf"):
                    result.to_netcdf(filepath)
                    gamma_files[key] = f"gammas/{filename}"
                else:
                    logger.warning(
                        f"Gamma result for {key!r} does not have "
                        f"to_netcdf method. Skipping save.")

            files['gamma_files'] = gamma_files

        dataset_snapshot = getattr(
            self.dataset, "dataset_directory", None)
        if dataset_snapshot is not None:
            dataset_snapshot = str(Path(dataset_snapshot).resolve())
        else:
            logger.warning(
                "Dataset does not have an associated saved directory. "
                "Model snapshots will be unable to reload the dataset. "
                "Call MutationDataset.save_dataset() and reload it "
                "before saving models.")

        manifest = {
            "version": 1,
            "dataset_snapshot": dataset_snapshot,
            "dataset_location": getattr(
                self.dataset, "location_maf_files", None),
            "covariate_names": self.covariate_names,
            "mu_taus_separate": isinstance(self._mu_taus, dict),
            "prob_g_tau_tau_independent": (
                self._prob_g_tau_tau_independent),
            "files": files,
        }

        manifest_path.write_text(
            json.dumps(_json_safe(manifest), indent=2))
        self._saved_location = str(directory.resolve())

    @classmethod
    def load_model(cls, directory):
        """Load a Model from a directory created by save_model()."""
        import json
        from pathlib import Path

        directory = Path(directory)
        manifest_path = directory / "model_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Model manifest not found at {manifest_path}.")

        manifest = json.loads(manifest_path.read_text())

        dataset_snapshot = manifest.get("dataset_snapshot")
        if not dataset_snapshot:
            raise ValueError(
                "Model manifest does not include a dataset_snapshot "
                "entry. Recreate the model by loading the appropriate "
                "MutationDataset and re-saving the model snapshot.")

        snapshot_path = Path(dataset_snapshot)
        if not snapshot_path.is_absolute():
            candidate = (directory / snapshot_path).resolve()
            if candidate.exists():
                snapshot_path = candidate

        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Dataset snapshot not found at {snapshot_path}. "
                "Ensure the dataset directory still exists or "
                "recreate it with MutationDataset.save_dataset().")

        model = cls(
            dataset=MutationDataset.load_dataset(snapshot_path),
            cov_matrix=None)

        files = manifest.get("files", {})

        def _load_dataframe(filename):
            return pd.read_parquet(directory / filename)

        if 'cov_matrix' in files:
            model.cov_matrix = _load_dataframe(files['cov_matrix'])

        if 'mu_taus' in files:
            mu_info = files['mu_taus']
            if isinstance(mu_info, dict):
                model._mu_taus = {
                    sig: _load_dataframe(path)
                    for sig, path in mu_info.items()}
            else:
                model._mu_taus = _load_dataframe(mu_info)

        if 'base_mus' in files:
            base_info = files['base_mus']
            if isinstance(base_info, dict):
                model._base_mus = {
                    sig: _load_dataframe(path)
                    for sig, path in base_info.items()}
            else:
                model._base_mus = _load_dataframe(base_info)

        if 'cov_effects' in files:
            model.cov_effects = np.load(
                directory / files['cov_effects'])

        if 'mu_gs' in files:
            model._mu_gs = _load_dataframe(files['mu_gs'])

        if 'mu_ms' in files:
            model.mu_ms = _load_dataframe(files['mu_ms'])

        # Load gamma results (new format: individual .nc files)
        if 'gamma_files' in files:
            import arviz as az
            model.gammas = {}
            for key, filepath in files['gamma_files'].items():
                full_path = directory / filepath
                if full_path.exists():
                    try:
                        model.gammas[key] = az.from_netcdf(full_path)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load gamma result for {key!r} "
                            f"from {filepath}: {e}")
                else:
                    logger.warning(
                        f"Gamma file not found: {filepath}")

        # Backward compatibility: load old JSON format
        elif 'gammas' in files:
            path = directory / files['gammas']
            model.gammas = json.loads(path.read_text())
        elif 'gamma_ms' in files or 'gamma_gs' in files:
            model.gammas = {}
            if 'gamma_ms' in files:
                path = directory / files['gamma_ms']
                model.gammas.update(json.loads(path.read_text()))
            if 'gamma_gs' in files:
                path = directory / files['gamma_gs']
                model.gammas.update(json.loads(path.read_text()))

        model._prob_g_tau_tau_independent = manifest.get(
            "prob_g_tau_tau_independent")
        model._saved_location = str(directory.resolve())

        return model

    @property
    def saved_location(self):
        """Directory where the model snapshot is stored."""
        if self._saved_location is None:
            raise ValueError(
                "Model has not been saved yet. "
                "Call save_model() or load_model() first.")
        return self._saved_location

    @property
    def mu_gs(self):
        """Per-gene, per-sample mutation rates (lazy loaded)."""
        if self._mu_gs is None:
            raise ValueError(
                "Mutation rates not computed. "
                "Call compute_mu_gs() first.")
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
                "load_dataset() to ensure gene indices match.")
            self._mu_gs = value
            return

        # Validate index for DataFrames
        if isinstance(value, pd.DataFrame):
            if not value.index.equals(
                    self.dataset.contexts_by_gene.index):
                logger.warning(
                    "mu_gs index does not match "
                    "dataset.contexts_by_gene.index. "
                    "This may cause errors in downstream analysis. "
                    "To fix: reindex mu_gs to match "
                    "contexts_by_gene.index, or recompute mu_gs "
                    "after loading contexts_by_gene.")

            # Also check against cov_matrix if present
            if self.cov_matrix is not None:
                if not value.index.equals(self.cov_matrix.index):
                    logger.warning(
                        "mu_gs index does not match cov_matrix.index. "
                        "This may cause errors in downstream analysis. "
                        "To fix: call model.assign_cov_matrix() again "
                        "to reindex cov_matrix to match "
                        "contexts_by_gene.index, then recompute mu_gs.")

        # Validate index for dicts of DataFrames (signature-separated)
        elif isinstance(value, dict):
            for sig_name, df in value.items():
                if isinstance(df, pd.DataFrame):
                    if not df.index.equals(
                            self.dataset.contexts_by_gene.index):
                        logger.warning(
                            "mu_gs['%s'] index does not match "
                            "dataset.contexts_by_gene.index. "
                            "This may cause errors in downstream "
                            "analysis. To fix: reindex all mu_gs "
                            "DataFrames to match contexts_by_gene.index, "
                            "or recompute mu_gs after loading "
                            "contexts_by_gene.",
                            sig_name)
                        break  # Only warn once for contexts_by_gene

                    # Also check against cov_matrix if present
                    if self.cov_matrix is not None:
                        if not df.index.equals(self.cov_matrix.index):
                            logger.warning(
                                "mu_gs['%s'] index does not "
                                "match cov_matrix.index. "
                                "This may cause errors in downstream "
                                "analysis. To fix: call "
                                "model.assign_cov_matrix() again to "
                                "reindex cov_matrix to match "
                                "contexts_by_gene.index, then recompute "
                                "mu_gs.",
                                sig_name)
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
                else None),
            cov_effects_kwargs=copy_module.deepcopy(
                self.cov_effects_kwargs))

        # Share large base results (memory efficient)
        new_model._base_mus = self._base_mus  # Share, don't copy
        new_model._mu_taus = self._mu_taus    # Share, don't copy
        new_model._prob_g_tau_tau_independent = (
            self._prob_g_tau_tau_independent)

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
                "Call compute_mu_taus() first.")
        return self._mu_taus

    @mu_taus.setter
    def mu_taus(self, value):
        """Set mutation burdens per tumor."""
        self._mu_taus = value

    def compute_mu_taus(
            self,
            separate_per_sigma=False,
            **kwargs):
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
            - L_low : float, optional (default 64)
                Lower burden threshold for correcting low-burden
                samples
            - L_high : float, optional (default 500)
                Upper burden threshold for intermediate-burden
                correction
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
        from .estimate_mus import compute_mu_tau_per_tumor
        from pathlib import Path
        import tempfile

        # Ensure mutation_db is loaded
        if self.dataset._mutation_db is None:
            raise ValueError(
                "Mutation database not loaded in dataset. "
                "Call dataset.generate_mutation_db() or "
                "dataset.load_dataset() first.")

        # Ensure signature decomposition has been run
        if self.dataset._sig_assignments is None:
            raise ValueError(
                "Signature decomposition not run. "
                "Call dataset.run_signature_decomposition() first.")

        if self.dataset._signature_matrix is None:
            raise ValueError(
                "Signature matrix not loaded. "
                "Call dataset.run_signature_decomposition() first.")

        # Write signature matrix to temporary file for compute_mu_tau_per_tumor
        with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False) as tmp_file:
            self.dataset.signature_matrix.to_csv(
                tmp_file.name, sep='\t')
            tmp_path = tmp_file.name

        try:
            l_low = kwargs.get("L_low", 64)
            l_high = kwargs.get("L_high", 500)

            if "L_low" not in kwargs:
                logger.warning(
                    "L_low was not provided and 64 was chosen as the "
                    "lower burden threshold for correcting low-burden "
                    "samples. If you want to run a model without "
                    "correction for low-burden samples set L_low=0.")
            if "L_high" not in kwargs:
                logger.warning(
                    "L_high was not provided and 500 was chosen as the "
                    "upper burden threshold for intermediate-burden "
                    "correction. If you want to run a model without a "
                    "limit for intermediate-burden correction set "
                    "L_high=np.inf.")

            compute_kwargs = kwargs.copy()
            compute_kwargs["L_low"] = l_low
            compute_kwargs["L_high"] = l_high

            # Compute mutation burdens
            self._mu_taus = compute_mu_tau_per_tumor(
                db=self.dataset.mutation_db,
                location_signature_matrix=tmp_path,
                assignments=self.dataset.sig_assignments,
                separate_per_sigma=separate_per_sigma,
                **compute_kwargs)
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
                "Call compute_mu_taus() first.")

        # Ensure contexts_by_gene are loaded
        if self.dataset._contexts_by_gene is None:
            raise ValueError(
                "Trinucleotide contexts by gene not loaded in dataset. "
                "Call dataset.generate_contexts_by_gene() or "
                "load_dataset() first.")

        # Compute baseline mutation rates per gene
        self._base_mus = compute_mu_g_per_tumor(
            mu_taus=self._mu_taus,
            contexts_by_gene=self.dataset.contexts_by_gene,
            prob_g_tau_tau_independent=prob_g_tau_tau_independent)

        self._prob_g_tau_tau_independent = prob_g_tau_tau_independent
        return self._base_mus

    def compute_mu_gs(
            self,
            assign_base_mus_to_rest=True,
            **kwargs):
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
                "dataset.load_dataset() first.")

        # Ensure base_mus have been computed
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first.")

        # If model has covariates but effects not estimated yet
        if self.cov_matrix is not None and self.cov_effects is None:
            raise ValueError(
                "Model has a covariate matrix but covariate effects "
                "have not been estimated yet. "
                "Call estimate_cov_effects() first to estimate how "
                "covariates affect mutation rates.")

        # If using covariate effects, ensure cov_matrix is provided
        if self.cov_effects is not None and self.cov_matrix is None:
            raise ValueError(
                "cov_matrix must be provided when using covariate "
                "effects. This should not happen if the Model was "
                "created properly.")

        # Compute per-gene, per-sample mutation rates
        result = compute_mus_per_gene_per_sample(
            db=self.dataset.mutation_db,
            base_mus=self.base_mus,
            cov_effect=self.cov_effects,
            cov_matrix=self.cov_matrix,
            **kwargs)

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
            self, sample="MAP", chains=4, burn=1000, tol=0.05):
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
        from .estimate_covariates_effect import (
            estimate_covariates_effect)
        from .estimate_presence import filter_passenger_genes_ensembl
        from .constants import random_seed

        # Step 1: Configuration - Extract bounds from kwargs
        cov_effects_kwargs = dict(self.cov_effects_kwargs)
        signature = inspect.signature(estimate_covariates_effect)
        default_lower_bound = signature.parameters[
            'lower_bounds_c'].default
        if default_lower_bound is inspect._empty:
            default_lower_bound = None
        default_upper_bound = signature.parameters[
            'upper_bounds_c'].default
        if default_upper_bound is inspect._empty:
            default_upper_bound = None
        lower_bounds_value = cov_effects_kwargs.get(
            'lower_bounds_c', default_lower_bound)
        upper_bounds_value = cov_effects_kwargs.get(
            'upper_bounds_c', default_upper_bound)

        # Step 1: Validation - Ensure prerequisites are available
        if self._base_mus is None:
            raise ValueError(
                "Baseline mutation rates (base_mus) not computed. "
                "Call compute_base_mus() first.")

        if self.dataset._genes_present is None:
            raise ValueError(
                "Gene presence matrix not computed in dataset. "
                "Call dataset.compute_gene_presence() first.")

        if self.cov_matrix is None:
            raise ValueError(
                "Covariate matrix is None. Cannot estimate covariate "
                "effects without covariates. Create model with "
                "cov_matrix or use assign_cov_matrix().")

        # Step 1: Determine draws and sampling mode
        if isinstance(sample, int) or (
                isinstance(sample, str) and
                sample.lower() == "full"):
            draws = 4000
            is_mcmc = True
        elif isinstance(sample, str) and sample.lower() == "map":
            draws = 1
            is_mcmc = False
        else:
            raise ValueError(
                f"sample must be 'MAP', 'full', or an integer, "
                f"got {sample}")

        # Step 2: Gene Filtering - Identify passenger genes
        passenger_gene_ids = filter_passenger_genes_ensembl(
            self.cov_matrix.index)

        # Step 2: Filter to genes with complete covariate data (no NaN)
        passenger_cov = self.cov_matrix.loc[passenger_gene_ids]
        complete_mask = ~passenger_cov.isna().any(axis=1)
        passenger_genes_complete = passenger_gene_ids[complete_mask]

        # Step 2: Optionally subsample genes if integer provided
        if draws > 1 and isinstance(sample, int):
            logger.info(
                f"Subsampling {sample} genes from "
                f"{len(passenger_genes_complete)} passenger genes "
                f"with complete covariates")
            passenger_genes_complete = pd.Index(
                passenger_genes_complete.to_series().sample(
                    sample, random_state=random_seed))

        # Step 2: Store gene count for later access
        self._n_in_cov_effects_estimation = len(
            passenger_genes_complete)

        # Step 3: Data Preparation - Detect signature-dependent mode
        is_signature_dependent = isinstance(self._base_mus, dict)

        # Step 3: Log estimation details
        if is_mcmc:
            logger.info(
                f"Estimating covariate effects posteriors for "
                f"{self._n_in_cov_effects_estimation} passenger genes "
                f"with {self.cov_matrix.shape[1]} covariate(s)")
            logger.info(
                f"MCMC parameters: {draws} draws, {chains} chains, "
                f"{burn} tuning steps")
        else:
            logger.info(
                f"Estimating covariate effects for "
                f"{self._n_in_cov_effects_estimation} passenger genes "
                f"with {self.cov_matrix.shape[1]} covariate(s)")

        if is_signature_dependent:
            n_sigs = len(self._base_mus)
            logger.info(
                f"Using signature-dependent mode "
                f"({n_sigs} signatures)")
        else:
            logger.info("Using signature-independent mode")

        # Step 3: Filter and transpose base_mus
        if isinstance(self._base_mus, dict):
            # Signature-separated: filter and transpose each DataFrame
            mus_transposed = {
                sig: df.loc[passenger_genes_complete].T.values
                for sig, df in self._base_mus.items()}
        else:
            # Signature-independent: filter and transpose DataFrame
            mus_transposed = (
                self._base_mus.loc[
                    passenger_genes_complete].T.values)

        # Step 3: Filter genes_present matrix
        presence_matrix = (
            self.dataset.genes_present.loc[
                passenger_genes_complete].T.values)

        # Step 3: Filter cov_matrix and convert to array
        cov_matrix_array = (
            self.cov_matrix.loc[passenger_genes_complete].values)

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
                **cov_effects_kwargs)

            # Store full posterior
            self.cov_effects_posteriors = result
            logger.info("MCMC sampling completed")

            # Extract posterior mean for use in subsequent
            # calculations
            import arviz as az
            posterior_mean = az.extract(
                result, var_names=['c']).mean(dim='sample').values
            self.cov_effects = posterior_mean
            logger.info(
                "Extracted posterior mean for subsequent "
                "calculations")

            lower_bounds_arr, upper_bounds_arr = (
                self._resolve_covariate_bounds(
                    self.cov_effects.shape,
                    lower_bounds_value,
                    upper_bounds_value))

            summary = az.summary(result, var_names=['c'])
            logger.info(
                "Posterior summary:\n%s",
                summary.to_string())

            if {'hdi_3%', 'hdi_97%'} <= set(summary.columns):
                hdi_lower = summary['hdi_3%'].to_numpy()
                hdi_upper = summary['hdi_97%'].to_numpy()
                self._warn_if_near_bounds(
                    lower_candidate=hdi_lower,
                    upper_candidate=hdi_upper,
                    lower_bounds=lower_bounds_arr,
                    upper_bounds=upper_bounds_arr,
                    tol=tol,
                    mode_desc="Posterior HDI")
            else:
                logger.warning(
                    "Posterior summary missing HDI columns; "
                    "skipping bounds proximity check for posterior.")
        else:
            logger.info("Running MAP estimation...")
            result = estimate_covariates_effect(
                mus=mus_transposed,
                presence_matrix=presence_matrix,
                cov_matrix=cov_matrix_array,
                draws=1,
                **cov_effects_kwargs)

            # Extract MAP estimate from result dict
            self.cov_effects = result['c']
            logger.info("MAP estimation completed")

            lower_bounds_arr, upper_bounds_arr = (
                self._resolve_covariate_bounds(
                    self.cov_effects.shape,
                    lower_bounds_value,
                    upper_bounds_value))
            self._warn_if_near_bounds(
                lower_candidate=self.cov_effects,
                upper_candidate=self.cov_effects,
                lower_bounds=lower_bounds_arr,
                upper_bounds=upper_bounds_arr,
                tol=tol,
                mode_desc="MAP estimates")

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

    def _resolve_covariate_bounds(
            self,
            coeffs_shape,
            lower_value,
            upper_value):
        """Broadcast configured bounds to match coefficient shape."""
        upper_array = None
        if upper_value is not None:
            upper_array = self._broadcast_bounds_value(
                upper_value, coeffs_shape)
        else:
            logger.warning(
                "upper_bounds_c was None; skipping bounds proximity "
                "checks.")
            return None, None

        if lower_value is None:
            lower_array = (
                -upper_array if upper_array is not None else None)
        else:
            lower_array = self._broadcast_bounds_value(
                lower_value, coeffs_shape)

        if lower_array is None or upper_array is None:
            logger.warning(
                "Unable to broadcast coefficient bounds to shape %s; "
                "skipping boundary proximity checks.",
                coeffs_shape)
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
                arr, shape)
            return None

    def _warn_if_near_bounds(
            self,
            lower_candidate,
            upper_candidate,
            lower_bounds,
            upper_bounds,
            tol,
            mode_desc):
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
                lower_candidate.size)
            return
        if upper_candidate.size != expected_size:
            logger.warning(
                "Expected %d upper-side values for bounds check but "
                "got %d; skipping.",
                expected_size,
                upper_candidate.size)
            return

        lower_candidate = lower_candidate.reshape(lower_bounds.shape)
        upper_candidate = upper_candidate.reshape(upper_bounds.shape)

        lower_mask = (
            np.isfinite(lower_candidate) &
            np.isfinite(lower_bounds) &
            ((lower_candidate - lower_bounds) <= tol))
        upper_mask = (
            np.isfinite(upper_candidate) &
            np.isfinite(upper_bounds) &
            ((upper_bounds - upper_candidate) <= tol))

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
                "adjusted lower_bounds_c/upper_bounds_c.")
            log_fn = getattr(logger, "warming", logger.warning)
            log_fn(warn_msg)

    def _coefficient_labels(self, shape):
        """Return human-readable labels for coefficients."""
        covariate_labels = ['intercept'] + list(self.covariate_names)

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

    def estimate_passenger_genes_r2(self):
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
        from .estimate_presence import filter_passenger_genes_ensembl
        from sklearn.metrics import r2_score
        import logging

        # Check if mu_gs need to be computed
        if self._mu_gs is None:
            # Check if this is a model without covariates
            if self.cov_matrix is None:
                logging.info(
                    "Model has no covariate matrix. Computing mu_gs "
                    "with baseline mutation rates only (no covariate "
                    "effects).")

            # Ensure base_mus are available
            if self._base_mus is None:
                raise ValueError(
                    "Baseline mutation rates (base_mus) not computed. "
                    "Call compute_base_mus() first.")

            # Compute mu_gs
            self.compute_mu_gs()

        # Ensure genes_present has been computed
        if self.dataset._genes_present is None:
            raise ValueError(
                "Gene presence matrix not computed in dataset. "
                "Call dataset.compute_gene_presence() first.")

        # Identify passenger genes
        passenger_gene_ids = filter_passenger_genes_ensembl(
            self._mu_gs.index)

        # Restrict to passenger genes
        mu_gs_passenger = self._mu_gs.loc[passenger_gene_ids]
        genes_present_passenger = (
            self.dataset.genes_present.loc[passenger_gene_ids])

        # Sum observed mutations across all samples for each gene
        present_sum = genes_present_passenger.sum(
            axis=1)  # Sum over genes

        # Convert mutation rates to presence probabilities and sum
        # across all samples for each gene
        expected = (1-np.exp(-mu_gs_passenger)).sum(axis=1)

        # Compute R² between expected and observed
        r2 = r2_score(present_sum, expected)

        # Store result
        self._passenger_genes_r2 = r2

        return r2

    def aggregate_signatures(
            self,
            signature_selection,
            include_other=False):
        """Aggregate signature-separated base_mus into chosen signatures.

        This method allows you to combine multiple related signatures
        (e.g., SBS10a, SBS10b, SBS10c) into aggregate signatures
        (e.g., SBS10), and optionally group all remaining signatures
        into an "other" category.

        The aggregated result replaces self._base_mus.

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
                "Call compute_base_mus() first.")

        # Ensure base_mus are signature-dependent
        if not isinstance(self._base_mus, dict):
            raise ValueError(
                "base_mus must be signature-dependent (dict) to "
                "aggregate signatures. Current base_mus is a "
                "DataFrame (signature-independent). "
                "To create signature-separated base_mus, use "
                "model.compute_mu_taus(separate_per_sigma=True) "
                "before compute_base_mus().")

        aggregated = {}
        matched_signatures = set()

        # Process each item in signature_selection
        for item in signature_selection:
            # Case 1: Tuple/list - explicit grouping
            if isinstance(item, (tuple, list)):
                group_name = '+'.join(item)
                group_sum = None
                for sig in item:
                    if sig in self._base_mus:
                        matched_signatures.add(sig)
                        if group_sum is None:
                            group_sum = self._base_mus[sig].copy()
                        else:
                            group_sum += self._base_mus[sig]
                if group_sum is not None:
                    aggregated[group_name] = group_sum

            # Case 2: String - exact match or prefix aggregation
            elif isinstance(item, str):
                # Try exact match first
                if item in self._base_mus:
                    aggregated[item] = self._base_mus[item].copy()
                    matched_signatures.add(item)
                else:
                    # Prefix aggregation
                    matching_sigs = [
                        sig for sig in self._base_mus.keys()
                        if sig.startswith(item)]

                    if matching_sigs:
                        agg_sum = None
                        for sig in matching_sigs:
                            matched_signatures.add(sig)
                            if agg_sum is None:
                                agg_sum = self._base_mus[sig].copy()
                            else:
                                agg_sum += self._base_mus[sig]
                        aggregated[item] = agg_sum

        # Add 'other' category if requested
        if include_other:
            other_sigs = [
                sig for sig in self._base_mus.keys()
                if sig not in matched_signatures]

            if other_sigs:
                other_sum = None
                for sig in other_sigs:
                    if other_sum is None:
                        other_sum = self._base_mus[sig].copy()
                    else:
                        other_sum += self._base_mus[sig]
                aggregated['other'] = other_sum

        # Replace base_mus with aggregated version
        self._base_mus = aggregated

        return self._base_mus
