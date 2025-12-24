"""Constants that we use in multiple modules."""

# Random seed, set to a value if you want to replicate the results,
# this seed would be used in the sampling
random_seed = None
# random_seed = 777


# These are the cuts that we use to define if the mutation burden in a
# sample is too low (likely due to bad genotyping) or too high, in
# which case the sample is considered a hypermutator
L_low = 64  # based on https://academic.oup.com/bib/article/25/4/bbae249/7680470
L_high = 500


# SBS signatures canonical order. These come directly from COSMIC
# https://cancer.sanger.ac.uk/signatures/downloads/
# Current release v3.4
sbs_signatures = ["SBS1", "SBS2", "SBS3", "SBS4", "SBS5", "SBS6",
                  "SBS7a", "SBS7b", "SBS7c", "SBS7d", "SBS8", "SBS9",
                  "SBS10a", "SBS10b", "SBS10c", "SBS10d", "SBS11",
                  "SBS12", "SBS13", "SBS14", "SBS15", "SBS16",
                  "SBS17a", "SBS17b", "SBS18", "SBS19", "SBS20",
                  "SBS21", "SBS22a", "SBS22b", "SBS23", "SBS24",
                  "SBS25", "SBS26", "SBS27", "SBS28", "SBS29",
                  "SBS30", "SBS31", "SBS32", "SBS33", "SBS34",
                  "SBS35", "SBS36", "SBS37", "SBS38", "SBS39",
                  "SBS40a", "SBS40b", "SBS40c", "SBS41", "SBS42",
                  "SBS43", "SBS44", "SBS45", "SBS46", "SBS47",
                  "SBS48", "SBS49", "SBS50", "SBS51", "SBS52",
                  "SBS53", "SBS54", "SBS55", "SBS56", "SBS57",
                  "SBS58", "SBS59", "SBS60", "SBS84", "SBS85",
                  "SBS86", "SBS87", "SBS88", "SBS89", "SBS90",
                  "SBS91", "SBS92", "SBS93", "SBS94", "SBS95",
                  "SBS96", "SBS97", "SBS98", "SBS99"]

# Chromosomes
chromosomes = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

# Nucleotides
nucleotides = ['A', 'C', 'G', 'T']


# Trinucleotide contexts canonical order. Order first by mutation,
# then by previous nucleotide and then by next nucleotide.
canonical_types_order = [f"{first}[{mid_from}>{mid_to}]{third}"
                         for mid_from in "CT"
                         for mid_to in "ACGT".replace(mid_from, "")
                         for first in "ACGT"
                         for third in "ACGT"]

# Following previous order but for contexts
canonical_contexts_order = [f"{first}{mid}{third}"
                            for mid in "CT"
                            for first in "ACGT"
                            for third in "ACGT"]


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGT", "TGCA")
    return seq.translate(complement)[::-1]


def extract_context(mutation_type):
    """Extract trinucleotide context from mutation type.

    Parameters
    ----------
    mutation_type : str
        Mutation type in COSMIC format, e.g., 'G[C>T]G'.

    Returns
    -------
    str
        Trinucleotide context, e.g., 'GCG'.

    Examples
    --------
    >>> extract_context('G[C>T]G')
    'GCG'
    >>> extract_context('A[T>C]A')
    'ATA'
    """
    return mutation_type[0] + mutation_type[2] + mutation_type[-1]


# TCGA Study Abbreviations from
# https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations
tcga_study_codes = {
    "LAML": "Acute Myeloid Leukemia",
    "ACC": "Adrenocortical carcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "LGG": "Brain Lower Grade Glioma",
    "BRCA": "Breast invasive carcinoma",
    "CESC": ("Cervical squamous cell carcinoma and endocervical "
             "adenocarcinoma"),
    "CHOL": "Cholangiocarcinoma",
    "LCML": "Chronic Myelogenous Leukemia",
    "COAD": "Colon adenocarcinoma",
    "CNTL": "Controls",
    "ESCA": "Esophageal carcinoma",
    "FPPP": "FFPE Pilot Phase II",
    "GBM": "Glioblastoma multiforme",
    "HNSC": "Head and Neck squamous cell carcinoma",
    "KICH": "Kidney Chromophobe",
    "KIRC": "Kidney renal clear cell carcinoma",
    "KIRP": "Kidney renal papillary cell carcinoma",
    "LIHC": "Liver hepatocellular carcinoma",
    "LUAD": "Lung adenocarcinoma",
    "LUSC": "Lung squamous cell carcinoma",
    "DLBC": ("Lymphoid Neoplasm Diffuse Large B-cell "
             "Lymphoma"),
    "MESO": "Mesothelioma",
    "MISC": "Miscellaneous",
    "OV": "Ovarian serous cystadenocarcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "PCPG": "Pheochromocytoma and Paraganglioma",
    "PRAD": "Prostate adenocarcinoma",
    "READ": "Rectum adenocarcinoma",
    "SARC": "Sarcoma",
    "SKCM": "Skin Cutaneous Melanoma",
    "STAD": "Stomach adenocarcinoma",
    "TGCT": "Testicular Germ Cell Tumors",
    "THYM": "Thymoma",
    "THCA": "Thyroid carcinoma",
    "UCS": "Uterine Carcinosarcoma",
    "UCEC": "Uterine Corpus Endometrial Carcinoma",
    "UVM": "Uveal Melanoma"}

# These are the columns of a TCGA MAF file, descriptions from:
# https://docs.gdc.cancer.gov/Data/File_Formats/MAF_Format/ and
# https://asia.ensembl.org/info/docs/tools/vep/index.html?
maf_column_descriptions = {
    "Hugo_Symbol": "Gene name (HGNC-approved)",
    "Entrez_Gene_Id": "NCBI Entrez gene ID",
    "Center": "Sequencing center or project contributor",
    "NCBI_Build": "Reference genome version (e.g., GRCh38)",
    "Chromosome": "Chromosome (e.g., chr1–chr22, chrX, chrY)",
    "Start_Position": "Genomic start coordinate of the mutation",
    "End_Position": "Genomic end coordinate of the mutation",
    "Strand": "Strand of the reference genome (+ or -)",
    "Variant_Classification": "Effect of the mutation on the protein",
    "Variant_Type": "Type of mutation (e.g., SNP, INS, DEL)",
    "Reference_Allele": "Reference allele at the position",
    "Tumor_Seq_Allele1": "First observed allele in tumor",
    "Tumor_Seq_Allele2": "Second observed allele in tumor",
    "dbSNP_RS": "dbSNP ID (e.g., rs123456)",
    "dbSNP_Val_Status": "Validation status from dbSNP",
    "Tumor_Sample_Barcode": "TCGA barcode of the tumor sample",
    "Matched_Norm_Sample_Barcode": "TCGA barcode of the matched normal sample",
    "Match_Norm_Seq_Allele1": "First observed allele in normal sample",
    "Match_Norm_Seq_Allele2": "Second observed allele in normal sample",
    "Tumor_Validation_Allele1": "Validated allele 1 in tumor",
    "Tumor_Validation_Allele2": "Validated allele 2 in tumor",
    "Match_Norm_Validation_Allele1": "Validated allele 1 in normal sample",
    "Match_Norm_Validation_Allele2": "Validated allele 2 in normal sample",
    "Verification_Status": "Verification status (e.g., Verified)",
    "Validation_Status": "Validation status (e.g., Validated)",
    "Mutation_Status": "Somatic or Germline",
    "Sequencing_Phase": "Phase of sequencing",
    "Sequence_Source": "Source of sequence (e.g., genomic, cDNA)",
    "Validation_Method": "Method used for validation",
    "Score": "Quality score (deprecated)",
    "BAM_File": "Associated BAM file name (optional)",
    "Sequencer": "Sequencing instrument",
    "Tumor_Sample_UUID": "UUID of the tumor sample",
    "Matched_Norm_Sample_UUID": "UUID of the matched normal sample",
    "HGVSc": "HGVS coding DNA notation",
    "HGVSp": "HGVS protein notation",
    "HGVSp_Short": "Simplified protein change (e.g., p.V600E)",
    "Transcript_ID": "Transcript ID (e.g., Ensembl)",
    "Exon_Number": "Exon number affected",
    "t_depth": "Tumor read depth",
    "t_ref_count": "Reads supporting reference allele in tumor",
    "t_alt_count": "Reads supporting variant allele in tumor",
    "n_depth": "Normal read depth",
    "n_ref_count": "Reads supporting reference allele in normal",
    "n_alt_count": "Reads supporting variant allele in normal",
    "all_effects": "All effects of the variant (deprecated)",
    "Allele": "Variant allele",
    "Gene": "Gene affected (as Ensembl stable gene identifier, e.g. ENSG00000139656)",
    "Feature": "Transcript or feature ID",
    "Feature_type": "Type of feature (e.g., Transcript)",
    "One_Consequence": "Most severe consequence of the variant",
    "Consequence": "All consequences of the variant",
    "cDNA_position": "Position in cDNA",
    "CDS_position": "Position in coding sequence",
    "Protein_position": "Position in protein",
    "Amino_acids": "Reference/variant amino acids",
    "Codons": "Reference/variant codons",
    "Existing_variation": "Known variant IDs (e.g., rsIDs)",
    "DISTANCE": "Distance from nearest gene (if intergenic)",
    "TRANSCRIPT_STRAND": "Strand of the transcript",
    "SYMBOL": "Gene symbol from VEP",
    "SYMBOL_SOURCE": "Source of the gene symbol",
    "HGNC_ID": "HGNC gene ID",
    "BIOTYPE": "Transcript biotype (e.g., protein_coding)",
    "CANONICAL": "Flag for canonical transcript",
    "CCDS": "Consensus CDS ID",
    "ENSP": "Ensembl protein ID",
    "SWISSPROT": "SwissProt ID",
    "TREMBL": "TrEMBL ID",
    "UNIPARC": "UniParc ID",
    "UNIPROT_ISOFORM": "UniProt isoform ID",
    "RefSeq": "RefSeq ID",
    "MANE": "Matched Annotation from NCBI and Ensembl",
    "APPRIS": "APPRIS annotation (principal isoform)",
    "FLAGS": "Special flags from VEP",
    "SIFT": "SIFT score and prediction",
    "PolyPhen": "PolyPhen score and prediction",
    "EXON": "Exon location (e.g., 3/8)",
    "INTRON": "Intron location (e.g., 2/7)",
    "DOMAINS": "Protein domains affected",
    "1000G_AF": "1000 Genomes global allele frequency",
    "1000G_AFR_AF": "AFR population frequency",
    "1000G_AMR_AF": "AMR population frequency",
    "1000G_EAS_AF": "EAS population frequency",
    "1000G_EUR_AF": "EUR population frequency",
    "1000G_SAS_AF": "SAS population frequency",
    "ESP_AA_AF": "ESP African American frequency",
    "ESP_EA_AF": "ESP European American frequency",
    "gnomAD_AF": "gnomAD global allele frequency",
    "gnomAD_AFR_AF": "gnomAD AFR population AF",
    "gnomAD_AMR_AF": "gnomAD AMR population AF",
    "gnomAD_ASJ_AF": "gnomAD ASJ population AF",
    "gnomAD_EAS_AF": "gnomAD EAS population AF",
    "gnomAD_FIN_AF": "gnomAD FIN population AF",
    "gnomAD_NFE_AF": "gnomAD NFE population AF",
    "gnomAD_OTH_AF": "gnomAD OTH population AF",
    "gnomAD_SAS_AF": "gnomAD SAS population AF",
    "MAX_AF": "Maximum allele frequency observed",
    "MAX_AF_POPS": "Population with max allele frequency",
    "gnomAD_non_cancer_AF": "gnomAD non-cancer global AF",
    "gnomAD_non_cancer_AFR_AF": "gnomAD non-cancer AFR AF",
    "gnomAD_non_cancer_AMI_AF": "gnomAD non-cancer AMI AF",
    "gnomAD_non_cancer_AMR_AF": "gnomAD non-cancer AMR AF",
    "gnomAD_non_cancer_ASJ_AF": "gnomAD non-cancer ASJ AF",
    "gnomAD_non_cancer_EAS_AF": "gnomAD non-cancer EAS AF",
    "gnomAD_non_cancer_FIN_AF": "gnomAD non-cancer FIN AF",
    "gnomAD_non_cancer_MID_AF": "gnomAD non-cancer MID AF",
    "gnomAD_non_cancer_NFE_AF": "gnomAD non-cancer NFE AF",
    "gnomAD_non_cancer_OTH_AF": "gnomAD non-cancer OTH AF",
    "gnomAD_non_cancer_SAS_AF": "gnomAD non-cancer SAS AF",
    "gnomAD_non_cancer_MAX_AF_adj": "Max adj. AF in non-cancer",
    "gnomAD_non_cancer_MAX_AF_POPS_adj": "Population with max adj. AF",
    "CLIN_SIG": "Clinical significance annotation",
    "SOMATIC": "Somatic status (1 = somatic)",
    "PUBMED": "PubMed IDs supporting the annotation",
    "TRANSCRIPTION_FACTORS": "Affected transcription factors",
    "MOTIF_NAME": "Name of regulatory motif affected",
    "MOTIF_POS": "Position in motif",
    "HIGH_INF_POS": "High information position flag",
    "MOTIF_SCORE_CHANGE": "Change in motif binding score",
    "miRNA": "Affected miRNA",
    "IMPACT": "Impact prediction from VEP",
    "PICK": "VEP pick flag (most relevant annotation)",
    "VARIANT_CLASS": "Type of variant (e.g., SNV, indel)",
    "TSL": "Transcript Support Level",
    "HGVS_OFFSET": "Offset for HGVS annotation",
    "PHENO": "Phenotype association flag",
    "GENE_PHENO": "Gene-phenotype association flag",
    "CONTEXT": "Nucleotide context of variant",
    "tumor_bam_uuid": "UUID for tumor BAM",
    "normal_bam_uuid": "UUID for normal BAM",
    "case_id": "GDC case identifier",
    "GDC_FILTER": "GDC-specific filtering info",
    "COSMIC": "COSMIC mutation ID(s)",
    "hotspot": "Mutation hotspot flag",
    "RNA_Support": "Support from RNA-seq data",
    "RNA_depth": "Total RNA-seq read depth",
    "RNA_ref_count": "RNA-seq reference reads",
    "RNA_alt_count": "RNA-seq variant reads",
    "callers": "Variant callers used (e.g., MuTect, VarScan)"
}
