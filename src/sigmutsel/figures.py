"""Plotting utilities for mutation analysis results.

This module provides publication-quality plotting functions for
visualizing selection coefficients and mutation rates. It includes
functions for creating posterior vs. observed count plots and
expected vs. observed mutation burden comparisons.
"""

import pandas as pd
import arviz as az

from .locations import location_cosmic_cancer_gene_census


oncogenes = pd.read_csv(location_cosmic_cancer_gene_census, sep="\t")
oncogenes = oncogenes[oncogenes["ROLE_IN_CANCER"] == "oncogene"]
oncogenes = sorted(oncogenes["GENE_SYMBOL"].unique())


def plot_posteriors_vs_counts(
    results: dict[str, "az.InferenceData"],
    counts: dict[str, int],
    passenger_genes: set[str],
    *,
    level: str = "gene",
    max_shift_x: int | None = None,
    min_shift_x: int | None = None,
    save: str | None = None,
    show: bool = False,
) -> None:
    """Plot posterior mean ± 94 % HDI against observed sample counts.

    The function produces a scatter/interval plot that compares the
    mean selection coefficient (`gamma`) from Bayesian posterior
    samples to the frequency with which each *gene* or *protein-level
    variant* occurs in a mutation database. Axis scaling, legend
    layout and marker styles are automatically adjusted depending on
    whether genes or variants are requested.

    Parameters
    ----------
    results : dict[str, arviz.InferenceData]
        Mapping ``label → InferenceData``.  Each object **must** contain a
        posterior variable named ``gamma`` with shape ``(draw, chain)`` or
        similar.
    counts : dict[str, int]
        Mapping ``label → count`` where each key exactly matches the
        entries in `results`. The count represents how many tumors have
        the corresponding gene or variant mutated.
    passenger_genes : set[str]
        Set of passenger gene names for marker styling.
    level : {'gene', 'variant'}, default 'gene'
        Selects whether plotting genes or variants. The plot labelling,
        axis limits, legend formatting and marker rules follow the choice.
    max_shift_x : int or None, optional
        Upper limit for the *x*-axis.  If *None* (default) the limit is set
        to slightly above the largest count observed.
    min_shift_x : int or None, optional
        Lower limit for the *x*-axis.  If *None* (default) the limit is set
        to slightly before the lowest count observed.
    save : str or None, optional
        File path (usually ``*.png``) to save the figure.  If *None*
        (default) the figure is only displayed and not written to disk.
    show : bool, default False
        Whether to display the figure after creation.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    Notes
    -----
    - Marker shapes / fill styles are determined by the global set
      `oncogenes` and the provided `passenger_genes` set.
    - Log-scales are used for the *y*-axis, with base 10 for both genes
      and variants.

    """
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns

    def _marker(label: str):
        gene = label if level == "gene" else label.split(" p.")[0]
        if gene in oncogenes:
            return "D", False  # filled diamond
        if gene in passenger_genes:
            return "o", True  # open circle
        return "o", False  # filled circle

    # ------------------------------------------------ collect data -------
    rows = []
    for lbl, idata in results.items():
        g = idata.posterior["gamma"].values.flatten()
        mean = g.mean()
        hdi_low, hdi_high = az.hdi(g, prob=0.94)
        rows.append(
            dict(
                label=lbl,
                count=counts.get(lbl, 0),
                mean=mean,
                low=hdi_low,
                high=hdi_high,
            )
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # -------------------------------------------------- plot -------------
    max_w = 307.28987 / 72.27
    max_h = 0.8 * 224.14662 / 72.27
    fig, ax = plt.subplots(figsize=(max_w, max_h / 0.8))

    palette = sns.color_palette("husl", len(df))
    handles = []

    for i, r in df.iterrows():
        mkr, hollow = _marker(r["label"])
        ax.vlines(
            r["count"], r["low"], r["high"], color=palette[i], lw=1
        )
        dot = ax.plot(
            r["count"],
            r["mean"],
            marker=mkr,
            markerfacecolor=("none" if hollow else palette[i]),
            color=palette[i],
            markersize=4,
            label=r["label"],
        )
        handles.append(dot[0])

    # axes ---------------------------------------------------------------
    xlabel = (
        "Number of tumours with gene mutated (excluding silent)"
        if level == "gene"
        else "Number of tumours with variant"
    )
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Selection coefficient", fontsize=8)

    if level == "gene":
        ymax = 130
        ax.set_yscale("log", base=10)
        ax.set_yticks([0.5, 1, 5, 10, 50, 100])
        ax.set_yticklabels([0.5, 1, 5, 10, 50, 100])
        ax.set_yticks(
            np.concatenate(
                [
                    np.arange(0.4, 1, 0.1),
                    np.arange(1, 10, 1),
                    np.arange(10, ymax + 1, 10),
                ]
            ),
            minor=True,
        )
        ax.set_ylim(0.4, ymax)
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:  # variant
        ax.set_yscale("log", base=10)
        # Ceiling follows the data (with headroom) rather than a
        # fixed 10**3: gamma's prior upper bound auto-expands when
        # bound-limited (see estimate_gammas.estimate_gamma_from_mus),
        # so posterior means/HDIs can legitimately land well above
        # 1000 and a fixed ceiling would silently clip them off-plot.
        ymax_data = max(df["high"].max() * 1.5, 10**3)
        ymax_pow = int(np.ceil(np.log10(ymax_data)))
        ax.set_yticks(np.power(10, np.arange(0, ymax_pow + 1)))
        ax.set_ylim(0.9, 10**ymax_pow)

    xmax = (
        max_shift_x
        if max_shift_x is not None
        else df["count"].max() * 1.02
    )
    xmin = (
        min_shift_x
        if min_shift_x is not None
        else df["count"].min() * 0.98
    )
    ax.set_xlim(xmin, xmax)
    major = 50 if level == "gene" else 10
    minor = 10 if level == "gene" else 5
    tick_start = int(np.ceil(xmin / major)) * major if major else xmin
    ax.set_xticks(np.arange(tick_start, xmax + major, major))
    ax.set_xticks(np.arange(xmin, xmax + minor, minor), minor=True)
    ax.set_xticklabels(
        np.arange(tick_start, xmax + major, major).astype(int),
        fontsize=6,
    )
    ax.axhline(1, color="gray", linestyle="--", lw=1)

    # legend -------------------------------------------------------------
    title = "Gene" if level == "gene" else "Variant"
    ncols = 2 if level == "gene" else 1
    bbox = (0.0, 1) if level == "gene" else (1.03, 1.02)
    ax.legend(
        handles=handles,
        title=title,
        ncols=ncols,
        prop={
            "style": "italic" if level == "gene" else "normal",
            "size": 6 if level == "gene" else 5.5,
        },
        loc="upper left",
        bbox_to_anchor=bbox,
    )

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    else:
        plt.close("all")
    return fig


def comparison_with_observed(
    expected, observed, fig_name_additional=None, show=True
):
    """Make an expected vs observed scatter plot for mutation counts.

    Plots each gene as a point (x = expected, y = observed), annotates
    R^2 and mean percentage error (MPE), and scales axes to cover the
    largest value present in either vector (with a small headroom).
    The 1:1 reference line is drawn across the full dynamic range so
    over/under-prediction is easy to see.

    Parameters
    ----------
    expected : pandas.Series
        Expected number of mutated samples per gene. Index must be
        gene IDs.
    observed : pandas.Series or pandas.DataFrame or array-like
        Observed counts. If Series, its index should be gene IDs.
        If DataFrame, rows are genes and columns are samples; numeric
        columns are summed across axis=1 to get per-gene counts.
        If array-like, it is aligned to `expected` by position.
    fig_name_additional : str or None, optional
        If provided, appended to the base filename when saving.

    Notes
    -----
    - R^2 is computed on finite, paired values only.
    - MPE is mean((obs - exp) / exp) * 100 on pairs with expected > 0.

    Saves
    -----
    ../presentations/images/observed_vs_expected_passenger{_suffix}

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.

    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator

    # --- Normalize observed to a per-gene Series ---
    if isinstance(observed, pd.DataFrame):
        obs_series = observed.select_dtypes(include=[np.number]).sum(
            axis=1
        )
    elif isinstance(observed, pd.Series):
        obs_series = observed
    else:
        # Align by position if an array is passed
        obs_array = np.asarray(observed, dtype=float)
        obs_series = pd.Series(obs_array, index=expected.index)

    # --- Align indices and coerce to arrays ---
    intersection = expected.index.intersection(obs_series.index)
    exp = np.asarray(expected.loc[intersection], dtype=float)
    obs = np.asarray(obs_series.loc[intersection], dtype=float)
    finite = np.isfinite(exp) & np.isfinite(obs)

    # --- Metrics ---
    r2 = (
        r2_score(obs[finite], exp[finite]) if finite.any() else np.nan
    )

    mpe_mask = finite & (exp > 0)
    if mpe_mask.any():
        mpe = np.mean((obs[mpe_mask] - exp[mpe_mask]) / exp[mpe_mask])
        mpe *= 100.0
    else:
        mpe = np.nan

    # --- Figure ---
    max_width_beamer = 307.28987 / 72.27
    max_height_beamer = 224.14662 / 72.27
    fig, ax = plt.subplots(
        figsize=(max_width_beamer, max_height_beamer)
    )

    ax.scatter(exp, obs, s=3, alpha=0.5)

    ax.set_xlabel("Expected mutated samples", fontsize=10)
    ax.set_ylabel("Observed mutated samples", fontsize=10)

    ax.text(
        0.95,
        0.05,
        (
            f"$R^2$: {r2:.3f}\n"
            f"MPE = {mpe:.2f}%\n"
            f"Passenger genes: {finite.sum():,}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(facecolor="none", edgecolor="none", alpha=1),
    )

    max_val = np.max([exp.max(), obs.max()])
    lim = int(round(max_val * 1.05))

    ax.plot([0, lim], [0, lim], "--", color="gray")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if fig_name_additional is not None:
        suffix = f"_{fig_name_additional}"
        fig_name = f"observed_vs_expected_passenger{suffix}"
        fig.savefig(f"../presentations/images/{fig_name}", dpi=300)
    if show:
        plt.show()
    else:
        plt.close("all")

    return r2


def plot_signature_correlations(
    db,
    assignments,
    location_sig_matrix_norm,
    L_low,
    L_high,
    cov_matrix_for_corr,
    top_n=None,
    figsize=None,
    mutations_log_scale=False,
    save_path=None,
    show=True,
):
    """Plot correlations with bars and mutation counts.

    Computes signature correlations with provided covariates and
    visualizes them as bars with mutation counts.

    Parameters
    ----------
    db : pd.DataFrame
        Mutation database with columns 'Tumor_Sample_Barcode',
        'type', and 'ensembl_gene_id'.
    assignments : pd.DataFrame
        Signature assignments per sample from
        signature_decomposition.
    location_sig_matrix_norm : str or Path
        Path to normalized signature matrix file.
    L_low : float
        Lower bound for mutation burden filter.
    L_high : float
        Upper bound for mutation burden filter.
    cov_matrix_for_corr : dict
        Dictionary mapping covariate names to pd.Series
        indexed by ensembl_gene_id. Example:
        {"mrt": mrt_series, "gtex": gtex_series, ...}
    top_n : int, optional
        If provided, only plot the top N signatures by
        mutation count. Default None (plot all).
    figsize : tuple, optional
        Figure size. Default .
    mutations_log_scale : bool, optional
        If True, use log scale for mutations y-axis.
        Default False.
    save_path : str or Path, optional
        Path to save the figure. If None, doesn't save.
    show : bool, optional
        Whether to display the figure. Default True.

    Returns
    -------
    tuple
        (fig, ax1, ax2) matplotlib figure and axes objects.
    """
    from .signature_attribution import assign_signatures_per_gene_id
    import matplotlib.pyplot as plt
    import numpy as np

    # Compute signature attribution per gene
    sigs_per_gene_id = assign_signatures_per_gene_id(
        db, assignments, location_sig_matrix_norm, L_low, L_high
    )

    # Get gene IDs that are present in database
    db_gene_ids = db["ensembl_gene_id"].unique()

    # Compute correlations with each covariate
    correlations = pd.DataFrame()
    for cov_name, cov_series in cov_matrix_for_corr.items():
        # Get shared gene IDs between signatures and covariate
        shared_genes = sigs_per_gene_id.index.intersection(
            cov_series.index
        ).intersection(db_gene_ids)

        # Compute correlations
        corr_values = (
            sigs_per_gene_id.reindex(shared_genes)
            .corrwith(cov_series.reindex(shared_genes))
            .dropna()
        )
        correlations[cov_name] = corr_values

    # Add mutation counts
    correlations["mutations"] = sigs_per_gene_id.sum(axis=0)

    # Sort by mutations descending
    correlations = correlations.sort_values(
        by="mutations", ascending=False
    )

    # Filter to top_n if specified
    if top_n is not None:
        correlations = correlations.head(top_n)

    # Separate correlation columns from mutations
    corr_cols = [
        col for col in correlations.columns if col != "mutations"
    ]
    corr_data = correlations[corr_cols]
    mutations = correlations["mutations"]

    # Set up figure and primary axis
    if figsize is None:
        max_width_beamer = 307.28987 / 72.27
        max_height_beamer = 224.14662 / 72.27
        figsize = (max_width_beamer, max_height_beamer)

    fig, ax1 = plt.subplots(figsize=figsize)

    # X positions for signatures (equal spacing)
    x_pos = np.arange(len(correlations))
    width = 0.8 / len(corr_cols)  # Bar width

    # Use Set1 colormap
    import matplotlib.cm as cm

    cmap = cm.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(corr_cols))]

    # Plot bars for each correlation column
    for i, col in enumerate(corr_cols):
        offset = (i - len(corr_cols) / 2 + 0.5) * width
        ax1.bar(
            x_pos + offset,
            corr_data[col],
            width,
            label=col,
            color=colors[i],
            alpha=0.8,
        )

    # Configure primary y-axis (correlations)
    ax1.set_xlabel("Signature", fontsize=9)
    ax1.set_ylabel("Correlation", fontsize=9)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(correlations.index, rotation=90, fontsize=7)
    ax1.tick_params(axis="y", labelsize=7)

    # Add minor ticks every 0.01 for correlation
    y_min, y_max = ax1.get_ylim()
    ax1.set_yticks(
        np.arange(
            np.ceil(y_min * 100) / 100,
            np.floor(y_max * 100) / 100 + 0.01,
            0.01,
        ),
        minor=True,
    )

    ax1.axhline(
        y=0, color="DimGray", linestyle="-", linewidth=0.5, alpha=0.5
    )
    ax1.legend(loc="upper right", fontsize=8, ncol=1)
    ax1.grid(axis="y", alpha=0.3)

    # Create secondary y-axis for mutations (horizontal lines)
    ax2 = ax1.twinx()

    # Draw horizontal lines spanning the bar width
    line_width = 0.8  # Total width for bars
    for i, (x, mut_count) in enumerate(zip(x_pos, mutations)):
        ax2.plot(
            [x - line_width / 2, x + line_width / 2],
            [mut_count, mut_count],
            color="DimGray",
            linewidth=1.5,
            alpha=0.7,
        )

    ax2.set_ylabel(
        "Mutations attributed to signature",
        fontsize=9,
        color="DimGray",
    )
    ax2.tick_params(axis="y", labelcolor="DimGray", labelsize=7)

    # Add a simple legend entry for mutations
    ax2.plot(
        [],
        [],
        color="DimGray",
        linewidth=1.5,
        alpha=0.7,
        label="Mutations",
    )[0]

    # Set legend text color to DimGray
    ax2_legend = ax2.legend(
        loc="upper left", fontsize=7, frameon=True
    )
    for text in ax2_legend.get_texts():
        text.set_color("DimGray")

    if mutations_log_scale:
        ax2.set_yscale("log")
        # For log scale, align minimum mutation count with
        # y=0 of correlation axis
        y_min, y_max = ax1.get_ylim()
        zero_frac = (0 - y_min) / (y_max - y_min)
        mut_min, mut_max = mutations.min(), mutations.max()
        log_ratio = np.log10(mut_max / mut_min)
        # Calculate bottom value such that mut_min appears
        # at zero_frac
        ax2_bottom = mut_min / (
            10 ** (log_ratio * zero_frac / (1 - zero_frac))
        )
        ax2.set_ylim(bottom=ax2_bottom, top=mut_max * 1.05)
    else:
        # Align mutations axis so 0 mutations aligns with
        # 0 correlation
        y_min, y_max = ax1.get_ylim()
        # Position of 0 on ax1 as fraction of axis height
        zero_frac = (0 - y_min) / (y_max - y_min)
        # Set ax2 limits so 0 aligns with 0 on ax1
        mut_max = mutations.max()
        ax2_top = mut_max * 1.1
        # Solve: (0 - bottom) / (top - bottom) = zero_frac
        # => bottom = -zero_frac * top / (1 - zero_frac)
        ax2_bottom = -zero_frac * ax2_top / (1 - zero_frac)
        ax2.set_ylim(bottom=ax2_bottom, top=ax2_top)

        # Format ticks as 0, 10K, 20K, etc. and remove
        # negative ticks
        ax2_ticks = ax2.get_yticks()
        ax2_ticks = [t for t in ax2_ticks if t >= 0]
        ax2.set_yticks(ax2_ticks)
        ax2_ticklabels = [
            f"{int(t/1000)}K" if t > 0 else "0" for t in ax2_ticks
        ]
        ax2.set_yticklabels(ax2_ticklabels)

        # Add minor ticks every 1K for mutations
        mut_min = max(0, ax2.get_ylim()[0])
        mut_max = ax2.get_ylim()[1]
        ax2.set_yticks(np.arange(0, mut_max + 1000, 1000), minor=True)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close("all")

    return fig, ax1, ax2


def plot_sig_decomposition_with_burden(
    df,
    assignments,
    L_low=None,
    L_high=None,
    corrected=False,
    save=None,
    show=False,
):
    """Heatmap of signature attribution probabilities ordered by burden.

    Left panel: horizontal log-scale bar chart of mutation burden per
    sample. Right panel: heatmap of per-sample signature attribution
    probabilities (alpha), with samples sorted by ascending burden.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Mutation database with a 'Tumor_Sample_Barcode' column, or a
        pre-computed burden Series indexed by sample ID.
    assignments : pd.DataFrame
        Raw signature assignment counts (rows: samples, cols: sigs).
    L_low : float or None
        Lower burden threshold. When provided together with L_high,
        adds purple dashed lines on both panels at L_low and L_high.
    L_high : float or None
        Upper burden threshold.
    corrected : bool, default False
        If True, apply the low-burden correction from
        :func:`compute_alphas.estimate_alphas`, which blends
        low-burden sample alphas toward the population mean alpha_bar
        computed from intermediate-burden samples.
    save : str or Path or None
        File path to save the figure (PNG at 600 dpi). Not saved if
        None.
    show : bool, default False
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    from .compute_mutation_burden import count_mutation_burden
    from .compute_alphas import estimate_alphas

    # Compute or extract burden
    if isinstance(df, pd.DataFrame):
        mb = count_mutation_burden(df)
        burden = mb["total_mutations"]
    else:
        burden = df.copy()
        burden.name = "total_mutations"
        mb = burden.to_frame()

    # Sort samples by ascending burden; restrict to shared index
    sorted_burden = burden.sort_values()
    common = sorted_burden.index.intersection(assignments.index)
    sorted_burden = sorted_burden.loc[common]

    # Compute alphas (raw or corrected)
    if corrected and L_low is not None:
        alphas = estimate_alphas(
            mb, assignments, L_low=L_low, L_high=L_high
        )
        alphas = alphas.loc[sorted_burden.index]
    else:
        denom = sorted_burden.replace(0.0, float("nan"))
        alphas = (
            assignments.loc[sorted_burden.index]
            .div(denom, axis=0)
            .fillna(0.0)
        )

    # Beamer slide dimensions
    max_w = 307.28987 / 72.27
    max_h = 0.8 * 224.14662 / 72.27

    fig, (ax_bar, ax_hm) = plt.subplots(
        1,
        2,
        figsize=(max_w, max_h),
        gridspec_kw={"width_ratios": [1, 10]},
        sharey=True,
    )

    # Left panel: horizontal log10-burden bars
    log_b = np.log10(sorted_burden.clip(lower=1).values)
    ax_bar.barh(
        np.arange(len(sorted_burden)),
        log_b,
        height=0.5,
        color="gray",
        edgecolor=None,
    )
    ax_bar.invert_xaxis()
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Burden", fontsize=7)

    if L_low is not None and L_high is not None:
        xtick_vals = np.log10([1, L_low, L_high, 10000])
        xtick_labels = ["0", f"{L_low}", f"{L_high}", "10K"]
        for val in [L_low, L_high]:
            ax_bar.axvline(
                np.log10(val),
                color="purple",
                linestyle="--",
                linewidth=0.5,
            )
    else:
        xtick_vals = np.log10([1, 10, 100, 1000, 10000])
        xtick_labels = ["0", "10", "100", "1K", "10K"]
    ax_bar.set_xticks(xtick_vals)
    ax_bar.set_xticklabels(xtick_labels, fontsize=5, rotation=90)

    # Right panel: alpha heatmap
    vmax = alphas.values.max()
    norm = mcolors.Normalize(vmin=0, vmax=vmax)

    hm = ax_hm.imshow(
        alphas.values,
        aspect="auto",
        interpolation="none",
        cmap="Reds",
        norm=norm,
    )

    # Label only the top-10 signatures by mean alpha
    top_sigs = (
        alphas.mean(axis=0).sort_values().tail(10).index.tolist()
    )
    all_pos = np.arange(len(alphas.columns))
    top_pos = [
        i for i, c in enumerate(alphas.columns) if c in top_sigs
    ]
    ax_hm.set_xticks(top_pos)
    ax_hm.set_xticklabels(
        [alphas.columns[i] for i in top_pos], rotation=90, fontsize=5
    )
    ax_hm.set_xticks(all_pos, minor=True)

    # Dashed horizontal lines at L_low / L_high row boundaries
    if L_low is not None and L_high is not None:
        idx = np.arange(len(sorted_burden))
        low_rows = idx[sorted_burden.values < L_low]
        if len(low_rows):
            ax_hm.axhline(
                low_rows.max() + 0.5,
                color="purple",
                linestyle="--",
                linewidth=0.5,
            )
        high_rows = idx[sorted_burden.values < L_high]
        if len(high_rows):
            ax_hm.axhline(
                high_rows.max() + 0.5,
                color="purple",
                linestyle="--",
                linewidth=0.5,
            )

    # Colorbar
    cbar = fig.colorbar(hm, ax=ax_hm, pad=0.02)
    ax_hm.text(
        1.0,
        -0.1,
        "Probability",
        transform=ax_hm.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    tick_pcts = range(0, 93, 25)
    cbar.set_ticks([p / 100 for p in tick_pcts])
    cbar.set_ticklabels([f"{p}%" for p in tick_pcts], fontsize=5)

    plt.tight_layout(pad=0.25)

    if save is not None:
        fig.savefig(str(save), dpi=600)

    if show:
        plt.show()
    else:
        plt.close("all")

    return fig
