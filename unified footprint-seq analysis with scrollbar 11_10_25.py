#!/usr/bin/env python3
import os
import sys
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import numpy as np
import pandas as pd
import matplotlib
# Use a non-interactive backend when no display is available (e.g., when run on a headless server)
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.optimize import least_squares

# =========================
# Appearance constants
# =========================
# Default color for coverage bar plots.
BAR_COLOR = "#4C78A8"

# =========================
# Scrollable container (vertical only)
# =========================
class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable container that behaves like a regular Frame.
    Place child widgets into self.body.
    """
    def __init__(self, parent, *, width=980, height=680, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width, height=height)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # A Frame placed inside the Canvas holds the actual content.
        self.body = ttk.Frame(self.canvas)
        self.body_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")

        # Keep scrollregion and internal frame width synchronized with canvas size.
        self.body.bind("<Configure>", self._on_body_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable mouse-wheel scrolling across platforms.
        self._bind_mousewheel(self.canvas)

    def _on_body_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.itemconfigure(self.body_id, width=self.canvas.winfo_width())

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.body_id, width=event.width)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel, add="+")     # Windows/macOS
        widget.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")  # Legacy X11
        widget.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel(self, event):
        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

# =========================
# Helpers for centered fig-level xlabel
# =========================
def add_axes_centered_xlabel(fig, axes, text, fontsize=6, pad=0.035):
    """
    Add a single x-axis label centered under the union of the provided axes
    (not the entire figure). Call after subplots_adjust().
    pad is a vertical offset in figure coordinates.
    """
    ax_list = np.atleast_1d(axes)
    xmin = min(ax.get_position().x0 for ax in ax_list)
    xmax = max(ax.get_position().x1 for ax in ax_list)
    ymin = min(ax.get_position().y0 for ax in ax_list)
    xmid = (xmin + xmax) / 2.0
    y = ymin - pad
    fig.text(xmid, y, text, ha="center", va="top", fontsize=fontsize, transform=fig.transFigure)

# =========================
# Filename helpers
# =========================
def short_tag(name, max_base_len=18):
    """
    Create a compact, mostly human-readable tag from a path by taking the base name,
    truncating it, and appending a short hash to keep it unique.
    """
    base = os.path.splitext(os.path.basename(name))[0]
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:6]
    base = base[:max_base_len]
    return f"{base}-{h}"

# =========================
# Parsing and math helpers
# =========================
def load_gff_sparse(path):
    """
    Load a tab-delimited coverage file where only non-zero positions are listed.
    Expected columns:
      - 4th column (index 3): 1-based genomic coordinate
      - 6th column (index 5): coverage value (may be signed; magnitude is used)
    Returns:
      positions: sorted numpy array of genomic positions with non-zero coverage
      coverages: numpy array of coverage values (abs-summed for duplicate positions)
      cov_dict:  dict mapping genomic position -> coverage
    """
    pos_to_cov = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            try:
                pos = int(parts[3])     # 1-based position
                cov = float(parts[5])   # coverage value
            except ValueError:
                continue
            cov = abs(cov)              # sum magnitudes across strands
            pos_to_cov[pos] = pos_to_cov.get(pos, 0.0) + cov
    positions = np.array(sorted(pos_to_cov.keys()), dtype=int)
    coverages = np.array([pos_to_cov[k] for k in positions], dtype=float)
    return positions, coverages, pos_to_cov

def sum_region(positions, coverages, start, end):
    """
    Sum coverage across inclusive region [start, end].
    positions must be sorted to allow binary searches for fast slicing.
    """
    if positions.size == 0:
        return 0.0
    if end < start:
        start, end = end, start
    left = np.searchsorted(positions, start, side="left")
    right = np.searchsorted(positions, end, side="right")  # inclusive end
    if right <= left:
        return 0.0
    return float(coverages[left:right].sum())

def read_centers_txt(path):
    """
    Read a text file containing one integer center coordinate per line.
    Ignores empty lines and comments (lines starting with '#').
    """
    centers = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                centers.append(int(s))
            except ValueError:
                continue
    return centers

def fractions_vs_control(control_vec, other_vecs):
    """
    Compute per-region fractions = (noncontrol / control) with clipping to [0,1].
    Where control <= 0, define fraction = 1 by convention.
    Returns:
      fractions: list of arrays (one per non-control dataset)
      fractional_occupancy: list of arrays (1 - fraction), same shape/order
    """
    control = np.asarray(control_vec, dtype=float)
    others = [np.asarray(x, dtype=float) for x in other_vecs]
    n = control.shape[0]
    fractions, frac_occ = [], []
    ctrl_pos = control > 0
    for arr in others:
        frac = np.ones(n, dtype=float)  # default where control <= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            tmp = np.zeros(n, dtype=float)
            tmp[ctrl_pos] = arr[ctrl_pos] / control[ctrl_pos]
        tmp = np.clip(tmp, 0.0, 1.0)
        frac[ctrl_pos] = tmp[ctrl_pos]
        fractions.append(frac)
        frac_occ.append(1.0 - frac)
    return fractions, frac_occ

def black_to_red_cmap():
    """Continuous colormap mapping 0→black and 1→red."""
    return LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)], N=256)

# =======================================================
# Kd fitting (Langmuir model; ymin=0, ymax=1; non-cooperative)
# =======================================================
def fit_kd_langmuir(conc_nM, y_fractional_occ):
    """
    Fit y = x / (Kd + x) with Kd > 0.
    Ensures the curve passes through (0,0) by anchoring y(0)=0.
    Returns:
      Kd_nM: estimated Kd in nM (float)
      success: boolean fit status
      yhat: fitted y values at the x used internally (including zero if added)
    """
    x = np.asarray(conc_nM, float)
    y = np.clip(np.asarray(y_fractional_occ, float), 0.0, 1.0)
    if x.size != y.size or x.size < 2:
        return np.nan, False, np.full_like(x, np.nan, dtype=float)

    # Ensure the origin is in the dataset for a stable anchor.
    zero_mask = np.isclose(x, 0.0)
    if np.any(zero_mask):
        y[zero_mask] = 0.0
    else:
        x = np.concatenate(([0.0], x))
        y = np.concatenate(([0.0], y))

    # Sort by concentration for monotonicity.
    idx = np.argsort(x)
    x = x[idx]; y = y[idx]

    # Initialize Kd using the first crossing near y=0.5 if available; otherwise geometric mean.
    pos = x[x > 0]
    if pos.size == 0:
        return np.nan, False, np.full_like(x, np.nan, dtype=float)
    half = 0.5
    xs, ys = x, y
    crossing = np.where((ys[:-1] - half) * (ys[1:] - half) <= 0)[0]
    if crossing.size:
        i = crossing[0]
        x1, x2 = xs[i], xs[i+1]; y1, y2 = ys[i], ys[i+1]
        kd0 = max(1e-12, x1 + (half - y1) * (x2 - x1) / max(1e-12, (y2 - y1)))
    else:
        kd0 = 10 ** np.mean(np.log10(pos))
    log10_kd0 = np.log10(max(1e-12, kd0))

    # Residuals in y-space; optimize log10(Kd) to keep positivity.
    def model(log10_kd, x_):
        kd = 10.0 ** log10_kd
        return x_ / (kd + x_)

    def resid(p):
        yhat = model(p[0], x)
        return (yhat - y)

    res = least_squares(
        resid, x0=np.array([log10_kd0], dtype=float),
        bounds=([-12.0], [12.0]),
        loss='soft_l1', f_scale=1.0, max_nfev=2000
    )
    log10_kd = res.x[0]
    kd = 10.0 ** log10_kd
    yhat = model(log10_kd, x)
    return float(kd), bool(res.success), yhat

# =======================================================
# Estimators for line graphs (coverage difference profiles)
# =======================================================
def trimmed_mean_1d(x, trim=0.10):
    """Mean after trimming equally from both tails by 'trim' proportion (0–0.5)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if trim <= 0:
        return float(np.mean(x))
    if trim >= 0.5:
        return float(np.median(x))
    lo = np.quantile(x, trim); hi = np.quantile(x, 1 - trim)
    keep = (x >= lo) & (x <= hi)
    if not np.any(keep):
        return float(np.mean([lo, hi]))
    return float(np.mean(x[keep]))

def winsorized_mean_1d(x, win=0.10):
    """Mean after winsorizing tails to the specified quantiles."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if win <= 0:
        return float(np.mean(x))
    if win >= 0.5:
        return float(np.median(x))
    lo = np.quantile(x, win); hi = np.quantile(x, 1 - win)
    return float(np.mean(np.clip(x, lo, hi)))

def huber_location_1d(x, k_mad=1.5, max_iter=50, tol=1e-6):
    """Robust location via Huber M-estimator using scale from MAD."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = np.median(np.abs(x - np.mean(x))) or 1.0
    delta = max(1e-12, k_mad * mad)
    mu = np.mean(x)
    for _ in range(max_iter):
        r = x - mu
        w = np.ones_like(r)
        mask = np.abs(r) > delta
        w[mask] = delta / np.maximum(np.abs(r[mask]), 1e-12)
        mu_new = np.sum(w * x) / np.sum(w)
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new
    return float(mu)

def tukey_biweight_location_1d(x, k_mad=4.685, max_iter=50, tol=1e-6):
    """Robust location via Tukey biweight with iterative reweighting."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0:
        mad = np.median(np.abs(x - np.mean(x))) or 1.0
    c = max(1e-12, k_mad * mad)
    mu = np.mean(x)
    for _ in range(max_iter):
        r = x - mu
        u = r / c
        w = (1 - u**2)**2
        w[np.abs(u) >= 1] = 0.0
        denom = np.sum(w)
        if denom <= 1e-12:
            break
        mu_new = np.sum(w * x) / denom
        if abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new
    return float(mu)

ESTIMATOR_FUNCS = {
    "mean":   lambda x, **kw: float(np.mean(x)) if x.size else np.nan,
    "median": lambda x, **kw: float(np.median(x)) if x.size else np.nan,
    "trimmed": lambda x, trim=0.10, **kw: trimmed_mean_1d(x, trim=trim),
    "winsor":  lambda x, win=0.10, **kw: winsorized_mean_1d(x, win=win),
    "huber":   lambda x, k_mad=1.5, **kw: huber_location_1d(x, k_mad=k_mad),
    "tukey":   lambda x, k_mad=4.685, **kw: tukey_biweight_location_1d(x, k_mad=k_mad),
}
ESTIMATOR_LABELS = {
    "mean":   "Mean coverage difference (control − non-control)",
    "median": "Median coverage difference (control − non-control)",
    "trimmed":"Trimmed-mean coverage difference (control − non-control)",
    "winsor": "Winsorized-mean coverage difference (control − non-control)",
    "huber":  "Huber M-estimate coverage difference (control − non-control)",
    "tukey":  "Tukey biweight coverage difference (control − non-control)",
}
ESTIMATOR_SUFFIX = {
    "mean": "mean",
    "median": "median",
    "trimmed": "trim",
    "winsor": "win",
    "huber": "huber",
    "tukey": "tukey",
}

# =======================================================
# Line graph helpers (±100 nt window around centers)
# =======================================================
def compute_diffs_matrix(control_dict, non_dict, centers, flank=100):
    """
    Build a matrix of coverage differences (control - non-control)
    across positions [center - flank, center + flank] for each center.
    Returns:
      offsets: array of relative offsets
      diffs:   (n_centers x len(offsets)) matrix of differences
    """
    offsets = np.arange(-flank, flank + 1, dtype=int)
    diffs = np.zeros((len(centers), offsets.size), dtype=float)
    for r, c in enumerate(centers):
        vals = []
        for off in offsets:
            pos = c + off
            ctrl = control_dict.get(pos, 0.0)
            nonv = non_dict.get(pos, 0.0)
            vals.append(ctrl - nonv)
        diffs[r, :] = vals
    return offsets, diffs

def profile_from_diffs(diffs, estimator_key, params):
    """
    Apply the chosen estimator to each column (offset) of the diffs matrix,
    aggregating across centers to produce a single profile over offsets.
    """
    func = ESTIMATOR_FUNCS[estimator_key]
    prof = np.zeros(diffs.shape[1], dtype=float)
    for j in range(diffs.shape[1]):
        col = diffs[:, j]
        if estimator_key == "trimmed":
            prof[j] = func(col, trim=params.get("trim", 0.10))
        elif estimator_key == "winsor":
            prof[j] = func(col, win=params.get("win", 0.10))
        elif estimator_key == "huber":
            prof[j] = func(col, k_mad=params.get("huber_k", 1.5))
        elif estimator_key == "tukey":
            prof[j] = func(col, k_mad=params.get("tukey_k", 4.685))
        else:
            prof[j] = func(col)
    return prof

# =======================================================
# Coverage bar page helpers (±flank bp around a center)
# =======================================================
def format_one_decimal(ax):
    """Format y-axis ticks to one decimal place."""
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.1f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins='auto', prune=None))

def compute_window_bars(cov_dict, center, flank=60):
    """
    Retrieve coverage values at integer positions in [center - flank, center + flank]
    and return relative offsets plus the corresponding heights.
    """
    offsets = np.arange(-flank, flank + 1, dtype=int)
    heights = np.array([cov_dict.get(center + off, 0.0) for off in offsets], dtype=float)
    return offsets, heights

def render_bars_page(
    pdf,
    center,
    gff_names,
    cov_dicts,
    concs_in_user_order,
    flank=60,
    page_width=3.9,        # compact width so tight bbox doesn't occupy entire PDF page
    height_per_row=0.72,   # per-dataset subplot height
    save_png=False,
    png_dpi=300,
    out_dir=None,
    ctag="bars",
    pdf_pad_inches=0.5,    # padding to prevent label clipping in vector output
    bar_color=BAR_COLOR,   # single color for bars
):
    """
    Render a PDF page with stacked bar plots (one per dataset) for the window
    [center - flank, center + flank]. Each subplot shows coverage vs relative offset.
    The y-axis label indicates the dataset's concentration.
    """
    n = len(cov_dicts)
    # Precompute data and a common y-limit per page to keep scales comparable.
    data = []
    y_max = 0.0
    for d in cov_dicts:
        offs, ys = compute_window_bars(d, center, flank=flank)
        data.append((offs, ys))
        if ys.size:
            y_max = max(y_max, float(np.max(ys)))
    if y_max <= 0:
        y_max = 1.0

    fig_height = max(1.3, height_per_row * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True,
                             figsize=(page_width, fig_height))
    if n == 1:
        axes = [axes]

    for idx, (ax, (name, (offs, ys))) in enumerate(zip(axes, zip(gff_names, data))):
        ax.bar(offs, ys, color=bar_color, width=1.0)
        ax.set_ylim(0, y_max * 1.05)
        format_one_decimal(ax)

        # Compact tick labels.
        ax.tick_params(axis='both', which='major', labelsize=5)
        ax.tick_params(axis='x', which='major', pad=1)

        # Annotate each subplot with its concentration.
        conc_val = float(concs_in_user_order[idx])
        ax.set_ylabel(f"{conc_val:g} nM protein",
                      rotation=0, ha='right', va='center',
                      labelpad=18, fontsize=5)

        # No per-subplot x-label; a single figure-level label is added below.
        ax.set_xlabel(None)

        # Reduce visual clutter.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for ax in axes:
        ax.set_xlim(-flank - 0.5, flank + 0.5)

    axes[0].set_title(f"Center {center} — Coverage (±{flank} bp)", fontsize=8)

    # Leave room for labels; add a figure-level x-label centered under the axes block.
    fig.subplots_adjust(left=0.40, right=0.98, top=0.86, bottom=0.20, hspace=0.22)
    add_axes_centered_xlabel(fig, axes, "Position relative to center (bp)", fontsize=6, pad=0.035)

    # Write this page to the multi-page PDF.
    pdf.savefig(fig, dpi=200, bbox_inches="tight", pad_inches=pdf_pad_inches)

    # Optional per-page PNG export.
    if save_png and out_dir:
        png_path = os.path.join(out_dir, f"{ctag}__center_{center}__bars.png")
        fig.savefig(png_path, dpi=int(max(50, png_dpi)), bbox_inches="tight")

    plt.close(fig)

# =========================
# Core processing
# =========================
def process_one_centers_file(
    centers_path,
    gff_paths_in_user_order,   # all datasets in user-chosen order
    concs_in_user_order,       # concentrations aligned to that order; exactly one must be 0 nM
    output_folder,
    selected_estimators,       # estimator keys for line graphs
    params,                    # parameters for estimators (trim/win/huber_k/tukey_k)
    y_limits=None,             # shared y-limits for line graphs; None => auto
    progress_hook=None,
    save_png_models=False,     # whether to save individual PNGs for Kd model plots
    heatmap_dpi=300,           # DPI for heatmap PNG
    bar_flank_bp=60,           # ±bp for coverage bar pages
    save_bar_png=False,        # whether to save each bar page as PNG
    bar_png_dpi=300,           # DPI for bar page PNGs
):
    """
    Process a single centers file:
      1) Sum coverage in 30-bp windows centered at each coordinate for all datasets.
      2) Identify the 0 nM control dataset.
      3) Compute per-region fractions vs control and fractional occupancy.
      4) Fit Kd per region using a Langmuir model with the non-control datasets.
      5) Save CSVs for coverage, fractions, fractional occupancy, and Kd estimates.
      6) Create a heatmap of fractional occupancy with a side Kd plot.
      7) Generate a multi-page PDF alternating coverage bar pages and Kd model pages.
      8) Produce line-graph profiles (control − non-control) using selected estimators.
    """
    def tick(frac, msg):
        if progress_hook:
            progress_hook(frac, msg)

    # Create output subfolder named after the centers file.
    centers_base = os.path.splitext(os.path.basename(centers_path))[0]
    out_dir = os.path.join(output_folder, centers_base)
    os.makedirs(out_dir, exist_ok=True)
    ctag = short_tag(centers_base)

    tick(0.02, "Reading centers…")
    centers = read_centers_txt(centers_path)
    if not centers:
        raise RuntimeError(f"No coordinates found in: {centers_path}")

    # Find the index of the control dataset (exactly one must be 0 nM).
    concs_arr = np.asarray(concs_in_user_order, dtype=float)
    zero_mask = np.isclose(concs_arr, 0.0)
    if np.sum(zero_mask) != 1:
        raise RuntimeError("Exactly one dataset must be at 0 nM (the control).")
    control_index = int(np.where(zero_mask)[0][0])

    # Load all coverage files into memory (sparse representation).
    tick(0.05, "Loading GFFs…")
    loaded = []
    for p in gff_paths_in_user_order:
        positions, coverages, cov_dict = load_gff_sparse(p)
        loaded.append((positions, coverages, cov_dict))

    # Sum coverage in 30-bp windows [center-15, center+14] for each dataset and center.
    tick(0.15, "Summing 30-bp windows…")
    per_gff_coverages = []
    total_tasks = len(loaded) * len(centers)
    done = 0
    for gi, (pos, cov, _) in enumerate(loaded):
        sums = np.zeros(len(centers), dtype=float)
        for idx, c in enumerate(centers):
            start = c - 15
            end = c + 14
            sums[idx] = sum_region(pos, cov, start, end)
            done += 1
            if done % 500 == 0:
                tick(0.15 + 0.45 * (done / max(1, total_tasks)), "Summing 30-bp windows…")
        per_gff_coverages.append(sums)

    # Save per-region coverage table for all datasets.
    gff_names = [os.path.basename(p) for p in gff_paths_in_user_order]
    region_labels = [f"{c-15}-{c+14}" for c in centers]
    coverage_df = pd.DataFrame({gff_names[i]: per_gff_coverages[i] for i in range(len(gff_names))},
                               index=region_labels)
    coverage_df.index.name = "region_30bp"
    cov_csv = os.path.join(out_dir, f"{ctag}__coverage.csv")
    coverage_df.to_csv(cov_csv)

    # Compute fractions vs control and fractional occupancy (1 - fraction) for all non-controls.
    control_cov = per_gff_coverages[control_index]
    non_idx = [i for i in range(len(gff_paths_in_user_order)) if i != control_index]
    non_names = [gff_names[i] for i in non_idx]
    non_covs = [per_gff_coverages[i] for i in non_idx]
    fracs, frac_occ = fractions_vs_control(control_cov, non_covs)
    frac_df = pd.DataFrame({non_names[j]: fracs[j] for j in range(len(non_names))},
                           index=region_labels)
    frac_df.index_name = "region_30bp"
    frac_occ_df = pd.DataFrame({non_names[j]: frac_occ[j] for j in range(len(non_names))},
                               index=region_labels)
    frac_occ_df.index_name = "region_30bp"
    frac_csv = os.path.join(out_dir, f"{ctag}__fractions_vs_control.csv")
    frac_df.to_csv(frac_csv)
    fracocc_csv = os.path.join(out_dir, f"{ctag}__fractional_occupancy.csv")
    frac_occ_df.to_csv(fracocc_csv)

    # Arrange fractional occupancy into a matrix (rows=regions, cols=non-control datasets).
    frac_occ_mat = np.vstack(frac_occ).T if frac_occ else np.zeros((len(centers), 0))

    # ---- Fit Kd per region using non-control concentrations and fractional occupancy. ----
    tick(0.62, "Fitting Kd per region…")
    non_concs = concs_arr[non_idx]
    if non_concs.size != frac_occ_mat.shape[1]:
        raise RuntimeError("Non-control concentrations misaligned with matrix columns.")

    kd_vals = np.full(frac_occ_mat.shape[0], np.nan, dtype=float)
    kd_ok = np.zeros(frac_occ_mat.shape[0], dtype=bool)
    per_region_xy = []  # stores (x_with_zero_sorted, y_with_zero_sorted, kd, success)

    for r in range(frac_occ_mat.shape[0]):
        y_nonctrl = np.clip(frac_occ_mat[r, :], 0.0, 1.0)
        kd, ok, _ = fit_kd_langmuir(non_concs, y_nonctrl)
        kd_vals[r] = kd
        kd_ok[r] = ok

        # Reconstruct x,y arrays used for plotting (prepend 0 if needed, then sort).
        x = np.asarray(non_concs, float)
        if not np.any(np.isclose(x, 0.0)):
            x_plot = np.concatenate(([0.0], x))
            y_plot = np.concatenate(([0.0], y_nonctrl))
        else:
            x_plot = x.copy()
            y_plot = y_nonctrl.copy()
            y_plot[np.isclose(x_plot, 0.0)] = 0.0
        idx_sort = np.argsort(x_plot)
        x_plot = x_plot[idx_sort]
        y_plot = y_plot[idx_sort]
        per_region_xy.append((x_plot, y_plot, kd, ok))

    kd_df = pd.DataFrame({
        "center": [str(c) for c in centers],
        "Kd_nM": kd_vals,
        "fit_success": kd_ok.astype(int)
    })
    kd_csv = os.path.join(out_dir, f"{ctag}__kd_estimates.csv")
    kd_df.to_csv(kd_csv, index=False)

    # ---- Heatmap of fractional occupancy with side panel of Kd values. ----
    tick(0.74, "Rendering heatmap and Kd side plot…")
    valid_mask = np.isfinite(kd_vals)
    sort_keys = np.where(valid_mask, kd_vals, np.inf)
    sort_idx = np.argsort(sort_keys)

    center_labels = [str(c) for c in centers]
    sorted_centers = [center_labels[i] for i in sort_idx]
    sorted_mat = frac_occ_mat[sort_idx, :] if frac_occ_mat.size else frac_occ_mat
    sorted_kd = kd_vals[sort_idx]

    heatmap_png = os.path.join(out_dir, f"{ctag}__heatmap_fractional_occupancy__Kd.png")
    if sorted_mat.size > 0:
        n_rows, n_cols = sorted_mat.shape
        heatmap_w = max(6, 0.5 * max(1, n_cols))
        total_w = heatmap_w * 1.5
        height = max(6, 0.20 * max(1, n_rows))

        fig, (ax_hm, ax_dot) = plt.subplots(
            nrows=1, ncols=2, figsize=(total_w, height),
            gridspec_kw={"width_ratios": [2, 1]}, sharey=True
        )
        cmap = black_to_red_cmap()
        im = ax_hm.imshow(sorted_mat, aspect="auto", interpolation="nearest",
                          cmap=cmap, vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax_hm); cbar.set_label("Fractional Occupancy")
        ax_hm.set_title(f"Fractional Occupancy vs 0 nM control\nCenters: {centers_base}")
        ax_hm.set_xlabel("Non-control GFFs (selected order)")
        ax_hm.set_ylabel("Centers (sorted by Kd, ascending)")
        ax_hm.set_xticks(np.arange(len(non_names)))
        ax_hm.set_xticklabels(non_names, rotation=45, ha="right", fontsize=9)
        ax_hm.set_yticks(np.arange(len(sorted_centers)))
        ax_hm.set_yticklabels(sorted_centers, fontsize=8)

        # Kd scatter on a shared y-axis.
        y_idx = np.arange(len(sorted_centers))
        mask_in = np.isfinite(sorted_kd) & (sorted_kd <= 1000.0)
        mask_off = np.isfinite(sorted_kd) & (sorted_kd > 1000.0)
        if np.any(mask_in):
            ax_dot.scatter(sorted_kd[mask_in], y_idx[mask_in],
                           s=36, alpha=0.9, marker='o', edgecolors='none', label="Kd in range")
        if np.any(mask_off):
            ax_dot.scatter(np.full(np.sum(mask_off), 1000.0), y_idx[mask_off],
                           s=72, alpha=0.9, marker='>', edgecolors='none', label="Kd > 1000 nM")

        ax_dot.set_xscale("log")
        ax_dot.set_xlim(0.1, 1000.0)
        ax_dot.set_xlabel("Kd (nM, log scale)")
        ax_dot.grid(axis="x", linestyle=":", alpha=0.5)
        # Hide y tick labels on Kd axis; heatmap provides the labels.
        ax_dot.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax_dot.spines['left'].set_visible(False)
        ax_dot.set_ylim(ax_hm.get_ylim())

        plt.tight_layout()
        plt.savefig(heatmap_png, dpi=int(max(50, float(heatmap_dpi))), bbox_inches="tight")
        plt.close(fig)
    else:
        with open(heatmap_png.replace(".png", "__EMPTY.txt"), "w") as f:
            f.write("No non-control GFFs selected; heatmap not generated.\n")

    # ---- Alternating PDF: coverage bar page then Kd model page for each center. ----
    tick(0.82, "Rendering alternating Bars/Kd pages (PDF)…")
    pdf_path = os.path.join(out_dir, f"{ctag}__bars_and_kd_plots.pdf")
    with PdfPages(pdf_path) as pdf:
        x_curve = np.logspace(-1, 3, 200)  # 0.1–1000 nM grid for smooth model curve

        # Use dataset order from gff_names for consistent page layout.
        cov_dicts = [loaded[i][2] for i in range(len(loaded))]

        for r, center in enumerate(centers):
            # (1) Coverage bar page.
            render_bars_page(
                pdf,
                center=center,
                gff_names=gff_names,
                cov_dicts=cov_dicts,
                concs_in_user_order=concs_in_user_order,
                flank=int(bar_flank_bp),
                page_width=3.9,
                height_per_row=0.72,
                save_png=bool(save_bar_png),
                png_dpi=int(bar_png_dpi),
                out_dir=out_dir,
                ctag=ctag,
                pdf_pad_inches=0.5,
                bar_color=BAR_COLOR,
            )

            # (2) Kd model page for the same center.
            x_plot, y_plot, kd, ok = per_region_xy[r]
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            mask_pos = x_plot > 0
            if np.any(mask_pos):
                ax.scatter(x_plot[mask_pos], y_plot[mask_pos], s=24, alpha=0.9, label="Data", edgecolors='none')

            if np.isfinite(kd) and kd > 0:
                y_curve = x_curve / (kd + x_curve)
                ax.plot(x_curve, y_curve, linewidth=1.8,
                        label=f"Model (Kd={kd:.3g} nM)" if ok else "Model (fit failed)",
                        zorder=2)

            ax.set_xscale("log")
            ax.set_xlim(0.1, 1000.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("Protein concentration (nM, log scale)", fontsize=5)
            ax.set_ylabel("Fractional Occupancy", fontsize=5)
            ttl_kd = f"{kd:.3g} nM" if np.isfinite(kd) else "NaN"
            fit_note = "" if ok else " [fit not converged]"
            ax.set_title(f"Center {center} — Kd {ttl_kd}{fit_note}", fontsize=10)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            ax.legend(loc="lower right", fontsize=8, frameon=False)

            pdf.savefig(fig, dpi=200, bbox_inches="tight", pad_inches=0.5)
            plt.close(fig)

            # Optional individual PNG export of model page.
            if save_png_models:
                png_path = os.path.join(out_dir, f"{ctag}__center_{center}__kd_model.png")
                fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
                if np.any(mask_pos):
                    ax2.scatter(x_plot[mask_pos], y_plot[mask_pos], s=28, alpha=0.9, label="Data", edgecolors='none')
                if np.isfinite(kd) and kd > 0:
                    y_curve = x_curve / (kd + x_curve)
                    ax2.plot(x_curve, y_curve, linewidth=2.0, label=f"Model (Kd={kd:.3g} nM)" if ok else "Model (fit failed)")
                ax2.set_xscale("log")
                ax2.set_xlim(0.1, 1000.0)
                ax2.set_ylim(0.0, 1.0)
                ax2.set_xlabel("Protein concentration (nM, log scale)", fontsize=5)
                ax2.set_ylabel("Fractional Occupancy", fontsize=5)
                ax2.set_title(f"Center {center} — Kd {ttl_kd}{fit_note}")
                ax2.grid(axis="x", linestyle=":", alpha=0.4)
                ax2.legend(loc="lower right", fontsize=8, frameon=False)
                plt.tight_layout()
                plt.savefig(png_path, dpi=200)
                plt.close(fig2)

    # ---- Line graphs (±100 nt) and per-center CSVs for each non-control vs control. ----
    tick(0.88, "Computing line graphs…")
    control_dict = loaded[control_index][2]
    ctrl_tag = short_tag(gff_names[control_index])

    for i, non_name in zip(non_idx, non_names):
        non_dict = loaded[i][2]
        offsets, diffs = compute_diffs_matrix(control_dict, non_dict, centers, flank=100)

        # Save raw per-center, per-offset differences: control, non-control, and (control - non-control).
        rows = []
        for row_idx, c in enumerate(centers):
            for col_idx, off in enumerate(offsets):
                pos = c + off
                ctrl = control_dict.get(pos, 0.0)
                nonv = non_dict.get(pos, 0.0)
                rows.append((c, off, ctrl, nonv, diffs[row_idx, col_idx]))
        per_center_df = pd.DataFrame(
            rows, columns=["center", "offset", "control_cov", "noncontrol_cov", "diff"]
        )
        per_center_csv = os.path.join(
            out_dir,
            f"{ctag}__{short_tag(non_name)}_vs_{ctrl_tag}__percenter.csv"
        )
        per_center_df.to_csv(per_center_csv, index=False)

        # Compute estimator profiles over offsets and save plots/CSVs.
        for est in selected_estimators:
            prof = profile_from_diffs(diffs, est, params)
            suffix = ESTIMATOR_SUFFIX[est]
            label = ESTIMATOR_LABELS[est]

            out_csv = os.path.join(out_dir, f"{ctag}__{short_tag(non_name)}_vs_{ctrl_tag}__{suffix}.csv")
            pd.DataFrame({"offset": offsets, f"{suffix}_control_minus_noncontrol": prof}).to_csv(out_csv, index=False)

            out_png = os.path.join(out_dir, f"{ctag}__{short_tag(non_name)}_vs_{ctrl_tag}__{suffix}.png")
            plt.figure(figsize=(8, 4.2))
            plt.plot(offsets, prof, linewidth=2)
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlim(offsets[0], offsets[-1])
            if y_limits:
                plt.ylim(y_limits[0], y_limits[1])
            plt.xlabel("Position relative to center (nt)", fontsize=5)
            plt.ylabel(label, fontsize=5)
            title = f"{label.split(' (')[0]} (±100 nt)\n{non_name} vs {gff_names[control_index]}"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()

    tick(0.98, "Finalizing…")
    return {
        "coverage_csv": cov_csv,
        "fractions_csv": frac_csv,
        "fractional_occupancy_csv": fracocc_csv,
        "kd_csv": kd_csv,
        "heatmap_png": heatmap_png if os.path.exists(heatmap_png) else None,
        "kd_models_pdf": pdf_path,
        "n_centers": len(centers),
    }

# ========== GUI ==========
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kd estimation (0 nM = control) + line-graph estimators + bar pages (±bp)")

        # Window size is adapted to screen; scrolling handles overflow.
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1140, max(720, sw - 40))
        h = min(880,  max(560, sh - 80))
        self.geometry(f"{w}x{h}+40+40")
        self.minsize(720, 560)

        # Main scrollable container.
        self.scroll = ScrollableFrame(self, width=w-20, height=h-20)
        self.scroll.pack(fill="both", expand=True)
        self.body = self.scroll.body

        # Data and state.
        self.gff_paths = []
        self.centers_paths = []
        self.output_folder = ""
        self.status_var = tk.StringVar(value="Add datasets, set order & concentrations (one must be 0 nM), choose estimators, then Run")

        # Per-dataset concentrations (nM) aligned to the displayed order.
        self.conc_vars = []   # list[tk.DoubleVar]

        # Estimator selections for line graphs.
        self.use_mean = tk.BooleanVar(value=True)
        self.use_median = tk.BooleanVar(value=True)
        self.use_trimmed = tk.BooleanVar(value=False)
        self.use_winsor = tk.BooleanVar(value=False)
        self.use_huber = tk.BooleanVar(value=False)
        self.use_tukey = tk.BooleanVar(value=False)

        # Estimator parameters.
        self.trim_pct = tk.DoubleVar(value=10.0)   # percent for trimmed mean
        self.win_pct  = tk.DoubleVar(value=10.0)   # percent for winsorized mean
        self.huber_k  = tk.DoubleVar(value=1.5)    # Huber k * MAD
        self.tukey_k  = tk.DoubleVar(value=4.685)  # Tukey k * MAD

        # Optional per-region Kd PNG export.
        self.save_png_models = tk.BooleanVar(value=False)

        # Heatmap output DPI.
        self.heatmap_dpi = tk.IntVar(value=300)

        # Coverage bar-page options.
        self.bar_flank_bp = tk.IntVar(value=60)         # ±bp around center
        self.save_bar_png = tk.BooleanVar(value=False)  # PNG export toggle
        self.bar_png_dpi  = tk.IntVar(value=300)        # PNG DPI

        self._build_gui()

    def _build_gui(self):
        pad = {"padx": 8, "pady": 6}

        # --- GFF selection ---
        gff_frame = ttk.LabelFrame(self.body, text="GFF files (coverage; tab-delimited) — coord=col4, coverage=col6")
        gff_frame.pack(fill="x", **pad)
        ttk.Button(gff_frame, text="Add GFF files…", command=self.add_gffs).grid(row=0, column=0, sticky="w", **pad)

        self.gff_list = tk.Listbox(gff_frame, height=6, selectmode=tk.SINGLE, exportselection=False)
        self.gff_list.grid(row=1, column=0, columnspan=5, sticky="nsew", **pad)
        gff_frame.grid_rowconfigure(1, weight=1)

        # --- Order & concentrations (ALL datasets) ---
        order_frame = ttk.LabelFrame(self.body, text="Datasets: order (heatmap columns) & concentrations (nM)")
        order_frame.pack(fill="x", **pad)

        self.order_list = tk.Listbox(order_frame, height=7, selectmode=tk.SINGLE, exportselection=False)
        self.order_list.grid(row=0, column=0, rowspan=4, sticky="nsew", **pad)
        order_frame.grid_columnconfigure(0, weight=1)

        ttk.Button(order_frame, text="↑ Move Up", command=lambda: self.move_selected(-1)).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(order_frame, text="↓ Move Down", command=lambda: self.move_selected(1)).grid(row=1, column=1, sticky="we", **pad)

        conc_frame = ttk.Frame(order_frame)
        conc_frame.grid(row=0, column=2, rowspan=4, sticky="nsew", **pad)
        ttk.Label(conc_frame, text="Enter concentration (nM) for each dataset.\nExactly one must be 0 nM (control).").grid(row=0, column=0, sticky="w")

        self.conc_entries_frame = ttk.Frame(conc_frame)
        self.conc_entries_frame.grid(row=1, column=0, sticky="nsew")
        conc_frame.grid_columnconfigure(0, weight=1)

        # --- Centers files ---
        centers_frame = ttk.LabelFrame(self.body, text="Centers files (.txt) — one integer coordinate per line")
        centers_frame.pack(fill="x", **pad)
        ttk.Button(centers_frame, text="Add centers .txt files…", command=self.add_centers).grid(row=0, column=0, sticky="w", **pad)
        self.centers_list = tk.Listbox(centers_frame, height=5, selectmode=tk.EXTENDED, exportselection=False)
        self.centers_list.grid(row=1, column=0, columnspan=2, sticky="nsew", **pad)
        centers_frame.grid_columnconfigure(0, weight=1)

        # --- Estimator options for line graphs ---
        est_frame = ttk.LabelFrame(self.body, text="Line-graph estimators (apply per offset across centers)")
        est_frame.pack(fill="x", **pad)

        cb_frame = ttk.Frame(est_frame)
        cb_frame.grid(row=0, column=0, sticky="w", **pad)
        ttk.Checkbutton(cb_frame, text="Mean",    variable=self.use_mean).grid(row=0, column=0, sticky="w", padx=(0,12))
        ttk.Checkbutton(cb_frame, text="Median",  variable=self.use_median).grid(row=0, column=1, sticky="w", padx=(0,12))
        ttk.Checkbutton(cb_frame, text="Trimmed", variable=self.use_trimmed).grid(row=0, column=2, sticky="w", padx=(0,12))
        ttk.Checkbutton(cb_frame, text="Winsor",  variable=self.use_winsor).grid(row=0, column=3, sticky="w", padx=(0,12))
        ttk.Checkbutton(cb_frame, text="Huber",   variable=self.use_huber).grid(row=0, column=4, sticky="w", padx=(0,12))
        ttk.Checkbutton(cb_frame, text="Tukey",   variable=self.use_tukey).grid(row=0, column=5, sticky="w", padx=(0,12))

        param_frame = ttk.Frame(est_frame)
        param_frame.grid(row=1, column=0, sticky="w", **pad)
        ttk.Label(param_frame, text="Trim %").grid(row=0, column=0, sticky="e", padx=(0,4))
        ttk.Spinbox(param_frame, from_=0, to=49, increment=1, textvariable=self.trim_pct, width=6).grid(row=0, column=1, sticky="w", padx=(0,16))
        ttk.Label(param_frame, text="Winsor %").grid(row=0, column=2, sticky="e", padx=(0,4))
        ttk.Spinbox(param_frame, from_=0, to=49, increment=1, textvariable=self.win_pct, width=6).grid(row=0, column=3, sticky="w", padx=(0,16))
        ttk.Label(param_frame, text="Huber k×MAD").grid(row=0, column=4, sticky="e", padx=(0,4))
        ttk.Spinbox(param_frame, from_=0.1, to=10.0, increment=0.1, textvariable=self.huber_k, width=6).grid(row=0, column=5, sticky="w", padx=(0,16))
        ttk.Label(param_frame, text="Tukey k×MAD").grid(row=0, column=6, sticky="e", padx=(0,4))
        ttk.Spinbox(param_frame, from_=0.5, to=10.0, increment=0.1, textvariable=self.tukey_k, width=6).grid(row=0, column=7, sticky="w", padx=(0,16))

        # --- Output options ---
        out_frame = ttk.LabelFrame(self.body, text="Output")
        out_frame.pack(fill="x", **pad)
        self.out_label_var = tk.StringVar(value="(no folder selected)")
        ttk.Button(out_frame, text="Choose output folder…", command=self.choose_output).grid(row=0, column=0, sticky="w", **pad)
        ttk.Label(out_frame, textvariable=self.out_label_var).grid(row=0, column=1, sticky="w", **pad)

        # Toggle for per-region Kd model PNGs.
        ttk.Checkbutton(out_frame, text="Also save individual PNGs for Kd model plots",
                        variable=self.save_png_models).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0,6))

        # Heatmap DPI control.
        dpi_frame = ttk.Frame(out_frame)
        dpi_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0,6))
        ttk.Label(dpi_frame, text="Heatmap PNG DPI:").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(dpi_frame, from_=50, to=2400, increment=50, width=7,
                    textvariable=self.heatmap_dpi).grid(row=0, column=1, sticky="w", padx=(6,0))
        ttk.Label(dpi_frame, text="(default 300)").grid(row=0, column=2, sticky="w", padx=(6,0))

        # Coverage bar-page options.
        bar_frame = ttk.LabelFrame(self.body, text="Coverage Bar Pages (±bp window & PNG export)")
        bar_frame.pack(fill="x", **pad)
        ttk.Label(bar_frame, text="Flank length (bp) on each side (default 60):").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(bar_frame, from_=10, to=2000, increment=10, width=8,
                    textvariable=self.bar_flank_bp).grid(row=0, column=1, sticky="w", padx=(6,16))

        ttk.Checkbutton(bar_frame, text="Also save a PNG of each bar page",
                        variable=self.save_bar_png).grid(row=0, column=2, sticky="w", padx=(6,16))
        ttk.Label(bar_frame, text="Bar page PNG DPI:").grid(row=0, column=3, sticky="e")
        ttk.Spinbox(bar_frame, from_=50, to=2400, increment=50, width=7,
                    textvariable=self.bar_png_dpi).grid(row=0, column=4, sticky="w", padx=(6,0))

        # --- Progress + Run ---
        prog_frame = ttk.Frame(self.body)
        prog_frame.pack(fill="x", **pad)
        prog_frame.grid_columnconfigure(0, weight=1)
        self.prog = ttk.Progressbar(prog_frame, orient="horizontal", mode="determinate", maximum=100)
        self.prog.grid(row=0, column=0, sticky="we", padx=8, pady=(4,4))
        ttk.Button(prog_frame, text="Run", command=self.run_pipeline).grid(row=0, column=1, sticky="e", padx=(8,8), pady=(4,4))

        ttk.Label(self.body, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10, pady=(0,10))

    # ----- GUI actions -----
    def add_gffs(self):
        paths = filedialog.askopenfilenames(
            title="Select GFF coverage files",
            filetypes=[("GFF / tab files", "*.gff *.tsv *.txt"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            if p not in self.gff_paths:
                self.gff_paths.append(p)
                self.gff_list.insert(tk.END, p)
        self.refresh_order_and_concs()

    def refresh_order_and_concs(self):
        self.order_list.delete(0, tk.END)
        for p in self.gff_paths:
            self.order_list.insert(tk.END, os.path.basename(p))
        self._rebuild_conc_entries()

    def _rebuild_conc_entries(self):
        for w in getattr(self, "conc_entries_frame", []).winfo_children():
            w.destroy()
        self.conc_vars = []
        names = [self.order_list.get(i) for i in range(self.order_list.size())]
        ttk.Label(self.conc_entries_frame, text="Dataset").grid(row=0, column=0, sticky="w", padx=(0,8))
        ttk.Label(self.conc_entries_frame, text="Conc (nM)").grid(row=0, column=1, sticky="w")
        for r, name in enumerate(names, start=1):
            ttk.Label(self.conc_entries_frame, text=name, width=48).grid(row=r, column=0, sticky="w", padx=(0,8), pady=2)
            default = 0.0 if r == 1 else 1.0
            var = tk.DoubleVar(value=default)
            ttk.Entry(self.conc_entries_frame, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2)
            self.conc_vars.append(var)

    def move_selected(self, delta):
        i = self.order_list.curselection()
        if not i:
            return
        idx = i[0]; new_idx = idx + delta
        if new_idx < 0 or new_idx >= self.order_list.size():
            return
        text = self.order_list.get(idx)
        self.order_list.delete(idx)
        self.order_list.insert(new_idx, text)
        self.order_list.selection_set(new_idx)
        self._rebuild_conc_entries()

    def add_centers(self):
        paths = filedialog.askopenfilenames(
            title="Select centers .txt files (one coordinate per line)",
            filetypes=[("Text files", "*.txt *.tsv"), ("All files", "*.*")]
        )
        if not paths:
            return
        for p in paths:
            if p not in self.centers_paths:
                self.centers_paths.append(p)
                self.centers_list.insert(tk.END, p)

    def choose_output(self):
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_folder = folder
            self.out_label_var.set(folder)

    # ----- helpers -----
    def _progress(self, frac, msg):
        """Update progress bar and status line."""
        frac = 0.0 if frac < 0 else (1.0 if frac > 1.0 else frac)
        self.prog["value"] = int(frac * 100)
        self.status_var.set(msg)
        self.update_idletasks()

    def _read_concentrations(self):
        """Read and validate user-entered concentrations; exactly one must be 0 nM."""
        try:
            concs = [float(v.get()) for v in self.conc_vars]
        except Exception:
            messagebox.showerror("Invalid concentration", "Please enter numeric concentrations (nM).")
            return None
        if any(c < 0 for c in concs):
            messagebox.showerror("Invalid concentration", "Concentrations must be non-negative.")
            return None
        zeros = [i for i, c in enumerate(concs) if abs(c) < 1e-12]
        if len(zeros) != 1:
            messagebox.showerror("0 nM control required", "Exactly one dataset must be 0 nM (the control).")
            return None
        return concs

    def _gather_estimators(self):
        """Collect selected estimator keys."""
        ests = []
        if self.use_mean.get():    ests.append("mean")
        if self.use_median.get():  ests.append("median")
        if self.use_trimmed.get(): ests.append("trimmed")
        if self.use_winsor.get():  ests.append("winsor")
        if self.use_huber.get():   ests.append("huber")
        if self.use_tukey.get():   ests.append("tukey")
        return ests

    def run_pipeline(self):
        """Validate inputs, pre-scan for global y-limits, and process each centers file."""
        try:
            if not self.gff_paths:
                messagebox.showerror("Missing input", "Please add at least one GFF file.")
                return
            if not self.centers_paths:
                messagebox.showerror("Missing centers", "Please add at least one centers .txt file.")
                return
            if not self.output_folder:
                messagebox.showerror("Missing output folder", "Please choose an output folder.")
                return

            # Build dataset order from the listbox; require a 1:1 match with selected files.
            name_to_path = {os.path.basename(p): p for p in self.gff_paths}
            ordered_names = [self.order_list.get(i) for i in range(self.order_list.size())]
            if set(ordered_names) != set(name_to_path.keys()):
                messagebox.showerror("Order mismatch", "The order list must contain all selected datasets exactly once.")
                return
            gff_paths_in_user_order = [name_to_path[n] for n in ordered_names]

            concs_in_user_order = self._read_concentrations()
            if concs_in_user_order is None:
                return

            selected_estimators = self._gather_estimators()
            if not selected_estimators:
                messagebox.showerror("No estimators", "Please select at least one line-graph estimator.")
                return

            # -------- Pre-scan for global y-axis limits across all centers and estimators. --------
            self._progress(0.02, "Scanning for global y-axis limits…")
            loaded_scan = []
            for p in gff_paths_in_user_order:
                positions, coverages, cov_dict = load_gff_sparse(p)
                loaded_scan.append((positions, coverages, cov_dict))
            concs_arr = np.asarray(concs_in_user_order, float)
            zero_mask = np.isclose(concs_arr, 0.0)
            if np.sum(zero_mask) != 1:
                messagebox.showerror("0 nM control required", "Exactly one dataset must be 0 nM.")
                return
            control_index = int(np.where(zero_mask)[0][0])
            non_idx_scan = [i for i in range(len(gff_paths_in_user_order)) if i != control_index]

            global_min, global_max = None, None
            total_files = len(self.centers_paths)
            for path_idx, rpath in enumerate(self.centers_paths, start=1):
                centers = read_centers_txt(rpath)
                if not centers:
                    continue
                ctrl_dict = loaded_scan[control_index][2]
                for i in non_idx_scan:
                    non_dict = loaded_scan[i][2]
                    offsets, diffs = compute_diffs_matrix(ctrl_dict, non_dict, centers, flank=100)
                    mins, maxs = [], []
                    for est in selected_estimators:
                        prof = profile_from_diffs(diffs, est, dict(
                            trim=max(0.0, min(0.49, self.trim_pct.get()/100.0)),
                            win=max(0.0, min(0.49, self.win_pct.get()/100.0)),
                            huber_k=max(0.01, float(self.huber_k.get())),
                            tukey_k=max(0.01, float(self.tukey_k.get()))
                        ))
                        mins.append(float(np.min(prof)))
                        maxs.append(float(np.max(prof)))
                    smin, smax = min(mins), max(maxs)
                    global_min = smin if global_min is None else min(global_min, smin)
                    global_max = smax if global_max is None else max(global_max, smax)
                self._progress(0.02 + 0.10*(path_idx/max(1,total_files)), "Scanning for global y-axis limits…")

            # Choose shared y-limits with a small padding. Fallback if data are degenerate.
            if global_min is None or global_max is None:
                global_min, global_max = -1.0, 1.0
            span = global_max - global_min
            if span <= 0:
                val = global_max
                pad = max(1.0, abs(val)*0.05)
                y_limits = (val - pad, val + pad)
            else:
                pad = max(1e-9, 0.05*span)
                y_limits = (global_min - pad, global_max + pad)

            # -------- Main processing for each centers file. --------
            self._progress(0.14, "Starting…")
            params = dict(
                trim=max(0.0, min(0.49, self.trim_pct.get()/100.0)),
                win=max(0.0, min(0.49, self.win_pct.get()/100.0)),
                huber_k=max(0.01, float(self.huber_k.get())),
                tukey_k=max(0.01, float(self.tukey_k.get())),
            )

            summary_lines = []
            nfiles = len(self.centers_paths)
            for k, rpath in enumerate(self.centers_paths, start=1):
                def file_hook(local_frac, msg):
                    overall = 0.14 + 0.86 * (((k - 1) + local_frac) / nfiles)
                    self._progress(overall, f"[{k}/{nfiles}] {msg}")

                res = process_one_centers_file(
                    rpath,
                    gff_paths_in_user_order,
                    concs_in_user_order,
                    self.output_folder,
                    selected_estimators,
                    params,
                    y_limits=y_limits,
                    progress_hook=file_hook,
                    save_png_models=self.save_png_models.get(),
                    heatmap_dpi=int(self.heatmap_dpi.get()),
                    bar_flank_bp=int(self.bar_flank_bp.get()),
                    save_bar_png=self.save_bar_png.get(),
                    bar_png_dpi=int(self.bar_png_dpi.get()),
                )
                line = (
                    f"Processed '{os.path.basename(rpath)}' → "
                    f"coverage: {os.path.relpath(res['coverage_csv'], self.output_folder)}, "
                    f"fractions: {os.path.relpath(res['fractions_csv'], self.output_folder)}, "
                    f"fractional_occ: {os.path.relpath(res['fractional_occupancy_csv'], self.output_folder)}, "
                    f"Kd: {os.path.relpath(res['kd_csv'], self.output_folder)}, "
                    f"heatmap: {os.path.relpath(res['heatmap_png'], self.output_folder) if res['heatmap_png'] else '—'}, "
                    f"Bars+Kd PDF: {os.path.relpath(res['kd_models_pdf'], self.output_folder)}"
                )
                summary_lines.append(line)

            self._progress(1.0, "Done.")
            messagebox.showinfo("Completed", "All centers files processed.\n\n" + "\n".join(summary_lines))

        except Exception as e:
            self._progress(0.0, "Error.")
            messagebox.showerror("Error while running", str(e))

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
