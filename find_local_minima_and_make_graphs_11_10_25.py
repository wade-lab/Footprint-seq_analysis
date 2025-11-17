#!/usr/bin/env python3
"""
Volcano-density plotting and local-minima selection (with a simple Tkinter GUI).

Overview
--------
This tool reads a whitespace/tab-delimited file with three columns:
    1) coord  : genomic coordinate (integer)
    2) pvalue : statistical p-value (float; expected > 0)
    3) ratio  : fold-change ratio (float > 0), or the string "inf"

It produces:
    - A volcano-density (hexbin) PNG for all valid points:
        * X-axis: log2(ratio)
        * Y-axis: -log10(pvalue)
    - A CSV listing "selected minima" (coordinate, p-value, ratio text)
    - A volcano-density PNG for only the selected minima

GUI controls let you choose:
    - Input file path (3-column text as above)
    - Replacement magnitude M for plotting edge cases:
        * ratio == 0   → plotted at x = -M
        * ratio == "inf" → plotted at x = +M
    - Hexbin grid size (density resolution)
    - Output folder (defaults to the input file's directory if not supplied)

Minima definition
-----------------
"Local minima" are selected greedily by ascending p-value. Once a coordinate
is selected, all coordinates within ±60 nucleotides are excluded from further
selection. This enforces a minimum spacing of 60 nt between reported sites and
ensures each chosen site represents a local low p-value.
"""

# =========================
# Windows 11 / Tkinter DPI & scaling drop-in fix
# =========================
import os
import sys
import math
import platform
import ctypes

# Best-effort OS-level DPI awareness for crisp Tk rendering on Windows.
# No-ops safely on other platforms or if APIs are unavailable.
if platform.system() == "Windows":
    try:
        # Prefer per-monitor DPI awareness on recent Windows versions.
        awareness_contexts = [
            -4,  # PER_MONITOR_AWARE_V2
            -3,  # PER_MONITOR_AWARE
            -2,  # SYSTEM_AWARE
        ]
        user32 = ctypes.windll.user32
        SetProcessDpiAwarenessContext = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if SetProcessDpiAwarenessContext:
            for ctx in awareness_contexts:
                if SetProcessDpiAwarenessContext(ctypes.c_void_p(ctx)):
                    break
        else:
            # Fallback to older system API.
            shcore = ctypes.windll.shcore
            shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            # Last-resort legacy call on some Windows builds.
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

# =========================
# Standard imports
# =========================
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------
# Rendering defaults
# -----------------------
# PNG export resolution (dots per inch).
DEFAULT_PNG_DPI = 300
# Matplotlib figure size in inches (width, height).
DEFAULT_FIGSIZE = (6.5, 5.0)
# The hexbin "gridsize" (density resolution) is provided by the GUI at runtime.

# -----------------------
# Utility: safe parsing
# -----------------------
def parse_ratio(s):
    """
    Parse the ratio column:
      - "inf"  → (math.inf, "inf")  (keep text for CSV)
      - float  → (value, original_text)
      - other  → (None, original_text)  (row will be skipped in plotting)
    """
    s = s.strip()
    if s.lower() == "inf":
        return math.inf, "inf"
    try:
        v = float(s)
        return v, s
    except Exception:
        return None, s

def parse_pval(s):
    """Parse p-value as float; return None if not parseable."""
    try:
        v = float(s)
        return v
    except Exception:
        return None

def safe_neglog10(p, eps=1e-300):
    """
    Compute -log10(p) with clipping at a very small epsilon to avoid log(0).
    """
    p_clipped = max(p, eps)
    return -math.log10(p_clipped)

def safe_log2_ratio(ratio, repl_mag):
    """
    Map ratio values to X-axis coordinates:
      - ratio == "inf" → +repl_mag
      - ratio == 0     → -repl_mag
      - ratio  > 0     → log2(ratio)
      - ratio <= 0 or None → return None (skip point)
    """
    if ratio is None:
        return None
    if ratio is math.inf:
        return float(repl_mag)
    if ratio == 0.0:
        return -float(repl_mag)
    if ratio > 0:
        return math.log2(ratio)
    return None

# -----------------------
# Minima selection
# -----------------------
def select_minima(coords, pvals, window=60):
    """
    Greedy selection based on ascending p-values.
    A coordinate is kept if it is > 'window' nt away from all previously selected coords.
    Returns a numpy array of selected indices (sorted by coordinate).
    """
    order = np.argsort(pvals)
    selected = []
    selected_coords = []

    for idx in order:
        c = coords[idx]
        if all(abs(c - sc) > window for sc in selected_coords):
            selected.append(idx)
            selected_coords.append(c)
    return np.array(sorted(selected, key=lambda i: coords[i]))

# -----------------------
# Plotting
# -----------------------
def make_hexbin(xvals, yvals, title, outpath, gridsize=80, dpi=DEFAULT_PNG_DPI, figsize=DEFAULT_FIGSIZE):
    """
    Create and save a hexbin density plot:
      - X: log2(ratio) (including replacements for 0/inf)
      - Y: -log10(p-value)
      - Color scale: log10 of bin counts
    Saves a PNG to 'outpath' using the specified DPI and figure size.
    """
    x = np.array(xvals, dtype=float)
    y = np.array(yvals, dtype=float)

    # Keep only finite points; discard NaN/Inf introduced by transformations.
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    if x.size == 0:
        raise ValueError("No valid points to plot.")

    plt.figure(figsize=figsize, dpi=dpi)
    hb = plt.hexbin(x, y, gridsize=gridsize, bins='log')
    plt.xlabel("log2(ratio)")
    plt.ylabel("-log10(p-value)")
    plt.title(title)
    cb = plt.colorbar(hb)
    cb.set_label("log10(count)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi)  # Explicit DPI for consistent raster resolution
    plt.close()

# -----------------------
# Data load
# -----------------------
def load_3col_txt(path):
    """
    Read a 3-column text file (coord, pvalue, ratio), ignoring comments/blank lines.
    The coordinate is accepted as int or a float with an integer value (robust parsing).
    Returns:
      coords       : numpy array of ints
      pvals        : numpy array of floats
      ratios       : numpy array of objects (float, math.inf, or None)
      ratio_texts  : numpy array of original ratio strings (e.g., "inf")
    """
    coords = []
    pvals = []
    ratios = []
    ratio_texts = []

    with open(path, 'r', newline='') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            c_str, p_str, r_str = parts[0], parts[1], parts[2]

            # Coordinate: allow "123.0" to be treated as 123
            try:
                c = int(float(c_str))
            except Exception:
                continue

            p = parse_pval(p_str)
            if p is None:
                continue

            r_val, r_txt = parse_ratio(r_str)

            coords.append(c)
            pvals.append(p)
            ratios.append(r_val)
            ratio_texts.append(r_txt)

    return (np.array(coords),
            np.array(pvals, dtype=float),
            np.array(ratios, dtype=object),
            np.array(ratio_texts, dtype=object))

# -----------------------
# Main workflow
# -----------------------
def run_workflow(infile, outdir, repl_mag, gridsize, dpi=DEFAULT_PNG_DPI, figsize=DEFAULT_FIGSIZE):
    """
    Full pipeline for a single input file:
      1) Parse data
      2) Build volcano arrays for all valid points
      3) Render and save "all points" hexbin PNG
      4) Select local minima and save a CSV
      5) Render and save "minima only" hexbin PNG
    Returns the paths to the two PNGs and the CSV, plus the number of minima.
    """
    coords, pvals, ratios, ratio_texts = load_3col_txt(infile)

    if coords.size == 0:
        raise ValueError("No valid rows parsed from input file.")

    # Build volcano arrays (all points).
    x_all = []
    y_all = []
    for r, p in zip(ratios, pvals):
        x = safe_log2_ratio(r, repl_mag)
        if x is None:
            continue
        y = safe_neglog10(p)
        x_all.append(x)
        y_all.append(y)

    base = os.path.splitext(os.path.basename(infile))[0]
    out_png_all = os.path.join(outdir, f"{base}_volcano_all.png")
    make_hexbin(
        x_all, y_all,
        f"Volcano density (all points): {base}",
        out_png_all,
        gridsize=gridsize,
        dpi=dpi,
        figsize=figsize
    )

    # Select coordinates that are local minima (spacing window = 60 nt).
    chosen_idx = select_minima(coords, pvals, window=60)

    # Write CSV of selected minima, preserving the original ratio text (e.g., "inf").
    out_csv = os.path.join(outdir, f"{base}_selected_minima.csv")
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["coord", "pvalue", "ratio"])
        for i in chosen_idx:
            writer.writerow([int(coords[i]), f"{pvals[i]:.10g}", ratio_texts[i]])

    # Build volcano arrays for minima only.
    x_min = []
    y_min = []
    for i in chosen_idx:
        x = safe_log2_ratio(ratios[i], repl_mag)
        if x is None:
            continue
        y = safe_neglog10(pvals[i])
        x_min.append(x)
        y_min.append(y)

    out_png_min = os.path.join(outdir, f"{base}_volcano_minima.png")
    make_hexbin(
        x_min, y_min,
        f"Volcano density (minima only): {base}",
        out_png_min,
        gridsize=gridsize,
        dpi=dpi,
        figsize=figsize
    )

    return out_png_all, out_png_min, out_csv, len(chosen_idx)

# -----------------------
# GUI
# -----------------------
class App:
    """
    Minimal Tkinter interface:
      - Choose input text file and output folder
      - Set replacement magnitude (M) and hexbin grid size
      - Run the workflow and report where outputs were saved
    """
    def __init__(self, root):
        self.root = root
        root.title("Volcano Density (log2 ratio) + Local Minima (60 nt)")

        # Apply Tk scaling to improve text sizing on high-DPI displays.
        try:
            # 1.0 corresponds to ~96 DPI; increase if your UI looks too small.
            root.call('tk', 'scaling', 1.0)
        except Exception:
            pass

        # Input file selection row.
        tk.Label(root, text="Input 3-column .txt (coord, pvalue, ratio):").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.in_entry = tk.Entry(root, width=60)
        self.in_entry.grid(row=0, column=1, padx=6, pady=6)
        tk.Button(root, text="Browse...", command=self.browse_in).grid(row=0, column=2, padx=6, pady=6)

        # Output folder selection row.
        tk.Label(root, text="Output folder:").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        self.out_entry = tk.Entry(root, width=60)
        self.out_entry.grid(row=1, column=1, padx=6, pady=6)
        tk.Button(root, text="Browse...", command=self.browse_out).grid(row=1, column=2, padx=6, pady=6)

        # Replacement magnitude input (controls where zero/inf map on X).
        tk.Label(root, text="Replacement magnitude M (ratio=0 → -M, ratio=inf → +M):").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.repl_entry = tk.Entry(root, width=12)
        self.repl_entry.insert(0, "6")
        self.repl_entry.grid(row=2, column=1, sticky="w", padx=6, pady=6)

        # Hexbin grid size input (density resolution).
        tk.Label(root, text="Hexbin grid size (density resolution):").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        self.grid_entry = tk.Entry(root, width=12)
        self.grid_entry.insert(0, "80")
        self.grid_entry.grid(row=3, column=1, sticky="w", padx=6, pady=6)

        # Run button spanning the layout width.
        tk.Button(root, text="Run", command=self.run).grid(row=4, column=0, columnspan=3, pady=12)

        # Status area to show output paths and counts.
        self.status = tk.Label(root, text="", fg="blue", anchor="w", justify="left")
        self.status.grid(row=5, column=0, columnspan=3, sticky="we", padx=6, pady=6)

    def browse_in(self):
        """Open a file chooser for the 3-column input text file."""
        path = filedialog.askopenfilename(
            title="Select 3-column .txt",
            filetypes=[("Text files", "*.txt *.tsv *.csv *.dat *.out"), ("All files", "*.*")]
        )
        if path:
            self.in_entry.delete(0, tk.END)
            self.in_entry.insert(0, path)
            # Default output folder to the input file's directory if blank.
            outdir = os.path.dirname(path)
            if not self.out_entry.get().strip():
                self.out_entry.insert(0, outdir)

    def browse_out(self):
        """Open a folder chooser for the output directory."""
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, path)

    def run(self):
        """Validate inputs, run the pipeline, and report results."""
        infile = self.in_entry.get().strip()
        outdir = self.out_entry.get().strip() or os.path.dirname(infile)
        try:
            repl_mag = float(self.repl_entry.get().strip())
            gridsize = int(float(self.grid_entry.get().strip()))
        except Exception:
            messagebox.showerror("Error", "Replacement magnitude and grid size must be numeric.")
            return

        if not infile or not os.path.isfile(infile):
            messagebox.showerror("Error", "Please select a valid input file.")
            return
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder:\n{e}")
                return

        try:
            png_all, png_min, csv_path, nmin = run_workflow(
                infile, outdir, repl_mag, gridsize,
                dpi=DEFAULT_PNG_DPI, figsize=DEFAULT_FIGSIZE
            )
            self.status.config(
                text=f"Done.\nAll points plot: {png_all}\nMinima-only plot: {png_min}\nSelected minima CSV ({nmin} rows): {csv_path}"
            )
            messagebox.showinfo("Success", f"Completed.\nMinima selected: {nmin}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
