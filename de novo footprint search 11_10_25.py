#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    import ctypes, sys
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", module="statsmodels")

import pandas as pd
import statsmodels.formula.api as smf

import tkinter as tk
from tkinter import filedialog, messagebox

# -------------------------
# Parameters
# -------------------------
GENOME_SIZE = 5_000_000
WINDOW = 30
THRESHOLD = 2

# -------------------------
# Data loading
# -------------------------
def read_in_coverage_values(gff_path, genome_size=GENOME_SIZE):
    """
    Builds a genome-length coverage vector by summing |score| per position.
    Expects a GFF-like file where:
      - column 4 (0-based index 3) is a 1-based coordinate
      - column 6 (0-based index 5) holds the numeric score
    Lines with score==0 are ignored.
    """
    genome_coverage = [0.0] * genome_size
    with open(gff_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                coord = int(parts[3])
                val = abs(float(parts[5]))
                if val > 0:
                    # Protect against coordinates outside range
                    if 0 <= coord < genome_size:
                        genome_coverage[coord] += val
            except (IndexError, ValueError):
                # Skip malformed lines
                continue
    return genome_coverage

# -------------------------
# Core analysis
# -------------------------
def analyze(rep1_0, rep2_0, rep1_exp, rep2_exp, window=WINDOW, threshold=THRESHOLD):
    features = list(range(window))  # 0..29
    output_data = []

    # iterate across genome
    for x in range(GENOME_SIZE):
        R1_0   = rep1_0[x: x + window]
        R2_0   = rep2_0[x: x + window]
        R1_exp = rep1_exp[x: x + window]
        R2_exp = rep2_exp[x: x + window]

        sum0  = sum(R1_0) + sum(R2_0)
        sumEX = sum(R1_exp) + sum(R2_exp)

        if sum0 > 0:
            ratio = (sumEX / sum0) if sum0 != 0 else float("inf")
            if ratio < 1/threshold or ratio > threshold:
                df = pd.DataFrame({
                    "feature":   features * 4,
                    "condition": ["X"]*len(R1_0)*2 + ["Y"]*len(R1_exp)*2,
                    "replicate": [1]*len(R1_0) + [2]*len(R2_0) + [1]*len(R1_exp) + [2]*len(R2_exp),
                    "value":     R1_0 + R2_0 + R1_exp + R2_exp
                })
                try:
                    model  = smf.mixedlm("value ~ condition", df, groups=df["feature"])
                    result = model.fit()
                    pval   = result.pvalues.get("condition[T.Y]", float("nan"))
                except Exception:
                    pval = float("nan")
                output_data.append([x + (window // 2), pval, ratio])

        elif sum0 == 0 and sumEX > 0:
            df = pd.DataFrame({
                "feature":   features * 4,
                "condition": ["X"]*len(R1_0)*2 + ["Y"]*len(R1_exp)*2,
                "replicate": [1]*len(R1_0) + [2]*len(R2_0) + [1]*len(R1_exp) + [2]*len(R2_exp),
                "value":     R1_0 + R2_0 + R1_exp + R2_exp
            })
            try:
                model  = smf.mixedlm("value ~ condition", df, groups=df["feature"])
                result = model.fit()
                pval   = result.pvalues.get("condition[T.Y]", float("nan"))
            except Exception:
                pval = float("nan")
            output_data.append([x + (window // 2), pval, "inf"])

    return output_data

# -------------------------
# File I/O
# -------------------------
def write_output(output_path, rows):
    with open(output_path, "w") as f:
        for row in rows:
            f.write("\t".join(str(v) for v in row) + "\n")

# -------------------------
# Simple Tkinter GUI pickers
# -------------------------
def main():
    root = tk.Tk()
    root.withdraw()  # no main window; just dialogs

    messagebox.showinfo(
        "Select files",
        "Choose the four input GFF files in order:\n"
        "1) CONTROL Replicate 1\n"
        "2) CONTROL Replicate 2\n"
        "3) EXPERIMENTAL Replicate 1\n"
        "4) EXPERIMENTAL Replicate 2\n"
        "Then choose an output filename."
    )

    ctrl1_path = filedialog.askopenfilename(
        title="Select CONTROL replicate 1 (.gff)",
        filetypes=[("GFF files", "*.gff"), ("All files", "*.*")]
    )
    if not ctrl1_path:
        messagebox.showwarning("Cancelled", "No file selected.")
        return

    ctrl2_path = filedialog.askopenfilename(
        title="Select CONTROL replicate 2 (.gff)",
        filetypes=[("GFF files", "*.gff"), ("All files", "*.*")]
    )
    if not ctrl2_path:
        messagebox.showwarning("Cancelled", "No file selected.")
        return

    exp1_path = filedialog.askopenfilename(
        title="Select EXPERIMENTAL replicate 1 (.gff)",
        filetypes=[("GFF files", "*.gff"), ("All files", "*.*")]
    )
    if not exp1_path:
        messagebox.showwarning("Cancelled", "No file selected.")
        return

    exp2_path = filedialog.askopenfilename(
        title="Select EXPERIMENTAL replicate 2 (.gff)",
        filetypes=[("GFF files", "*.gff"), ("All files", "*.*")]
    )
    if not exp2_path:
        messagebox.showwarning("Cancelled", "No file selected.")
        return

    out_path = filedialog.asksaveasfilename(
        title="Save output as",
        defaultextension=".txt",
        filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
    )
    if not out_path:
        messagebox.showwarning("Cancelled", "No output file selected.")
        return

    # Load data
    messagebox.showinfo("Working", "Loading CONTROL Replicate 1...")
    rep1_0 = read_in_coverage_values(ctrl1_path)

    messagebox.showinfo("Working", "Loading CONTROL Replicate 2...")
    rep2_0 = read_in_coverage_values(ctrl2_path)

    messagebox.showinfo("Working", "Loading EXPERIMENTAL Replicate 1...")
    rep1_exp = read_in_coverage_values(exp1_path)

    messagebox.showinfo("Working", "Loading EXPERIMENTAL Replicate 2...")
    rep2_exp = read_in_coverage_values(exp2_path)

    # Analyze
    messagebox.showinfo("Working", "Running mixed-effects analysis; this may take a while…")
    results = analyze(rep1_0, rep2_0, rep1_exp, rep2_exp, window=WINDOW, threshold=THRESHOLD)

    # Write
    write_output(out_path, results)
    messagebox.showinfo("Done", f"Finished.\nResults written to:\n{out_path}")

if __name__ == "__main__":
    main()
