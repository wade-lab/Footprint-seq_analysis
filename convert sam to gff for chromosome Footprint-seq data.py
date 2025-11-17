import os
import re
import threading
from collections import Counter
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure headless-safe (non-interactive backend for batch/GUI use)
import matplotlib.pyplot as plt

# -------------------
# Core logic
# -------------------

def log_write(log_fn, msg: str):
    """
    Simple logging helper:
      - If a log function is provided (e.g. GUI logger), send the message there.
      - Otherwise, print to stdout.
    """
    if log_fn:
        log_fn(msg)
    else:
        print(msg)

# SAM → coordinate extraction

def extract_coordinates(path: str):
    """
    Parse a SAM file and extract transposition-adjusted coordinates.

    Assumptions / behavior:
      - Only lines with exactly 11 tab-separated fields are used.
        (This ignores SAM optional fields by design.)
      - FLAG field (column 2) is treated as a string and checked for:
          * "0"   → plus strand
          * "16"  → minus strand
        Other flags are ignored.
      - pos = int(fields[3]) is the 1-based leftmost mapping position.
      - seq = fields[9] is the read sequence; its length is used for minus-strand math.
      - Plus strand:
          coord = pos + 4
        to account for a 4-nt duplication at the transposition site.
      - Minus strand:
          coord = pos - 1 - 4 + len(seq)
        i.e., rightmost base (pos + len(seq) - 1) minus 4 nt duplication.
      - Returns:
          A list of integer coordinates (no normalization, no RPM here).
    """
    coords = []
    with open(path, "r") as f:
        for line in f:
            # Split on tabs; require strict 11 fields (standard SAM without optional tags)
            fields = line.split("\t")
            if len(fields) != 11:
                continue
            flag = fields[1]       # FLAG field as string ("0", "16", etc.)
            pos = int(fields[3])   # 1-based leftmost position
            seq = fields[9]        # read sequence (for length)

            if flag == "0":  # plus strand
                # For plus strand, shift the coordinate +4 to account for 4-nt duplication
                coords.append(4 + pos)  # add 4 nt to account for sequence duplication in transposition
            elif flag == "16":  # minus strand
                # For minus strand, use (pos - 1 + len(seq)) as rightmost base,
                # then subtract 4 to account for sequence duplication
                coords.append(pos - 1 - 4 + len(seq)) # subtract 4 nt to account for sequence duplication in transposition
          
    return coords


def clean_end_R(name_stem: str) -> str:
    """Remove trailing _R1 or _R2 (or -R1/-R2) from a stem."""
    return re.sub(r'[_-]?R[12]$', '', name_stem)


def shared_stem(stem1: str, stem2: str) -> str:
    """If cleaned stems match, use it once; else fall back to 'stem1_and_stem2'."""
    return stem1 if stem1 == stem2 else f"{stem1}_and_{stem2}"


def sam_pair_to_gff(sam_r1: str, sam_r2: str, outdir: str, log_fn=None) -> str:
    """
    Merge an R1/R2 pair into a single GFF (RPM-normalized). Returns GFF path.

    Steps:
      1) Extract coordinates from R1 and R2 using extract_coordinates().
      2) Build a shared "base" stem from the filenames (with _R1/_R2 trimmed).
      3) Count occurrences of each coordinate.
      4) Convert counts to RPM (reads per million).
      5) Write a simple 9-column GFF-like file:
           seqid = "NA"
           source = "Agilent"
           type   = track_name (derived from shared stem)
           start  = pos
           end    = pos
           score  = coverage (RPM)
           strand, phase, attributes = "."
      6) Log progress via log_write.
    """
    coords = extract_coordinates(sam_r1) + extract_coordinates(sam_r2)

    stem1 = clean_end_R(os.path.basename(sam_r1)[:-4])
    stem2 = clean_end_R(os.path.basename(sam_r2)[:-4])
    base = shared_stem(stem1, stem2)  # single clean stem for the merged pair
    track_name = base
    out_path = os.path.join(outdir, f"{base}.gff")

    # If no coordinates, create an empty GFF file but still log and return the path
    if not coords:
        open(out_path, "w").close()
        log_write(log_fn, f"⚠️ No coordinates for pair → {os.path.basename(out_path)} (empty)")
        return out_path

    # Count how many times each coordinate appears
    counts = Counter(coords)
    total = float(len(coords))  # total number of reads/coordinates
    # Normalize to RPM for each coordinate
    for k in list(counts.keys()):
        counts[k] = counts[k] * 1_000_000.0 / total  # RPM

    # Write out the coverage as a simple position-based GFF
    with open(out_path, "w") as f:
        for pos, cov in counts.items():
            f.write(f"NA\tAgilent\t{track_name}\t{pos}\t{pos}\t{cov}\t.\t.\t.\n")
    log_write(log_fn, f"✅ Wrote merged GFF: {os.path.basename(out_path)}")
    return out_path


# GFF → Series loader (for heatmap)

def load_series(gff_path: str) -> pd.Series:
    """
    Load a GFF file and return coverage as a pandas Series indexed by coordinate.

    Input:
      - gff_path: path to a GFF-like file with coverage in column 6.

    Behavior:
      - Reads only columns 4 (start), 5 (end), and 6 (coverage).
      - Interprets them as strings first; coerces to numeric.
      - Drops rows where start/end/cov are invalid.
      - Rounds start positions to the nearest integer coordinate.
      - Aggregates coverage by coordinate via sum.
      - Returns:
          Series where index = int coordinate, values = float coverage.
        Returns an empty float Series if the file has no valid data.
    """
    # Expect: col4=start, col5=end, col6=coverage
    df = pd.read_csv(
        gff_path,
        sep="\t",
        header=None,
        usecols=[3, 4, 5],
        comment="#",
        names=["start", "end", "cov"],
        dtype={"start": "object", "end": "object", "cov": "object"},
        engine="python",
    )
    if df.empty:
        return pd.Series(dtype=float)

    # Convert to numeric; invalid values become NaN
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["cov"] = pd.to_numeric(df["cov"], errors="coerce")
    # Remove rows with any NaNs in these critical columns
    df = df.dropna(subset=["start", "end", "cov"])

    # Coordinate = rounded start position; coverage = float
    df = pd.DataFrame({"coord": df["start"].round().astype(int),
                       "cov": df["cov"].astype(float)})
    # Sum coverage per coordinate
    s = df.groupby("coord", sort=False)["cov"].sum()
    return s


def build_aligned_matrix(series_map: dict) -> pd.DataFrame:
    """
    Build a coordinate-aligned wide matrix from a dict of Series.

    Input:
      series_map: {label: Series(coord -> coverage)}

    Behavior:
      - Collects all unique coordinates from all series.
      - Sorts coordinates to create a "master" index.
      - Reindexes each series onto this master index, filling missing with 0.0.
      - Returns:
          DataFrame with index = coord, columns = labels, values = coverage.
    """
    if not series_map:
        return pd.DataFrame()
    # All coordinates from all series, in sorted order
    master = sorted(set().union(*[s.index for s in series_map.values()]))
    # For each label, reindex its coverage series onto the master coordinate list
    cols = {name: s.reindex(master, fill_value=0.0).astype(float).values
            for name, s in series_map.items()}
    # Build wide DataFrame; dtype ensures float column types
    wide = pd.DataFrame(cols, index=master, dtype=float)
    wide.index.name = "coord"
    return wide


def correlation_matrix_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an R² (squared Pearson correlation) matrix from a wide coverage table.

    Input:
      wide: DataFrame where each column is a sample and each row is a coordinate.

    Behavior:
      - For each pair of columns (i, j), compute Pearson correlation.
      - If either column has zero standard deviation, set R² = 0.0 for that pair.
      - Diagonal entries are set to 1.0 (self-correlation).
      - Result is a symmetric n x n matrix of R² values.

    Returns:
      DataFrame with index/columns = sample names, values = R² (float).
    """
    names = list(wide.columns); n = len(names)
    out = np.zeros((n, n), dtype=float)
    X = wide.values  # underlying 2D numeric array
    for i in range(n):
        xi = X[:, i]; sxi = xi.std(ddof=1)  # sample std. dev. of column i
        for j in range(i, n):
            if i == j:
                # Perfect self-correlation by definition
                out[i, j] = 1.0; continue
            xj = X[:, j]; sxj = xj.std(ddof=1)
            if sxi == 0 or sxj == 0:
                # If either vector has no variance, correlation is undefined; treat as 0
                r2 = 0.0
            else:
                # Pearson correlation coefficient
                r = np.corrcoef(xi, xj)[0, 1]
                r2 = 0.0 if np.isnan(r) else float(r * r)
            # Symmetric fill
            out[i, j] = out[j, i] = r2
    return pd.DataFrame(out, index=names, columns=names)


def plot_heatmap_tri(corr: pd.DataFrame, out_png: str, title="Correlation (R²) Similarity"):
    """
    Plot a lower-triangular correlation (R²) heatmap and save to PNG.

    Input:
      corr    : square R² correlation matrix (DataFrame).
      out_png : output PNG path.
      title   : title string for the plot.

    Behavior:
      - Copies the matrix and sets the upper triangle (incl. diagonal) to NaN.
      - Uses a masked array so only the lower triangle is drawn.
      - Colors range from 0 to 1 (R²).
      - Adds labels on both axes.
      - Annotates each drawn cell with its numeric R² value.
      - Scales figure size with the number of samples.
    """
    n = corr.shape[0]
    data = corr.values.copy()
    # Indices of the upper triangle (including diagonal) to mask out
    iu = np.triu_indices(n, k=0)  # mask upper incl diagonal
    data[iu] = np.nan
    # Mask invalid entries for plotting
    M = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis.copy(); cmap.set_bad(alpha=0)  # transparent outside lower triangle
    # Scale figure size so that more samples produce a larger figure
    fig_w, fig_h = max(6, 0.5 * n), max(5, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(M, vmin=0, vmax=1, aspect="equal", cmap=cmap)
    labels = corr.columns.tolist()
    # Place ticks and labels for each sample
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    # Annotation font size shrinks as n grows
    ann_fs = max(3, 9 - 0.25 * n)
    # Add numeric R² text in each visible (non-NaN) cell
    for i in range(n):
        for j in range(n):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=ann_fs)
    # Colorbar for R² scale
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("R²")
    # Tight layout and save figure to disk
    plt.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)


def numeric_key(name: str):
    """Primary sort by first integer found (0,4,20,100,500), fallback alphabetical; non-numeric last."""
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (999999, name)


def label_from_gff_path(gff_path: str) -> str:
    """Label = GFF filename without extension (merged pairs are single-stem already)."""
    return os.path.splitext(os.path.basename(gff_path))[0]


# -------------------
# GUI Application
# -------------------
class App(tk.Tk):
    def __init__(self):
        """
        Main Tkinter application class.

        Responsibilities:
          - Let the user select multiple SAM files (expected to contain R1/R2 pairs).
          - Let the user choose an output folder and heatmap options.
          - Run the SAM→GFF→correlation heatmap pipeline in a background thread.
          - Display progress and log messages in the GUI.
        """
        super().__init__()
        self.title("SAM → GFF Correlations")
        self.geometry("980x640")
        self.minsize(900, 600)
        self._build_style()
        self._build_widgets()
        # Internal state
        self.sam_paths = []  # list of selected SAM file paths
        self.outdir = ""     # chosen output directory
        self.worker = None   # background worker thread handle

    def _build_style(self):
        """
        Configure ttk styles (theme and basic widget styling).
        """
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            # If 'clam' isn't available, keep the default theme silently
            pass
        self.style.configure("TButton", padding=6)
        self.style.configure("Accent.TButton", padding=8, font=("Segoe UI", 10, "bold"))
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

    def _build_widgets(self):
        """
        Build and lay out all GUI widgets:
          - Top: controls for adding/clearing SAM files.
          - Middle-left: listbox showing selected SAM files.
          - Middle-right: output folder, options, and Run button.
          - Bottom: log console and status bar.
        """
        # Top controls frame
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="1) Select SAM files (R1/R2 pairs)", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        btn_add = ttk.Button(top, text="Add SAMs…", command=self.on_add_sams)
        btn_add.grid(row=0, column=1, padx=8)
        btn_clear = ttk.Button(top, text="Clear", command=self.on_clear_sams)
        btn_clear.grid(row=0, column=2)

        # List of files
        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(mid, selectmode=tk.EXTENDED, activestyle="none")
        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        # Right panel: output folder + options + Run
        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        ttk.Label(right, text="2) Output folder", style="Header.TLabel").pack(anchor="w")
        self.outdir_var = tk.StringVar()
        out_row = ttk.Frame(right)
        out_row.pack(fill=tk.X, pady=4)
        self.out_entry = ttk.Entry(out_row, textvariable=self.outdir_var)
        self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse…", command=self.on_pick_outdir).pack(side=tk.LEFT, padx=6)

        # Options
        opts = ttk.LabelFrame(right, text="Options", padding=10)
        opts.pack(fill=tk.X, pady=(8, 0))
        self.var_make_avg = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts,
            text="Create averaged *_avg.gff for rep1+rep2 (excluded from heatmap)",
            variable=self.var_make_avg
        ).pack(anchor="w")
        self.var_csv_precision = tk.StringVar(value="6")
        row_prec = ttk.Frame(opts)
        row_prec.pack(fill=tk.X, pady=4)
        ttk.Label(row_prec, text="CSV float precision:").pack(side=tk.LEFT)
        ttk.Entry(row_prec, width=6, textvariable=self.var_csv_precision).pack(side=tk.LEFT, padx=6)
        self.var_title = tk.StringVar(value="Correlation (R²) Similarity")
        row_title = ttk.Frame(opts)
        row_title.pack(fill=tk.X, pady=4)
        ttk.Label(row_title, text="Heatmap title:").pack(side=tk.LEFT)
        ttk.Entry(row_title, textvariable=self.var_title).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # Run button
        run_row = ttk.Frame(right)
        run_row.pack(fill=tk.X, pady=(12, 0))
        self.btn_run = ttk.Button(run_row, text="Run", style="Accent.TButton", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT)
        self.prog = ttk.Progressbar(run_row, mode="indeterminate")
        self.prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Log console
        log_frame = ttk.LabelFrame(self, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.pack(fill=tk.BOTH, expand=True)

        # Footer
        foot = ttk.Frame(self, padding=10)
        foot.pack(fill=tk.X)
        self.status = ttk.Label(foot, text="Ready.")
        self.status.pack(side=tk.LEFT)

    # ------------ Events ------------
    def on_add_sams(self):
        """
        Handler for "Add SAMs…" button:
          - Opens a file selection dialog.
          - Adds newly selected SAM paths to internal list and listbox (no duplicates).
        """
        paths = filedialog.askopenfilenames(
            title="Select SAM files",
            filetypes=[("SAM files", "*.sam"), ("All files", "*.*")],
        )
        if not paths:
            return
        added = 0
        for p in paths:
            p = os.path.abspath(p)
            if p not in self.sam_paths:
                self.sam_paths.append(p)
                self.listbox.insert(tk.END, p)
                added += 1
        if added:
            self._log(f"➕ Added {added} files.")

    def on_clear_sams(self):
        """
        Handler for "Clear" button:
          - Empties the internal SAM path list and the on-screen listbox.
        """
        self.sam_paths.clear()
        self.listbox.delete(0, tk.END)
        self._log("🧹 Cleared file list.")

    def on_pick_outdir(self):
        """
        Handler for choosing the output directory:
          - Opens a folder picker.
          - Stores the chosen directory path and updates the entry box.
        """
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir = d
            self.outdir_var.set(d)

    def on_run(self):
        """
        Handler for "Run" button:
          - Validates that at least two SAM files are present.
          - Validates that an output folder is chosen.
          - Creates output directory if needed.
          - Disables Run button and starts the progress bar.
          - Spawns a background thread to do the actual processing.
        """
        if not self.sam_paths:
            messagebox.showwarning("No files", "Please add at least two SAM files (forming at least one R1/R2 pair).")
            return
        outdir = self.outdir_var.get().strip()
        if not outdir:
            messagebox.showwarning("No output folder", "Please choose an output folder.")
            return
        os.makedirs(outdir, exist_ok=True)

        # Disable run during processing
        self.btn_run.config(state=tk.DISABLED)
        self.prog.start(10)
        self.status.config(text="Running…")
        self._log("\n▶️ Started run at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Run in background thread
        self.worker = threading.Thread(target=self._do_work, args=(outdir,), daemon=True)
        self.worker.start()
        self.after(150, self._poll_worker)

    def _poll_worker(self):
        """
        Periodic check for whether the background worker thread is still alive.
        When it finishes:
          - Stop progress bar.
          - Re-enable Run button.
          - Update status and log a completion message.
        """
        if self.worker and self.worker.is_alive():
            self.after(150, self._poll_worker)
        else:
            self.prog.stop()
            self.btn_run.config(state=tk.NORMAL)
            self.status.config(text="Done.")
            self._log("✅ Finished.")

    # ------------ Core workflow ------------
    def _do_work(self, outdir: str):
        """
        Main pipeline executed in the background thread.

        Steps:
          1) Group SAM files into R1/R2 pairs by stripping _R1/_R2 (or -R1/-R2).
          2) For each pair, generate a merged GFF with RPM-normalized coverage.
          3) Optionally create averaged *_avg.gff for rep1/rep2 pairs.
          4) Build a coordinate-aligned coverage matrix from the merged GFFs.
          5) Compute R² correlation matrix and plot a triangular heatmap.
          6) Save coverage_matrix.csv, correlation_matrix.csv, and correlation_heatmap.png.
        """
        try:
            # Group into R1/R2 by removing trailing _R1/_R2 from stems
            stems = {p: os.path.basename(p)[:-4] for p in self.sam_paths}
            pair_buckets = {}
            for p, stem in stems.items():
                # Remove an optional underscore/hyphen plus R1/R2 to form a pair key
                pair_key = re.sub(r'[_-]?R[12]$', '', stem)
                pair_buckets.setdefault(pair_key, []).append(p)

            # Create per-pair GFFs (merged R1+R2) with a single shared stem
            gff_paths = []
            for key, files in pair_buckets.items():
                # Identify R1 and R2 by suffix in the stem
                r1 = [f for f in files if re.search(r'[_-]?R1$', os.path.basename(f)[:-4])]
                r2 = [f for f in files if re.search(r'[_-]?R2$', os.path.basename(f)[:-4])]
                if r1 and r2:
                    self._log(f"🔗 Pair: {os.path.basename(r1[0])}  +  {os.path.basename(r2[0])}")
                    gff = sam_pair_to_gff(r1[0], r2[0], outdir, log_fn=self._log)
                    gff_paths.append(gff)
                else:
                    # If both R1 and R2 aren't present, skip that pair key
                    self._log(f"⚠️ Skipping '{key}': need both R1 and R2.")

            if not gff_paths:
                self._log("❌ No GFFs were produced. Aborting.")
                return

            # Create averaged GFFs for rep1/rep2 pairs (NOT included in heatmap)
            if self.var_make_avg.get():
                by_core = {}
                for g in gff_paths:
                    name = os.path.splitext(os.path.basename(g))[0]
                    # Convert rep1 or rep2 to repX to pair them
                    core = re.sub(r"rep[12]", "repX", name)
                    by_core.setdefault(core, []).append(g)

                for core, files in by_core.items():
                    if len(files) == 2:
                        dfs = []
                        for fpath in files:
                            # Load just position and coverage columns from GFF
                            df = pd.read_csv(
                                fpath, sep="\t", header=None, usecols=[3, 5],
                                names=["pos", "cov"], dtype={"pos": "int64", "cov": "float64"},
                                engine="python",
                            )
                            dfs.append(df)
                        # Concatenate and compute mean coverage per position
                        merged = pd.concat(dfs, ignore_index=True).groupby("pos", as_index=False)["cov"].mean()
                        out_avg = os.path.join(outdir, f"{core}_avg.gff")
                        # Write averaged cov back out as GFF with same general schema
                        with open(out_avg, "w") as f:
                            for _, row in merged.iterrows():
                                pos_i = int(row["pos"]); cov_v = float(row["cov"])
                                f.write(f"NA\tAgilent\t{core}\t{pos_i}\t{pos_i}\t{cov_v}\t.\t.\t.\n")
                        self._log(f"🧮 Wrote averaged GFF: {os.path.basename(out_avg)}")

            # ----- Heatmap: EXCLUDE *_avg.gff -----
            series_map = {}
            for g in sorted(gff_paths):  # only merged pair GFFs
                label = label_from_gff_path(g)
                s = load_series(g)
                if not s.empty:
                    series_map[label] = s
                else:
                    self._log(f"(empty series) {os.path.basename(g)}")

            if len(series_map) < 2:
                self._log("❌ Need at least two non-empty datasets to build a heatmap.")
                return

            # Order labels by numeric portion (if present) for more intuitive layout
            ordered_labels = [name for _, name in sorted((numeric_key(n) for n in series_map.keys()))]
            wide = build_aligned_matrix({n: series_map[n] for n in ordered_labels})
            corr = correlation_matrix_from_wide(wide)

            # Save outputs
            float_prec = int(self.var_csv_precision.get() or 6)
            heatmap_png = os.path.join(outdir, "correlation_heatmap.png")
            plot_heatmap_tri(corr, heatmap_png, title=self.var_title.get().strip() or "Correlation (R²) Similarity")
            wide.to_csv(os.path.join(outdir, "coverage_matrix.csv"), float_format=f"%.{float_prec}f")
            corr.to_csv(os.path.join(outdir, "correlation_matrix.csv"), float_format=f"%.{float_prec}f")

            # Log summary of output files
            self._log("\n📦 Outputs saved to: " + outdir)
            self._log(" - correlation_heatmap.png")
            self._log(" - coverage_matrix.csv")
            self._log(" - correlation_matrix.csv")
            self._log(" - merged pair .gff files")
            if self.var_make_avg.get():
                self._log(" - averaged rep .gff files (excluded from heatmap)")
        except Exception as e:
            # Catch-all error report (keeps GUI alive)
            self._log("❌ Error: " + str(e))

    # ------------ Utilities ------------
    def _log(self, text: str):
        """
        Append a line of text to the log Text widget and keep it read-only.

        Notes:
          - Enables the widget, inserts text, auto-scrolls to the end,
            then disables it again to prevent user editing.
        """
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")


def main():
    """
    Entrypoint for running the GUI application.
    """
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
