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
matplotlib.use("Agg")  # Use non-interactive backend so plots can be saved without a display
import matplotlib.pyplot as plt

# -------------------
# Helpers / logging
# -------------------

def log_write(log_fn, msg: str):
    """
    Helper to route log messages either to a provided logging function
    (typically the GUI log appender) or to stdout if no logger is supplied.
    """
    if log_fn:
        log_fn(msg)  # Send message to GUI log
    else:
        print(msg)   # Fallback: print to console

# -------------------
# Core logic (plasmid-aware coordinate extraction)
# -------------------

def extract_coordinates(path: str, plasmid_size: int):
    """
    Tandem 2-copy reference; STRICT parsing:
      - Only accept flag '0' (plus) or '16' (minus) as STRING.
      - Only accept SAM lines with exactly 11 fields (ignore optional tags).
      - Base coordinate (no shift): plus → POS; minus → POS + len(seq) - 1.
      - Fold copy #2 back to [1..plasmid_size] (no modulo).
      - Apply ATAC-like shift: +4 (plus), -4 (minus).
      - Wrap circularly into [1..plasmid_size].
    """
    coords = []  # List of final shifted & wrapped coordinates
    with open(path, "r") as f:
        for line in f:
            # Skip empty lines and SAM header lines (starting with '@')
            if not line or line.startswith("@"):
                continue
            fields = line.split("\t")
            if len(fields) != 11:  # strict: ignore lines with optional tags
                continue

            flag = fields[1]           # FLAG field as a string; expect '0' (plus) or '16' (minus)
            try:
                pos = int(fields[3])   # 1-based leftmost mapping position from SAM
                seq = fields[9]        # Read sequence (for determining read length)
            except (ValueError, IndexError):
                # Skip lines with malformed position or sequence field
                continue

            L = len(seq)               # Read length

            # 1) unshifted base per strand (strict flags)
            if flag == "0":           # plus
                base = pos            # Base coordinate is leftmost position
                shift = +4            # ATAC-style +4 bp shift for plus strand
            elif flag == "16":        # minus
                base = pos + L - 1    # Base coordinate is rightmost position
                shift = -4            # ATAC-style -4 bp shift for minus strand
            else:
                # Ignore all other flags (e.g., secondary, unmapped, etc.)
                continue              # ignore all other flags

            # 2) fold copy #2 back
            # Reference is a tandem duplication of the plasmid (2 copies in series).
            # Here we map coordinates from copy #2 back into the 1..plasmid_size range.
            if 1 <= base <= plasmid_size:
                folded = base
            elif plasmid_size < base < 2 * plasmid_size:
                folded = base - plasmid_size
            else:
                # skip exact 2*plasmid_size or out-of-range coordinates
                continue

            # 3) apply shift
            shifted = folded + shift

            # 4) wrap into [1..plasmid_size] (circular plasmid)
            if shifted < 1:
                shifted += plasmid_size
            elif shifted > plasmid_size:
                shifted -= plasmid_size

            # Store final coordinate
            coords.append(shifted)

    return coords


def clean_end_R(name_stem: str) -> str:
    """
    Remove a trailing R1/R2 (with optional preceding _ or -) from a filename stem.
    Example: 'sample_R1' -> 'sample', 'sample-R2' -> 'sample'.
    """
    return re.sub(r'[_-]?R[12]$', '', name_stem)


def shared_stem(stem1: str, stem2: str) -> str:
    """
    If two stems are identical, return that stem; otherwise return a combined name.
    Used to generate a common base name for an R1/R2 pair.
    """
    return stem1 if stem1 == stem2 else f"{stem1}_and_{stem2}"


def sam_pair_to_gff(sam_r1: str, sam_r2: str, outdir: str, plasmid_size: int, log_fn=None) -> str:
    """
    Take a pair of SAM files (R1 and R2) and:
      - Extract shifted plasmid coordinates from each.
      - Merge coordinates and compute coverage as RPM.
      - Write a GFF file where each position has one entry with coverage in the score field.
    Returns the path to the output GFF.
    """
    # Combine coordinates from both read files
    coords = extract_coordinates(sam_r1, plasmid_size) + extract_coordinates(sam_r2, plasmid_size)

    # Derive a clean base name for this R1/R2 pair
    stem1 = clean_end_R(os.path.basename(sam_r1)[:-4])  # drop ".sam" and trim R1/R2
    stem2 = clean_end_R(os.path.basename(sam_r2)[:-4])  # drop ".sam" and trim R1/R2
    base = shared_stem(stem1, stem2)
    track_name = base  # GFF "feature" name
    out_path = os.path.join(outdir, f"{base}.gff")

    # If no usable coordinates, write an empty file but still create it
    if not coords:
        open(out_path, "w").close()
        log_write(log_fn, f"⚠️ No coordinates for pair → {os.path.basename(out_path)} (empty)")
        return out_path

    # Count occurrences of each coordinate
    counts = Counter(coords)
    total = float(len(coords))  # total reads contributing

    # Convert counts to RPM (reads per million)
    for k in list(counts.keys()):
        counts[k] = counts[k] * 1_000_000.0 / total  # RPM

    # Write GFF file with coverage in column 6, position as start=end
    with open(out_path, "w") as f:
        for pos, cov in counts.items():
            f.write(f"NA\tAgilent\t{track_name}\t{pos}\t{pos}\t{cov}\t.\t.\t.\n")
    log_write(log_fn, f"✅ Wrote merged GFF: {os.path.basename(out_path)}")
    return out_path


# -------------------
# GFF → Series / Matrices / Plot
# -------------------

def load_series(gff_path: str) -> pd.Series:
    """
    Load a GFF file and return a pandas Series indexed by coordinate with coverage values.
    - Reads start, end, and coverage columns.
    - Coerces to numeric and drops invalid rows.
    - Rounds start positions to integers and sums coverage per coordinate.
    """
    df = pd.read_csv(
        gff_path,
        sep="\t",
        header=None,
        usecols=[3, 4, 5],  # start, end, cov
        comment="#",
        names=["start", "end", "cov"],
        dtype={"start": "object", "end": "object", "cov": "object"},
        engine="python",
    )
    if df.empty:
        # Return an empty float Series if file had no data rows
        return pd.Series(dtype=float)

    # Convert to numeric, coercing errors to NaN
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["cov"] = pd.to_numeric(df["cov"], errors="coerce")
    df = df.dropna(subset=["start", "end", "cov"])  # remove rows with any NaNs

    # Construct a minimal DataFrame with integer coordinates and float coverage
    df = pd.DataFrame({"coord": df["start"].round().astype(int),
                       "cov": df["cov"].astype(float)})
    # Group by coordinate and sum coverage if multiple entries share the same position
    s = df.groupby("coord", sort=False)["cov"].sum()
    return s


def build_aligned_matrix(series_map: dict) -> pd.DataFrame:
    """
    Given a mapping of {name: Series(coord -> coverage)}, build a wide DataFrame
    with all coordinates aligned on the index and each column corresponding to a sample.
    Missing positions are filled with 0.0 coverage.
    """
    if not series_map:
        return pd.DataFrame()
    # Build a sorted list of all coordinates present across all series
    master = sorted(set().union(*[s.index for s in series_map.values()]))
    # Reindex each series onto the master coordinate list and store as columns
    cols = {name: s.reindex(master, fill_value=0.0).astype(float).values
            for name, s in series_map.items()}
    wide = pd.DataFrame(cols, index=master, dtype=float)
    wide.index.name = "coord"  # Set index name for clarity
    return wide


def correlation_matrix_from_wide(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an R² (squared Pearson correlation) matrix from a wide coverage DataFrame.
    - Each column is a sample.
    - The output is an n x n DataFrame where entry (i, j) is R² between samples i and j.
    - If a sample has zero variance, R² is set to 0 for pairs involving that sample.
    """
    names = list(wide.columns); n = len(names)
    out = np.zeros((n, n), dtype=float)  # container for R² values
    X = wide.values                      # underlying 2D numpy array (rows: coords, cols: samples)
    for i in range(n):
        xi = X[:, i]
        sxi = xi.std(ddof=1)             # sample standard deviation of sample i
        for j in range(i, n):
            if i == j:
                # Perfect self-correlation
                out[i, j] = 1.0
                continue
            xj = X[:, j]
            sxj = xj.std(ddof=1)
            if sxi == 0 or sxj == 0:
                # If either vector is constant, correlation is undefined; treat as 0
                r2 = 0.0
            else:
                # np.corrcoef returns the 2x2 correlation matrix; [0,1] is corr(xi, xj)
                r = np.corrcoef(xi, xj)[0, 1]
                r2 = 0.0 if np.isnan(r) else float(r * r)
            # Symmetric matrix: fill both (i,j) and (j,i)
            out[i, j] = out[j, i] = r2
    return pd.DataFrame(out, index=names, columns=names)


def plot_heatmap_tri(corr: pd.DataFrame, out_png: str, title="Correlation (R²) Similarity"):
    """
    Plot a triangular correlation (R²) heatmap:
      - Uses the correlation matrix but masks the upper triangle (including diagonal) with NaN.
      - Displays values in the lower triangle as colors with numerical annotations.
      - Saves the figure to 'out_png' as a PNG.
    """
    n = corr.shape[0]
    data = corr.values.copy()
    # Indices of upper triangle (including diagonal)
    iu = np.triu_indices(n, k=0)
    # Mask out upper triangle by setting to NaN
    data[iu] = np.nan
    # Mask invalid (NaN) entries for imshow
    M = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis.copy(); cmap.set_bad(alpha=0)  # transparent where masked
    # Scale figure size with number of samples
    fig_w, fig_h = max(6, 0.5 * n), max(5, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    # Heatmap from lower triangle
    im = ax.imshow(M, vmin=0, vmax=1, aspect="equal", cmap=cmap)
    labels = corr.columns.tolist()
    # Tick labels and positions
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    # Annotation font size scales inversely with n (more samples → smaller font)
    ann_fs = max(3, 9 - 0.25 * n)
    # Add numeric annotations for non-NaN cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=ann_fs)
    # Colorbar for R² values
    cbar = plt.colorbar(im, ax=ax); cbar.set_label("R²")
    # Layout and save
    plt.tight_layout(); fig.savefig(out_png, dpi=300); plt.close(fig)


def numeric_key(name: str):
    """
    Extract the first integer found in 'name' to use as a sort key.
    Returns (number, name) if a number is found, otherwise a large default integer.
    This allows ordering like sample1, sample2, sample10, etc.
    """
    m = re.search(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (999999, name)


def label_from_gff_path(gff_path: str) -> str:
    """
    Convert a GFF file path into a simple label by stripping directory and extension.
    """
    return os.path.splitext(os.path.basename(gff_path))[0]

# -------------------
# GUI Application
# -------------------
class App(tk.Tk):
    def __init__(self):
        """
        Main Tkinter application:
          - Lets user select SAM files (R1/R2 pairs).
          - Lets user pick output directory and options.
          - On 'Run', performs SAM→GFF conversion and correlation heatmap generation in a worker thread.
        """
        super().__init__()
        self.title("Plasmid SAM → GFF Correlations")
        self.geometry("1020x680")
        self.minsize(900, 600)
        self._build_style()
        self._build_widgets()
        # Internal state
        self.sam_paths = []   # List of absolute paths to SAM files
        self.outdir = ""      # Output directory path
        self.worker = None    # Background thread handle

    def _build_style(self):
        """
        Configure ttk styles (theme, buttons, headers).
        """
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")  # Prefer 'clam' theme if available
        except tk.TclError:
            pass  # Silently ignore if theme is not available
        self.style.configure("TButton", padding=6)
        self.style.configure("Accent.TButton", padding=8, font=("Segoe UI", 10, "bold"))
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))

    def _build_widgets(self):
        """
        Build and lay out all GUI widgets:
          - Top frame: file selection controls.
          - Middle frame: listbox of SAM files and right-hand options.
          - Bottom: log window and status bar.
        """
        # Top control bar
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="1) Select SAM files (R1/R2 pairs)", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Add SAMs…", command=self.on_add_sams).grid(row=0, column=1, padx=8)
        ttk.Button(top, text="Clear", command=self.on_clear_sams).grid(row=0, column=2)

        # Middle: list of SAMs + right-side panel
        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(fill=tk.BOTH, expand=True)
        # Listbox showing selected SAM files
        self.listbox = tk.Listbox(mid, selectmode=tk.EXTENDED, activestyle="none")
        sb = ttk.Scrollbar(mid, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        # Right-hand panel with options and run button
        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        ttk.Label(right, text="2) Output folder", style="Header.TLabel").pack(anchor="w")
        self.outdir_var = tk.StringVar()
        out_row = ttk.Frame(right)
        out_row.pack(fill=tk.X, pady=4)
        # Entry showing chosen output directory
        self.out_entry = ttk.Entry(out_row, textvariable=self.outdir_var)
        self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(out_row, text="Browse…", command=self.on_pick_outdir).pack(side=tk.LEFT, padx=6)

        # Options group
        opts = ttk.LabelFrame(right, text="Options", padding=10)
        opts.pack(fill=tk.X, pady=(8, 0))

        # Plasmid parameters
        row_ps = ttk.Frame(opts); row_ps.pack(fill=tk.X, pady=2)
        self.var_plasmid_size = tk.StringVar(value="9010")  # Default plasmid size
        ttk.Label(row_ps, text="Plasmid size (bp):").pack(side=tk.LEFT)
        ttk.Entry(row_ps, width=10, textvariable=self.var_plasmid_size).pack(side=tk.LEFT, padx=6)

        # Whether to compute averaged *_avg.gff for rep1+rep2
        self.var_make_avg = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts,
            text="Create averaged *_avg.gff for rep1+rep2 (excluded from heatmap)",
            variable=self.var_make_avg
        ).pack(anchor="w", pady=(6,0))

        # CSV float precision option
        self.var_csv_precision = tk.StringVar(value="6")
        row_prec = ttk.Frame(opts); row_prec.pack(fill=tk.X, pady=4)
        ttk.Label(row_prec, text="CSV float precision:").pack(side=tk.LEFT)
        ttk.Entry(row_prec, width=6, textvariable=self.var_csv_precision).pack(side=tk.LEFT, padx=6)

        # Heatmap title option
        self.var_title = tk.StringVar(value="Correlation (R²) Similarity")
        row_title = ttk.Frame(opts); row_title.pack(fill=tk.X, pady=4)
        ttk.Label(row_title, text="Heatmap title:").pack(side=tk.LEFT)
        ttk.Entry(row_title, textvariable=self.var_title).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # Run button + progress bar
        run_row = ttk.Frame(right)
        run_row.pack(fill=tk.X, pady=(12, 0))
        self.btn_run = ttk.Button(run_row, text="Run", style="Accent.TButton", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT)
        self.prog = ttk.Progressbar(run_row, mode="indeterminate")
        self.prog.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        # Log window at bottom
        log_frame = ttk.LabelFrame(self, text="Log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self.log.pack(fill=tk.BOTH, expand=True)

        # Status bar
        foot = ttk.Frame(self, padding=10)
        foot.pack(fill=tk.X)
        self.status = ttk.Label(foot, text="Ready.")
        self.status.pack(side=tk.LEFT)

    # ------------ Events ------------
    def on_add_sams(self):
        """
        Event handler: open a file dialog to add SAM files to the list.
        Avoids duplicate paths and logs how many were added.
        """
        paths = filedialog.askopenfilenames(
            title="Select SAM files",
            filetypes=[("SAM files", "*.sam"), ("All files", "*.*")],
        )
        if not paths:
            return
        added = 0
        for p in paths:
            p = os.path.abspath(p)  # store absolute path to avoid duplicates with different forms
            if p not in self.sam_paths:
                self.sam_paths.append(p)
                self.listbox.insert(tk.END, p)
                added += 1
        if added:
            self._log(f"➕ Added {added} files.")

    def on_clear_sams(self):
        """
        Event handler: clear the list of selected SAM files and reset the listbox.
        """
        self.sam_paths.clear()
        self.listbox.delete(0, tk.END)
        self._log("🧹 Cleared file list.")

    def on_pick_outdir(self):
        """
        Event handler: open a directory chooser to set the output folder.
        """
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.outdir = d
            self.outdir_var.set(d)

    def on_run(self):
        """
        Event handler: validate inputs, then start background processing thread.
        Handles:
          - Ensuring there are enough SAM files.
          - Ensuring an output directory is chosen.
          - Parsing and validating plasmid size.
        """
        if not self.sam_paths:
            messagebox.showwarning("No files", "Please add at least two SAM files (forming at least one R1/R2 pair).")
            return
        outdir = self.outdir_var.get().strip()
        if not outdir:
            messagebox.showwarning("No output folder", "Please choose an output folder.")
            return
        try:
            plasmid_size = int(self.var_plasmid_size.get())
            if plasmid_size <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Bad parameters", "Plasmid size must be a positive integer.")
            return

        # Ensure output directory exists
        os.makedirs(outdir, exist_ok=True)

        # Disable run button and start progress indicator
        self.btn_run.config(state=tk.DISABLED)
        self.prog.start(10)
        self.status.config(text="Running…")
        self._log("\n▶️ Started run at " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        # Start worker thread to perform the heavy lifting without freezing the GUI
        self.worker = threading.Thread(target=self._do_work, args=(outdir, plasmid_size), daemon=True)
        self.worker.start()
        # Begin polling to know when the worker is finished
        self.after(150, self._poll_worker)

    def _poll_worker(self):
        """
        Periodically check whether the worker thread is still alive.
        When done, stop the progress bar, re-enable the Run button, and update status.
        """
        if self.worker and self.worker.is_alive():
            # Worker still running; poll again after a short delay
            self.after(150, self._poll_worker)
        else:
            # Worker finished (or never started); clean up UI state
            self.prog.stop()
            self.btn_run.config(state=tk.NORMAL)
            self.status.config(text="Done.")
            self._log("✅ Finished.")

    # ------------ Core workflow ------------
    def _do_work(self, outdir: str, plasmid_size: int):
        """
        Main pipeline executed in the background thread:
          1) Group SAM files into R1/R2 pairs.
          2) Convert each pair to a merged GFF with coverage in RPM.
          3) Optionally create averaged *_avg.gff for rep1/rep2 pairs.
          4) Build a coverage matrix, compute correlation R², and produce:
             - correlation_heatmap.png
             - coverage_matrix.csv
             - correlation_matrix.csv
             - merged pair .gff files
             - optionally averaged rep .gff files (excluded from heatmap)
        """
        try:
            # Group into R1/R2 by removing trailing _R1/_R2 from stems
            stems = {p: os.path.basename(p)[:-4] for p in self.sam_paths}  # drop ".sam"
            pair_buckets = {}
            for p, stem in stems.items():
                # Remove optional _/- plus R1/R2 to form a pairing key
                pair_key = re.sub(r'[_-]?R[12]$', '', stem)
                pair_buckets.setdefault(pair_key, []).append(p)

            # Create per-pair GFFs (merged R1+R2)
            gff_paths = []
            for key, files in pair_buckets.items():
                # Identify R1 and R2 based on stem suffix
                r1 = [f for f in files if re.search(r'[_-]?R1$', os.path.basename(f)[:-4])]
                r2 = [f for f in files if re.search(r'[_-]?R2$', os.path.basename(f)[:-4])]
                if r1 and r2:
                    # Use the first R1 and first R2 match under this key
                    self._log(f"🔗 Pair: {os.path.basename(r1[0])}  +  {os.path.basename(r2[0])}")
                    gff = sam_pair_to_gff(r1[0], r2[0], outdir, plasmid_size, log_fn=self._log)
                    gff_paths.append(gff)
                else:
                    # If we can't find both R1 and R2, skip this group
                    self._log(f"⚠️ Skipping '{key}': need both R1 and R2.")

            if not gff_paths:
                self._log("❌ No GFFs were produced. Aborting.")
                return

            # Create averaged GFFs for rep1/rep2 pairs (NOT included in heatmap)
            if self.var_make_avg.get():
                by_core = {}
                for g in gff_paths:
                    name = os.path.splitext(os.path.basename(g))[0]
                    # Replace rep1/rep2 with repX to pair them conceptually
                    core = re.sub(r"rep[12]", "repX", name)
                    by_core.setdefault(core, []).append(g)

                for core, files in by_core.items():
                    if len(files) == 2:
                        # Read each GFF and extract pos/cov into DataFrames
                        dfs = []
                        for fpath in files:
                            df = pd.read_csv(
                                fpath, sep="\t", header=None, usecols=[3, 5],
                                names=["pos", "cov"], dtype={"pos": "int64", "cov": "float64"},
                                engine="python",
                            )
                            dfs.append(df)
                        # Concatenate and average coverage per position
                        merged = pd.concat(dfs, ignore_index=True).groupby("pos", as_index=False)["cov"].mean()
                        out_avg = os.path.join(outdir, f"{core}_avg.gff")
                        # Write averaged GFF with same schema as original GFFs
                        with open(out_avg, "w") as f:
                            for _, row in merged.iterrows():
                                pos_i = int(row["pos"]); cov_v = float(row["cov"])
                                f.write(f"NA\tAgilent\t{core}\t{pos_i}\t{pos_i}\t{cov_v}\t.\t.\t.\n")
                        self._log(f"🧮 Wrote averaged GFF: {os.path.basename(out_avg)}")

            # Heatmap (exclude *_avg.gff)
            series_map = {}
            for g in sorted(gff_paths):
                label = label_from_gff_path(g)  # use filename stem as label
                s = load_series(g)
                if not s.empty:
                    series_map[label] = s
                else:
                    self._log(f"(empty series) {os.path.basename(g)}")

            if len(series_map) < 2:
                self._log("❌ Need at least two non-empty datasets to build a heatmap.")
                return

            # Order labels by numeric key (if present) for a nicer heatmap layout
            ordered_labels = [name for _, name in sorted((numeric_key(n) for n in series_map.keys()))]
            wide = build_aligned_matrix({n: series_map[n] for n in ordered_labels})
            corr = correlation_matrix_from_wide(wide)

            # Save outputs
            float_prec = int(self.var_csv_precision.get() or 6)
            heatmap_png = os.path.join(outdir, "correlation_heatmap.png")
            plot_heatmap_tri(
                corr,
                heatmap_png,
                title=self.var_title.get().strip() or "Correlation (R²) Similarity"
            )
            # Coverage matrix: coordinates x samples
            wide.to_csv(os.path.join(outdir, "coverage_matrix.csv"), float_format=f"%.{float_prec}f")
            # Correlation matrix: samples x samples (R²)
            corr.to_csv(os.path.join(outdir, "correlation_matrix.csv"), float_format=f"%.{float_prec}f")

            # Log summary of outputs
            self._log("\n📦 Outputs saved to: " + outdir)
            self._log(" - correlation_heatmap.png")
            self._log(" - coverage_matrix.csv")
            self._log(" - correlation_matrix.csv")
            self._log(" - merged pair .gff files")
            if self.var_make_avg.get():
                self._log(" - averaged rep .gff files (excluded from heatmap)")
        except Exception as e:
            # Catch-all error handler to avoid killing the GUI thread
            self._log("❌ Error: " + str(e))

    # ------------ Utilities ------------
    def _log(self, text: str):
        """
        Append a line of text to the GUI log Text widget in a thread-safe-ish way.
        (All calls should be from the main thread or via Tk's event loop.)
        """
        self.log.configure(state="normal")
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)  # Auto-scroll to the end
        self.log.configure(state="disabled")


def main():
    """
    Entrypoint: create and run the Tkinter application.
    """
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
