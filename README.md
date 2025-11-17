Python code for analysis of Footprint-seq data

Input files (.fastq, .sam, .gff) used in Wang et al. (2025) will be provided through EBI ArrayExpress


Description of "convert sam to gff for plasmid Footprint-seq data.py"
- specific to analysis of Footprint-seq data generated using plasmid DNA
- takes .sam files as input; two replicate .sam files per experiment
- outputs a .gff with transposition frequencies (RPM) for every input .sam, and a .gff that averages the .gffs for replicate .sam files
- generates a heatmap of R2 values between all samples analyzed

Description of "convert sam to gff for chromosome Footprint-seq data.py"
- as above but for chromosomal DNA

Description of "unified footprint-seq analysis with scrollbar 11_10_25.py"
- takes .gff files and a .txt with a list of genome coordinates as input
- compares transposition frequencies between control (no protein) and experimental (+ protein) datasets
- outputs a .csv with transposition frequencies (RPM) for 30 bp windows centered on all regions tested
- outputs a .csv with fraction values (1 - Fractional Occupancy) for each region tested
- outputs a .csv with Fractional Occupancy values for each region tested
- outputs a .csv with estimated Kd values for each region tested
- outputs a .png with a heatmap of Fractional Occupancy values and a scatter plot of Kd values
- outputs a .pdf with two pages per region tested; one page shows transposition frequencies and the other shows a graph of Fractional Occupancies vs protein concentration, with a Kd model superimposed
- outputs a .csv for each protein concentration with average differences in transposition frequencies compared to the "no protein" dataset for positions in a 200 bp window around the positions being analyzed 
- outputs a .png for each protein concentration with a line graph showing average differences in transposition frequencies compared to the "no protein" dataset for positions in a 200 bp window around the positions being analyzed 
- outputs a .csv with average values for each position within a 200 bp window around the positions being analyzed; different types of average are selected using the GUI
- outputs a .csv with for each protein concentration with raw differences in transposition frequencies compared to the "no protein" dataset for positions in a 200 bp window around every position being analyzed 
- has options for different types of average calculation

Description of "de novo footprint search 11_10_25.py"
- takes .gff files (2 x control, 2 x experimental) as input
- outputs a .txt listing genome coordinates, p-values for differences in coverage between control and experimental samples, ratio of coverage experimental:control ("inf" for division by zero cases)

Description of "find_local_minima_and_make_graphs_11_10_25.py"
- takes the .txt output file from "de novo footprint search 11_10_25.py" as input
- outputs a .csv that is a subset of the input .txt, listing only local minima (at least 60 bp apart)
- outputs a .png volcano plot (hex-plot) for all positions
- outputs a .png volcano plot (hex-plot) for local minima only
