# Rao (2007) decision-matrix corpus

Twenty-eight decision matrices from R. V. Rao, *Decision Making in the
Manufacturing Environment* (Springer, 2007), used in Section 6 of the main
paper and Supplement S5. Each matrix is a CSV in this folder, one per
published table, named `rao_table_<chapter>_<table>.csv` (for example
`rao_table_29_1.csv` for Table 29.1, vendor selection).

## Status of this folder

`metadata.json` and `expected_audit.json` are complete for all 28 matrices.
The CSV files themselves must be transcribed from the book (they are
copyrighted tabular data reprinted from the original journal papers, and
were not recoverable from the manuscript). Run

    python scripts/validate_corpus.py

after adding each file. The validator recomputes κ_j, the effective weights,
ν, δ_w (both cost conventions), the flagged pairs and the TOPSIS inversions,
and compares them with the values published in Supplement S5, Table S2.6 and
Table 6 of the main paper, so a transcription error in any cell is caught.
Pass `--update-metadata` to record which matrices have been verified.

## CSV format

    alternative,C1,C2,C3
    __orientation__,1,-1,1
    __weight__,0.5,0.3,0.2
    Alternative 1,0.020,490000,3.1
    Alternative 2,0.005,600000,4.5

* Row 1: blank or `alternative`, then the criterion abbreviations exactly as
  listed under `criteria` in `metadata.json` (they are the abbreviations used
  in Supplement S5).
* `__orientation__`: `1` for a benefit (larger is better), `-1` for a cost.
  `+`/`-` and `benefit`/`cost` are also accepted.
* `__weight__`: the stated weights as printed in the source. They need not
  sum to one; the loader renormalises them (Rao's weights for Table 10.2 sum
  to 1.046).
* One row per alternative, using the alternative names as they appear in the
  flagged-pair lists of `expected_audit.json` (e.g. `Vendor 1`, `Design 2`,
  `P1`, `FMS 3`).
* Enter the columns **as published**, without transforming cost criteria; the
  audit computes κ_j on the column as it will be vector-normalised.
* Tables 14.1, 24.2, 25.1 and 26.1 are printed attribute-by-alternative in the
  book and must be transposed. Table 24.2 has twelve criteria of which
  *Integration* is constant across the three alternatives; include it or not,
  the audit drops it either way and S5 reports the eleven retained criteria.
* Converted linguistic ratings (type M and C in Table 6) use Rao's 0–1
  conversion scale (his Table 4.3); enter the converted values.

`TEMPLATE.csv` is a small illustrative file in the right format (not one of
the 28 matrices) and is ignored by `run_audit.py` and `validate_corpus.py`.

## metadata.json

For every matrix: Rao table number, problem, original source, provenance of
the weights (Rao's AHP judgements, the original paper's weights, or equal
weights), data type (raw / mixed / converted / priorities) with the list of
converted columns, transcription notes, dimensions, criterion abbreviations,
orientations, renormalised stated weights, and the CSV filename.

## expected_audit.json

For every matrix: ν and δ_w under the untransformed convention; ν and δ_w
under the max−x convention (Table S2.6); per criterion the stated weight,
κ_j, effective weight and shift in percentage points; the flagged pairs with
a flag for those classical TOPSIS inverts; and the TOPSIS inversions that
the audit does not flag (pure aggregation effects); plus the Table 6 counts.
