# Host and Gram-type classification: handover audit

Audit date: 29 September 2026. Source root: `/home/siddiqun/Bureau/Internship/`.

## Conclusion

The attached `Codes/Host Classification/Host_Prediction (data arranging).ipynb` contains host-score preparation logic, but its saved active cells are **not a complete, reliable regeneration pipeline**. Preserve it as historical source alongside the actual inputs and outputs. Do not send that notebook alone or describe it as independently reproducible.

This folder contains the audit, original notebook copies, extracted code for inspection, and file checksums. **The large research datasets are not copied into this folder.** Nothing was sent to anyone, and original Internship files were not modified.

## What to send

Paths below are relative to the Internship root. Keep the same directory structure.

| Purpose | Files to preserve | Notes |
|---|---|---|
| Shared labels and fold assignments | `Data/metadata.csv` | 1,853,074 proteins; 18,474 phages |
| Existing host-analysis input | `Data/phage_host_scores.parquet` | 18,443 phages × 2,036 scores |
| Gram-analysis input | `Data/host_type_all_preds.csv` | 1,853,074 proteins × 40 scores, plus protein ID |
| Gram output actually read by main host notebooks | `Codes/gram_type_predictions.csv` | Preserve this exact version |
| Alternative Gram output | `Codes/Gram Type Classification/gram_type_predictions.csv` | Different predictions; preserve separately, never silently substitute |
| Host-score reconstruction inputs | `Data/all_random_pc_splits.pkl` and all prediction `.pkl` files in `Data/random_pc_r_5_da_2_bal_s_0/` through `_s_4/` | 401 prediction files plus the split file; about 12.63 GB combined |
| Code | Both classification directories and the attached preparation notebook | Original notebook copies are in `original_code/`; retain variants until the analysis used for the professor's results is identified |
| Optional intermediate archive | `Data/processed_chunks/` | Two separate 19-file variants, described below |
| Wider upstream archive | Remaining research files in `Data/` | Includes embeddings, functional predictions and alternative prediction runs; preserve for future upstream work, although not direct inputs of the selected final analysis path |

The four primary analysis inputs total 466.36 MB (decimal). The original Gram notebooks also load the 9.00 GB split pickle, so include it when preserving the notebooks **as written**. Do not omit it on the assumption that its later use is unnecessary without explicitly editing and validating a new notebook.

The manifest covers 700 selected files, totaling 24.21 GB: the research Data files inventoried, 40 notebooks, and the two Gram exports. It is **not a complete filesystem backup** and excludes personal documents, environments and temporary lock files. Reports, external library code, original upstream training sources and any external datasets require separate preservation if they form part of the final research archive.

`file_manifest.csv` gives each path, role, byte count and SHA-256 checksum. The role “other_data_not_referenced_by_selected_pipeline” means not found as a direct input of this selected path; it does not mean safe to delete. `inventory_totals.json` gives totals by role.

## Code and dependency map

### Host analysis

`Codes/Host Classification/Host_Classification_Final_cleaned.ipynb`, `Host_Classification_Final_mq.ipynb`, `Host_Classification_Final_nodim.ipynb`, and `Cleaned_Host.ipynb` read metadata, the saved host-score parquet and the Gram export directly under `Codes/`. Multiple “final” variants exist; filenames alone do not establish which produced the reported results.

The root `host_classification_library.ipynb` also uses these data, but points to an absolute library path under `Internship/Conformal Library` that no longer exists. The observed environment has an editable installation recorded at that same missing location. Preserve the actual library source revision used for results, and repair the path in a clearly marked working copy. The current library at `/home/siddiqun/Bureau/Conformal Library` must not automatically be treated as the historical version.

### Gram analysis

`Codes/Gram Type Classification/Gram_Classification_Final.ipynb` reads metadata, protein-level Gram scores and the split pickle, then aggregates scores by phage. Its ordinary split setup trains on folds 1–4 and tests on fold 0.

`Gram_Classification_Final (split 0 only).ipynb` later replaces this with a fold-0-only 40% / 30% / 30% division using seed 42. The report describes the fold-0 approach. These are different experiments; preserve both and identify the intended one in the handover.

The export uses `to_csv("gram_type_predictions.csv")`, whose destination depends on the notebook's current working directory. Running it in the Gram notebook directory will not update the copy directly under `Codes/` that host notebooks consume.

## Findings requiring attention

### 1. Two different Gram exports

Both exports contain the same 18,474 phage IDs, but **1,407 predictions differ**. The Codes-root copy assigns exactly one Gram class to every phage. The Gram-directory copy assigns both classes to 1,407 phages. This can change the host candidate filtering and therefore results.

See `gram_export_disagreements.csv`. The existence of these files does not by itself establish which notebook/run generated each. Record the chosen version and its checksum explicitly.

### 2. Preparation notebook is incomplete as saved

Cell numbers here are zero-based, matching the comments in `code_text/`.

- Its `../Data` paths resolve to `Codes/Data` if run from its saved `Codes/Host Classification` directory; the actual Data folder is two levels above. Use an explicit project root in a repaired copy.
- The earlier chunk-generation cell 19 is commented out.
- Active cells 22–24 select source-fold predictions using per-feature split assignments and accumulate per-phage sums and nonmissing counts.
- There is no active division and save connecting those accumulators to the final parquet. Cell 25 instead reads an already-existing parquet.
- Cell 27 references `pkl_files`, which is not defined by the active earlier code. Saved outputs therefore do not establish a successful clean run.
- Missing files and read failures can be skipped, and columns without split assignments can be skipped. A reproducible version needs an explicit missing-input policy and a log of every omission.

The missing numerical step is conceptually `phage_scores = sum_df.div(count_df.where(count_df > 0))`, followed by an explicit export. This alone does **not** establish identical historical output: index/column ordering, missing-data policy and the selection of 18,443 rather than 18,474 phages must first be resolved and checked against the preserved file.

Across source folds, `all_preds_gramneg_DNA-associated_1.pkl` exists in fold 0 but is absent from folds 1–4. This is a filename-level finding; it does not prove the corresponding predictions cannot exist under another file arrangement.

### 3. Two intermediate chunk families

`master_chunk_*` and `uncontaminated_chunk_*` each comprise 19 files covering 1,853,074 protein rows. Combining both families would double-count proteins. Treat them as alternative intermediate versions.

For five evenly spaced score columns, averaging each family by phage matched the saved parquet across all 18,443 common phages, with matching NaN masks. Maximum absolute differences were about 4.42e-8 for master and 2.22e-16 for uncontaminated. This supports the aggregation lineage **for those five columns only**; it is not a full 2,036-column or upstream reconstruction verification. See `chunk_comparison.json`.

### 4. Dimensions and missingness

Metadata has unique protein IDs and no conflicting host, host type or split labels within a phage. Protein IDs stripped of their final underscore component agree with metadata accession for every row checked.

The saved host table has 18,443 unique phages, all present in metadata; 31 metadata phages are absent. About 62.70% of its score cells are missing, and no score column is entirely missing. The retained score range is approximately 1e-7 to 1.0. These checks do not determine why the 31 phages were omitted; that filtering step remains to be documented.

The report refers to 2,160 models while the saved table contains 2,036 score columns. These counts should not be presented as interchangeable without explaining the difference.

### 5. Fold/provenance interpretation needs confirmation

The report describes contamination handling in terms of model/protein-cluster folds and the phage fold. The active preparation code selects source folder `k` when a feature's split assignment equals `k`; it does not explicitly compare against the metadata phage fold in that loop. Clarify the intended independence guarantee before claiming the implementation proves the report's contamination statement.

The inspected analysis notebooks do not establish the full training history of the upstream protein-level predictors, or the provenance of every embeddings/functional-prediction artifact. These should be requested/preserved separately if the professor needs reproduction from raw biological data rather than reuse of saved model scores.

## Verification scope and future completion

Completed: file inventory and checksums; static notebook dependency review; metadata and saved-table consistency checks; Gram export comparison; prediction-filename coverage across five folds; sample aggregation comparisons for both chunk variants; unchanged notebook snapshots.

Not completed: a clean end-to-end notebook run, loading/rebuilding from the 9 GB split pickle, regenerating the full host table from all fold prediction files, training upstream models, or reproducing reported metrics. The split pickle can expand substantially in memory, so the full rebuild was not attempted with the available memory. Do not describe this audit as a certified reproduction.

Before certifying the final handover: identify the host/Gram notebook versions corresponding to the reported results; choose and record the Gram export; implement a separate repaired preparation copy; explain the 31 omitted phages and the score-column count; then run it from a fresh kernel and compare the full output with the preserved parquet (IDs, columns, NaN masks and numeric values). Record seeds, folds, package versions, exact library source and output checksums.

`environment_observed.txt` records the currently installed environment, not necessarily the environment used for historical results and not a tested portable requirements file. It includes many unrelated packages and a broken editable library path.

Audit scripts are included for inspection; they contain local absolute paths and may need adjustment on another computer. Extracted `.py` files in `code_text/` are reading aids, not validated executable replacements for notebooks. `original_code/` preserves original notebook bytes and outputs.
