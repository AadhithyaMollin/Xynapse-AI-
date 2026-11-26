# Tsuboyama Mega-scale stability data card

Predictions are computational hypotheses, not experimental findings.

## Source and retrieval

- Record: Zenodo record `7992926`, title recorded in the fetch manifest as “Mega-scale experimental analysis of protein folding stability in biology and design”.
- Retrieval date: 2026-11-26 UTC, from the saved manifests.
- Archive member used: `Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv`; uncompressed member size 1,284,629,306 bytes; compressed member size 446,873,715 bytes; SHA-256 `701647b9cc330c2cb5cbd5c1f003e39147cffdf2c330af7c278b1c97d73e0c8a`.
- Archive member used: `Processed_K50_dG_datasets/Single_DMS_list.csv`; uncompressed size 425,270 bytes; compressed member size 197,283 bytes; SHA-256 `8254a5a3ffbacdaac47660ca7a7e24ce155f92b2274a616c00803dab8cb25079`.
- Both files were fetched by range-extracting members from the archive. The 1,013,678,473-byte archive itself and the other dataset members were not downloaded as whole files.
- Retrieval URLs are recorded in each `.manifest.json` alongside the data; Zenodo API source: `https://zenodo.org/api/records/7992926`.
- Manifests: `data/raw/tsuboyama/Tsuboyama2023_Dataset1_20230416.csv.manifest.json` and `data/raw/tsuboyama/Single_DMS_list.csv.manifest.json`.

## Columns and inspection

The Dataset 1 inspection used only `name`, `deltaG`, `deltaG_95CI`, `deltaG_95CI_high`, and `deltaG_95CI_low` in chunked reads; `dna_seq` was not loaded. The sequence-reference list used `name`, `aa_seq`, and `DMS_group`.

The saved census classified 1,841,285 rows: 1,562 wild-type-like names, 1,025,049 single-substitution names, and 814,674 other names. Examples and full outputs are in `reports/tables/megascale_name_census.csv`. The “other” class is not interpreted as a biological category.

The saved inspection found 983 listed domains, 979 with numbering verified at exact or constant offset, 4 numbering failures, 858 domains with at least one reliable WT row, and 856 domains meeting the reliability, numbering, and minimum-observation rules. The full domain-level results are in `reports/tables/megascale_domain_summary.csv` and `reports/tables/megascale_numbering_check.csv`.

## Sign convention and label definition

The fetched bioRxiv v1 text (`https://www.biorxiv.org/content/10.1101/2022.12.06.519132v1.full`, retrieved 2026-11-26; 817,298 bytes; SHA-256 `7f230b4f7feec3732a241663a6958820839ad443fa44b8b010be5e42c62b5813`) contains these exact statements:

> “All ΔΔG measurements are in the original wild-type background; positive ΔΔG indicates stabilizing substitutions.”

> “ΔΔG indicates the stability of the PROSS design (ΔG) minus the stability of the original wild-type sequence; positive ΔΔG indicates the design stabilized the domain.”

> “black corner slashes indicate lower confidence ΔG estimates, (95% confidence interval > 0.5 kcal/mol), including ΔG estimates near the edges of the dynamic range.”

> “All folding stability data were clipped between from -1 to 5 (kcal/mol) because the folding stability outside the dynamic range is not reliable, and then the average of the stability for 20 amino acids for each site was subtracted from the data.”

Source positive ΔΔG is stabilizing. The project convention is positive = destabilizing, so the project target is `ddg_project = deltaG_wt - deltaG_mut`, where `deltaG_wt` is the median deltaG over reliable WT-reference rows for that domain.

A row is reliable only when `deltaG_95CI <= 0.5` and `-1.0 <= deltaG <= 5.0`; the domain must also have at least one reliable WT-reference row. A candidate domain additionally requires verified numbering and at least 100 reliable single-substitution rows. Position parsing uses the right-most underscore-delimited token. When the exact numbering match is below 99%, a constant offset is checked; the four remaining failures are not selected.

The saved pooled X→P sign check contains 36,712 reliable substitutions across eligible domains and has mean project ddG `+1.40110918 kcal/mol`, consistent with the required project sign sanity direction. Source: `reports/tables/megascale_inspection_summary.txt` and `reports/tables/sign_check_megascale.csv`.

## Proposed training subset

The deterministic seed-42 proposal contains 60 domains sampled from the 856 eligible domains. It is a proposed training set, not a model result. See `reports/tables/proposed_training_ddg_domains.csv`.

## Resolved and unresolved checks

- Resolved: mutation-token parsing is from the right-most `_`; the listed sequence is used as the WT reference; exact or constant-offset numbering must reach at least 99%; WT references and mutant observations use the same reliability rule; project sign sanity check is positive.
- Resolved: the source text exactly contains all four quoted stability/sign/reliability statements above; the full verification artifact is `reports/tables/megascale_quote_verification.txt`.
- Unresolved: “other” row-name classes remain unassigned and are not used to infer meaning.
- No 650M ESM model or model fitting was used in this inspection.
