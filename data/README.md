# WMDP Data

Latest unlearning corpora are in `wmdp_deduped_bio` and `wmdp_deduped_cyber` directories.

## File naming convention

- `*_corpus_simple.jsonl` - Latest version of the corpus files (use these)
- `*_corpus.jsonl` - Older corpus format

## Splits
Following Deeb & Roger, we unlearn on the T and V splits, then retrain on T, and evaluate always on V.

- `*T_*` - First split (80%)
- `*V_*` - Second split (20%)
- `dev_*` - Development splits
- `all_*` - Concatenation of dev and main splits

## Recommended usage

Use `all_T_corpus_simple.jsonl` and `all_V_corpus_simple.jsonl` for the full dataset:
- `all_T_corpus_simple.jsonl` = `dev_T_corpus_simple.jsonl` + `T_corpus_simple.jsonl`
- `all_V_corpus_simple.jsonl` = `dev_V_corpus_simple.jsonl` + `V_corpus_simple.jsonl`
