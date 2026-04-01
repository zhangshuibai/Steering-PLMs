# UniRef50 Lysozyme-Like Pool

This note records the larger UniRef50 lysozyme-like dataset path for the preprocessing pipeline.

## Fetch

Fetcher:

- [`fetch_uniref_family_pool.py`](/Users/chloe/Desktop/project/protein/Steering-PLMs/fetch_uniref_family_pool.py)

Default query:

```text
identity:0.5 AND lysozyme AND length:[60 TO 400]
```

Outputs:

- [`lysozyme_uniref50_family_input.csv`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme_uniref50/lysozyme_uniref50_family_input.csv)
- [`lysozyme_uniref50_family_input.metadata.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme_uniref50/lysozyme_uniref50_family_input.metadata.json)

Current fetched size:

- raw rows: `18,014`
- canonical + deduplicated estimate before scoring: `17,931`

## Preprocess Config

Config:

- [`lysozyme_uniref50_preprocess.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/configs/lysozyme_uniref50_preprocess.json)

Key settings:

- `train_fraction = 0.8`
- `negative_fraction = 0.3`
- `test_target_size = 1000`
- `device = mps`
- `ppl` is not involved in preprocessing

Estimated split sizes from the current fetched input:

- post-filter pool: `17,931`
- estimated test split: `3,586`
- expected bottom-tail export per property: `min(1000, ceil(3586 * 0.3)) = 1000`

So this setup is large enough to target an `N_sol_test` around `1000`.

## Smoke Test

Smoke config:

- [`lysozyme_uniref50_preprocess.smoke.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/configs/lysozyme_uniref50_preprocess.smoke.json)

Smoke manifest:

- [`preprocess_manifest.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme_uniref50/smoke_preprocessed/preprocess_manifest.json)

The smoke run used the first `256` fetched sequences and completed successfully with scoring enabled.

## Notes

- This pool is much broader than the earlier reviewed-UniProt lysozyme family pool.
- Because the query is broad, nearest-neighbor identity or downstream family-homogeneity checks may matter more in later evaluation.
- The full `17.9k`-sequence scoring run is expected to be much longer than the smoke run and should be treated as a separate long-running preprocessing job.

## Pilot Eval

A practical local pilot was run on a scored `2k` sample benchmark rather than the full `17.9k` pool.

- preprocess config: [`lysozyme_uniref50_2k_preprocess.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/configs/lysozyme_uniref50_2k_preprocess.json)
- preprocess manifest: [`preprocess_manifest.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme_uniref50_2k/preprocessed/preprocess_manifest.json)

The reason for this downscaling was runtime:

- the full `120 x 4` eval suite on the `2k` benchmark was already trending toward a multi-hour local run
- the actual completed pilot therefore used `n = 30` per experiment

Analysis of that pilot is recorded here:

- [`lysozyme_uniref50_2k_pilot_analysis.md`](/Users/chloe/Desktop/project/protein/Steering-PLMs/docs/lysozyme_uniref50_2k_pilot_analysis.md)
