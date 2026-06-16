# UniRef50 2k Pilot Analysis

This note summarizes the first broader-pool MVP evaluation on the UniRef50-derived lysozyme-like benchmark.

It should be read alongside:

- [`lysozyme_uniref50_pool.md`](/Users/chloe/Desktop/project/protein/Steering-PLMs/docs/lysozyme_uniref50_pool.md)
- [`lysozyme_mvp_v4_analysis.md`](/Users/chloe/Desktop/project/protein/Steering-PLMs/docs/lysozyme_mvp_v4_analysis.md)

## Scope

This is a pilot, not the final large-scale benchmark.

- The full UniRef50 fetch produced about `18k` representative sequences.
- A full scored preprocess on the whole pool is a separate long-running job.
- For practical local runtime, experiments here use a scored `2k` sample benchmark:
  - [`preprocess_manifest.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme_uniref50_2k/preprocessed/preprocess_manifest.json)
- The benchmark itself is larger than the earlier reviewed-UniProt family benchmark:
  - pool `1993`
  - train `1594`
  - test `399`
  - `N_sol_test = 120`
  - `N_therm_test = 120`
- The actual eval suite was further downsampled to `n = 30` per run because the full `120 x 4` experiment suite was trending toward a multi-hour local run.

Important comparability caveats:

- `ppl` was skipped in these pilot runs.
- nearest-natural identity was not requested here.
- results are therefore directly comparable to the v4 lysozyme-family runs on fitness and drift, but not on protein-likeness.
- this broader pool is much more heterogeneous than the reviewed-family benchmark, so absolute gains should not be compared naively.

## Runs

- Solubility, random: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_uniref50_2k_sol_random_n30/summary.json)
- Solubility, ASPO: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_uniref50_2k_sol_aspo_n30/summary.json)
- Thermostability, random: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_uniref50_2k_therm_random_n30/summary.json)
- Thermostability, ASPO: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_uniref50_2k_therm_aspo_n30/summary.json)

## Dataset Quality Check

The low-tail construction behaved correctly on this benchmark.

- `N_sol_test` threshold: `0.0359`
- `N_therm_test` threshold: `44.04`
- `requested_size_exceeds_tail = false` for both tasks

So unlike the early small-family runs, the exported test sets are true bottom-tail subsets rather than the entire test split.

The source tails are also much harsher than the reviewed-family benchmark:

- solubility source mean: `0.00215`
- thermostability source mean: `37.54`

That matters for interpretation: this benchmark gives the editor much more room to improve very poor starting points.

## Main Results

### Solubility

Random task:

- `no_steering`: `delta_sol_mean +0.142`, `success_rate 0.80`, `top_10 delta +0.367`
- `naive_steering`: `delta_sol_mean +0.122`, `success_rate 0.933`, `top_10 delta +0.329`
- `alignment_steering`: `delta_sol_mean +0.105`, `success_rate 0.967`, `top_10 delta +0.275`

ASPO task:

- `naive_steering`: `delta_sol_mean +0.289`, `success_rate 1.00`, `top_10 delta +0.699`
- `alignment_steering`: `delta_sol_mean +0.236`, `success_rate 1.00`, `top_10 delta +0.554`

Interpretation:

- On the broader UniRef50 tail, `ASPO` is clearly stronger than random masking for solubility.
- In this pilot, `ASPO + naive_steering` beats `ASPO + alignment_steering` on mean solubility gain.
- That is different from the reviewed-family benchmark, where `ASPO + alignment_steering` was the best final solubility method.
- Within the random family, steering improves per-sequence success rate, but does not improve the mean as much as `no_steering`.

One important caution:

- the source tail is so poor that even large deltas do not usually reach the positive-family regime
- `sol_threshold_cross_rate` only reaches:
  - `0.10` for `random + no_steering`
  - `0.30` for `ASPO + naive_steering`

So this pilot shows strong tail rescue, but not broad conversion into clearly positive examples.

### Thermostability

Random task:

- `no_steering`: `delta_therm_mean +4.89`, `success_rate 0.80`, `top_10 delta +8.62`
- `naive_steering`: `delta_therm_mean +6.98`, `success_rate 0.867`, `top_10 delta +14.37`
- `alignment_steering`: `delta_therm_mean +7.48`, `success_rate 0.967`, `top_10 delta +14.08`

ASPO task:

- `naive_steering`: `delta_therm_mean +10.12`, `success_rate 0.933`, `top_10 delta +18.56`
- `alignment_steering`: `delta_therm_mean +8.48`, `success_rate 0.967`, `top_10 delta +15.45`

Interpretation:

- Thermostability is much easier to improve on this broader benchmark than on the reviewed-family benchmark.
- `ASPO + naive_steering` is the strongest therm method by mean gain in this pilot.
- `alignment_steering` is still the most consistent therm variant by success rate, but it is no longer the clear best method by mean gain.
- `random + alignment_steering` is already very strong here, which suggests the broader low-tail benchmark is more forgiving than the narrow family benchmark.

Note:

- the threshold-cross field in the current therm summaries is still stored under the JSON key `sol_threshold_cross_rate`
- semantically it is still the positive-threshold crossing rate for the active property
- the observed therm threshold-cross rates in this pilot are still modest at about `0.13 - 0.20`

## Edit Size And Drift

The broader benchmark makes the current edit regime look even more global.

Across the four pilot runs:

- `hamming_mean` ranges from about `129` to `178`
- `percent_identity_mean` ranges from about `0.13` to `0.32`
- `representation_l2_mean` ranges from about `1.49` to `3.21`
- `representation_cosine_mean` falls to about `0.94 - 0.98`

The locality metrics point in the same direction:

- `mutation_position_entropy_mean` is about `0.99` in every run
- `global_mutation_position_entropy` is essentially `1.0`
- `pairwise_hamming_mean` is about `0.91 - 0.94`
- `pairwise_edit_similarity_mean` is only about `0.21`

Interpretation:

- mutations are spread broadly across the sequence rather than concentrated in a local patch
- different edited outputs remain diverse relative to one another
- the methods are not collapsing to one identical edited sequence
- but they are also not behaving like sparse local editing

So the right mental model for the current pipeline on this benchmark is still iterative sequence rewriting, not conservative mutation editing.

## Round Dynamics

The broad-pool pilot behaves differently from the reviewed-family benchmark in one useful way.

- Most best rounds occur at the end of the schedule:
  - sol random: round `10` for all three modes
  - sol ASPO: round `10` for both modes
  - therm random: round `10`, `9`, `10`
  - therm ASPO: round `10` for both modes

That contrasts with the earlier narrow-family runs, where several methods peaked very early and then degraded.

Interpretation:

- on the broader and harsher tail, additional refinement rounds continue to help more often
- early stopping still matters, but it looks less urgent here than it did on the narrow family benchmark

## Cross-Benchmark Takeaways

### 1. Broader low-tail benchmarks are easier to improve

Compared with the reviewed-family benchmark:

- starting fitness is much lower
- mean gains are much larger
- both random and ASPO become stronger, especially for thermostability

This suggests the earlier reviewed-family benchmark was a much tighter and harder within-family editing problem.

### 2. Alignment steering is not universally best

On the reviewed-family benchmark:

- `ASPO + alignment_steering` was the best final solubility method
- `ASPO + alignment_steering` was also one of the strongest therm methods

On the broader UniRef50 pilot:

- `ASPO + naive_steering` is best on both solubility and thermostability by mean gain
- `alignment_steering` helps consistency more than peak mean gain

So the ranking between naive and alignment steering is benchmark-dependent.

### 3. Better fitness does not imply local editing

The new pilot strengthens the main caution from the earlier runs:

- property gains can be large
- success rates can be high
- but edit size and representation drift remain very large

So improvements should still be interpreted as strong generative refinement, not yet as precise protein editing.

## Recommendations

The next highest-value experiments are:

1. Scale this exact UniRef50 benchmark from `n = 30` to `n = 60` before committing to the full `n = 120` suite.
2. Add nearest-natural identity on the broader pool so the strong fitness gains can be evaluated against sequence realism.
3. Compare the current full-coverage multi-round schedule against a genuinely sparse schedule, because the current winner methods are still operating in a global-rewrite regime.
