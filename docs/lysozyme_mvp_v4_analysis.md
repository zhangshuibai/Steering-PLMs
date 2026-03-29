# Lysozyme MVP V4 Analysis

This note summarizes the full lysozyme reruns after enabling the richer evaluation metrics in [`mvp_eval_pipeline.py`](/Users/chloe/Desktop/project/protein/Steering-PLMs/mvp_eval_pipeline.py).

Compared with the earlier v3 runs, v4 adds:

- nearest-neighbor identity to a natural-sequence DB
- pseudo-perplexity (`ppl`)
- source-vs-edited deltas for those protein-likeness metrics
- representation-drift summaries
- per-round drift / fitness trajectories
- top-k source baselines and top-k deltas

For these reruns:

- `pLDDT` was intentionally skipped
- `ppl_model` was set to `150M` for practical local runtime
- the natural-sequence DB was [`lysozyme_train.csv`](/Users/chloe/Desktop/project/protein/Steering-PLMs/data/lysozyme/preprocessed/lysozyme_train.csv)

Important caveat:

- nearest-neighbor identity here is family-relative naturalness, not broad proteome naturalness, because the DB is the lysozyme train split rather than UniRef or a larger natural protein corpus

## Runs

- Solubility, random: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_sol_random_v4/summary.json)
- Solubility, ASPO: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_sol_aspo_v4/summary.json)
- Thermostability, random: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_therm_random_v4/summary.json)
- Thermostability, ASPO: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/mvp_lysozyme_therm_aspo_v4/summary.json)

## Main Results

### Solubility

Source benchmark mean:

- `source_sol_mean = 0.264`

Random task:

- `no_steering`: `delta_sol_mean +0.380`, `success_rate 1.00`, `hamming_mean 91.8`, `nearest_natural_identity_mean 0.493`, `delta_ppl_mean +0.255`
- `naive_steering`: `delta_sol_mean +0.212`, `success_rate 0.833`, `hamming_mean 95.2`, `nearest_natural_identity_mean 0.462`, `delta_ppl_mean +0.165`
- `alignment_steering`: `delta_sol_mean +0.165`, `success_rate 0.833`, `hamming_mean 94.7`, `nearest_natural_identity_mean 0.464`, `delta_ppl_mean +0.900`

ASPO task:

- `naive_steering`: `delta_sol_mean +0.353`, `success_rate 0.833`, `hamming_mean 108.2`, `nearest_natural_identity_mean 0.395`, `delta_ppl_mean +1.781`
- `alignment_steering`: `delta_sol_mean +0.505`, `success_rate 0.833`, `hamming_mean 104.8`, `nearest_natural_identity_mean 0.433`, `delta_ppl_mean +1.796`

Interpretation:

- The old headline still holds: `ASPO + alignment_steering` is the best final solubility method by mean gain.
- But the richer metrics show a real cost for that win. The ASPO steering runs have much larger representation drift, worse family-nearest identity, and much worse `ppl` than the random runs.
- `random + no_steering` is now the clearest low-penalty solubility baseline. Its final gain is smaller than `ASPO + alignment_steering`, but it preserves higher natural-family similarity and has the smallest `ppl` penalty among the strong solubility improvers.
- `ASPO + naive_steering` has a strong early gain but degrades across rounds. Its best solubility happens at round `1`, not round `10`.

### Thermostability

Source benchmark mean:

- `source_therm_mean = 46.51`

Random task:

- `no_steering`: `delta_therm_mean -0.50`, `success_rate 0.333`, `hamming_mean 101.2`, `nearest_natural_identity_mean 0.448`, `delta_ppl_mean +0.632`
- `naive_steering`: `delta_therm_mean +1.94`, `success_rate 0.500`, `hamming_mean 111.3`, `nearest_natural_identity_mean 0.393`, `delta_ppl_mean +1.099`
- `alignment_steering`: `delta_therm_mean -0.09`, `success_rate 0.500`, `hamming_mean 118.2`, `nearest_natural_identity_mean 0.366`, `delta_ppl_mean +1.938`

ASPO task:

- `naive_steering`: `delta_therm_mean +0.61`, `success_rate 0.500`, `hamming_mean 122.5`, `nearest_natural_identity_mean 0.346`, `delta_ppl_mean +3.037`
- `alignment_steering`: `delta_therm_mean +1.74`, `success_rate 0.667`, `hamming_mean 118.8`, `nearest_natural_identity_mean 0.356`, `delta_ppl_mean +2.450`

Interpretation:

- The old ranking mostly still holds: `random + naive_steering` gives the highest final thermostability mean gain, while `ASPO + alignment_steering` gives the best success rate with a nearly comparable mean gain.
- The richer metrics sharpen the tradeoff: `random + naive_steering` is much less damaging to `ppl` and family-nearest identity than `ASPO + alignment_steering`.
- `ASPO + naive_steering` is now clearly high-variance and high-cost. Its `top_1` gain is huge (`+5.29`) but its mean gain is much smaller, its best round is `2`, and it has the worst `ppl` penalty in the whole set.
- `random + no_steering` again shows why final-only reporting can be misleading. It has a positive intermediate best round (`round 6`, `delta +1.11`) but drifts into a negative final result by round `10`.

## Cross-Task Takeaways

### 1. Best final fitness and best edit quality are not the same thing

The highest-fitness methods are usually not the most conservative ones:

- solubility winner by final mean: `ASPO + alignment_steering`
- thermostability winner by final mean: `random + naive_steering`

But both of the ASPO steering runs show much larger penalties in:

- `nearest_natural_identity`
- `ppl`
- representation drift
- Hamming distance

So the richer metrics make it clear that the current pipeline is still closer to global refinement than sparse local editing.

### 2. ASPO is strong but expensive

ASPO can improve the target property strongly, especially for solubility, but it tends to:

- move farther from the family pool
- raise `ppl` more
- increase representation drift more

That is most visible in:

- `sol_aspo_v4 alignment_steering`
- `therm_aspo_v4 alignment_steering`

### 3. Early stopping matters

Several methods peak before the last round:

- `sol_aspo_v4 naive_steering`: best at round `1`
- `therm_random_v4 no_steering`: best at round `6`
- `therm_aspo_v4 naive_steering`: best at round `2`
- `therm_aspo_v4 alignment_steering`: best at round `6`

This is one of the main benefits of the new round metrics. The current fixed `10`-round schedule is not always the best stopping rule.

## Practical Recommendations

### If the goal is strongest final solubility

Prefer:

- `ASPO + alignment_steering`

But interpret it as a high-drift regime, not a conservative editing regime.

### If the goal is solubility with lower naturalness penalty

Prefer:

- `random + no_steering`

It gives a smaller gain than `ASPO + alignment_steering`, but a better protein-likeness profile under the current metrics.

### If the goal is strongest final thermostability

Prefer:

- `random + naive_steering`

It has the best final mean gain and a noticeably better `ppl` / nearest-identity profile than `ASPO + alignment_steering`.

### If the goal is thermostability with higher consistency

Prefer:

- `ASPO + alignment_steering`

It has the best success rate and a strong mean gain, but the protein-likeness penalty is larger.

## Engineering Status

No new engineering blockers showed up in the full reruns:

- all four v4 experiments completed locally
- richer metrics were written successfully into the run summaries and per-sequence CSVs
- `ppl` worked with the `150M` model
- `pLDDT` remained disabled as intended

## Next Steps

The current results suggest three high-value follow-ups:

1. Add an early-stopping comparison using the per-round metrics instead of always taking round `10`.
2. Repeat the richer runs with a larger natural sequence DB so nearest-neighbor identity is not limited to the lysozyme train split.
3. Compare the current full-coverage schedule against a sparse-edit schedule, because most top-performing methods are still winning with very large sequence drift.
