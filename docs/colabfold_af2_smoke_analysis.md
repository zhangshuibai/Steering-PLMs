# ColabFold AF2 Smoke Analysis

This note summarizes the copied smoke-run outputs under [`results/colabfold_af2_smoke`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/colabfold_af2_smoke).

Primary files:

- run summary: [`summary.json`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/colabfold_af2_smoke/sol_random/summary.json)
- per-sequence outputs: [`no_steering/per_sequence_results.csv`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/colabfold_af2_smoke/sol_random/no_steering/per_sequence_results.csv)
- per-sequence outputs: [`naive_steering/per_sequence_results.csv`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/colabfold_af2_smoke/sol_random/naive_steering/per_sequence_results.csv)
- per-sequence outputs: [`alignment_steering/per_sequence_results.csv`](/Users/chloe/Desktop/project/protein/Steering-PLMs/results/colabfold_af2_smoke/sol_random/alignment_steering/per_sequence_results.csv)

## What This Run Is

This is a very small solubility smoke test:

- property: `sol`
- masking strategy: `random`
- modes: `no_steering`, `naive_steering`, `alignment_steering`
- sample count: `n = 2`
- model: `ESM2-650M`
- rounds: `10`
- pLDDT backend: `colabfold`
- pPPL: skipped

The run is best treated as a pipeline sanity check, not a stable model ranking, because every reported mean is based on only two sequences.

## Main Results

| Mode | Final delta sol mean | Success rate | Hamming mean | Mutation fraction mean | Best round | Best-round delta sol mean |
|------|----------------------:|-------------:|-------------:|-----------------------:|-----------:|--------------------------:|
| `no_steering` | `+0.0090` | `0.50` | `143.5` | `0.8213` | `6` | `+0.0409` |
| `naive_steering` | `+0.2283` | `1.00` | `140.0` | `0.7982` | `10` | `+0.2283` |
| `alignment_steering` | `+0.0285` | `1.00` | `140.5` | `0.8055` | `7` | `+0.2516` |

Takeaways:

- `naive_steering` is the best final mode in this smoke run by a large margin. It is the only mode that improves both sequences substantially at the final round.
- `alignment_steering` has the most interesting trajectory. It peaks much earlier at round `7` with `delta_sol_mean = +0.2516`, then collapses to `+0.0285` by round `10`.
- `no_steering` behaves like a weak baseline. It gets a modest temporary gain by round `6`, then gives most of it back by the end.

The round trajectories make early stopping the clearest lesson from this run. If the pipeline had stopped at the best round for each mode instead of always taking round `10`, `alignment_steering` would look much stronger.

## Sequence-Level Read

Because `n = 2`, it helps to inspect the two rows directly.

For source sequence `0`:

- `naive_steering` is the clear winner: `sol 0.2443` from `source_sol 0.00027`
- `alignment_steering` helps, but much less: `sol 0.0557`
- `no_steering` gives only a small gain: `sol 0.0185`

For source sequence `1`:

- `naive_steering` again gives the only meaningful jump: `sol 0.2131` from `source_sol 0.00053`
- `alignment_steering` is technically a success, but only barely: `sol 0.00210`
- `no_steering` slightly hurts this sequence relative to source

That sequence-level view is important because the headline `success_rate = 1.0` for `alignment_steering` sounds stronger than it really is here. With only two sequences, one tiny positive delta is enough to count as a success.

## Drift And Naturalness

All three modes are extremely aggressive in this setup:

- mean Hamming distance is about `140`
- mean mutation fraction is about `0.80`
- nearest-natural identity stays around `0.286-0.295`

So even the "best" method in this smoke run is not acting like a conservative editor. It is rewriting most of the sequence. That matters when reading both the solubility gains and the pLDDT drops.

## pLDDT Results

| Mode | Source mean pLDDT mean | Edited mean pLDDT mean | Delta mean pLDDT mean |
|------|------------------------:|-----------------------:|----------------------:|
| `no_steering` | `67.81` | `36.99` | `-30.82` |
| `naive_steering` | `67.81` | `37.74` | `-30.08` |
| `alignment_steering` | `67.81` | `39.35` | `-28.47` |

How to read this:

- `source_mean_plddt` is the average confidence for the original input sequence after folding it with the selected backend.
- `mean_plddt` is the same score for the edited sequence.
- `delta_mean_plddt` is edited minus source, so negative is worse.

In this repo's ColabFold path, `mean_plddt` is extracted from the chosen prediction PDB and averaged over `CA`-atom B-factors when those are available. So these are structure-confidence summaries, not classifier outputs.

Rough interpretation:

- `67.8` for the sources is only moderate confidence to begin with
- `37-39` for the edited sequences is very low confidence
- the `-28` to `-31` deltas are large enough that the edited sequences should be treated as structurally dubious in this smoke run

The important point is that pLDDT is not measuring the target property directly. A sequence can get a much better predicted solubility score while simultaneously getting a much worse structural-confidence score. That is exactly what happens here for `naive_steering`: it wins on final `sol`, but its edited sequences still fall into a very low-confidence pLDDT regime.

Among the three modes, `alignment_steering` has the least bad final pLDDT mean, but that does not make it the best overall method here because its final solubility gain largely disappears by round `10`.

## Caveats About The Copied Artifacts

The copied repo contains the saved summaries and per-sequence CSVs, which is enough to analyze the run. It does not currently include the ColabFold cache/output paths referenced by the summary:

- `results/shared_plddt_cache.csv`
- `results/colabfold_af2_smoke_outputs/sol_random`

So this note is based on the recorded metrics in the saved result files plus the implementation in [`mvp_eval_pipeline.py`](/Users/chloe/Desktop/project/protein/Steering-PLMs/mvp_eval_pipeline.py), rather than on re-reading the original PDB artifacts.

## Bottom Line

This smoke test says the ColabFold-backed pLDDT integration is working and exposing a real tradeoff:

- `naive_steering` gives the best final solubility improvement
- `alignment_steering` looks best only with early stopping
- all modes cause very large sequence drift
- all modes drive pLDDT sharply downward, so the edits are not yet structurally convincing

That makes this run a useful validation of the metric plumbing, and a caution that property gains alone are not enough to judge edit quality.
