# TraceRCA

Implementation of **Separating Physical Propagation Delay from Observation Latency for Root-Cause Localization**.

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

Use Python 3.11+. No industrial dataset is bundled. Obtain Tennessee Eastman, SWaT, WADI, or HAI from the corresponding dataset providers under their access terms. A process topology and reference root/path annotations must come from the system model or an independently documented annotation protocol.

## Event preparation

For process samples, create an NPZ with `x` of shape `samples × sources`, `time` in seconds, `episode` identifiers, and `split` strings `train` or `test`. Use episode-relative nonnegative time and split whole fault episodes. An optional boolean `normal` array restricts the training baseline to verified normal samples; otherwise all training samples define channel means and standard deviations. The extraction command emits a threshold-crossing event when an absolute standardized residual first exceeds the chosen threshold:

```bash
tracerca extract --source process.npz --output events.json --threshold 3
```

For existing alarm logs, provide `events.json` directly: a list of episode objects with `id`, `split`, `horizon`, and `events`. Each event has integer `node`, integer `source`, and observed `time`. Events after the diagnosis horizon are ignored. Optional evaluation references are `root` (node index), `path` (edge-index list), `edge_delay` (mapping from edge index to reference physical mean delay), and `injection_time`. Extraction does not invent these annotations; add real references before scoring them.

The supplied `graph.json` illustrates a three-node engineering graph. Replace nodes, typed edges, source reliabilities, and timing priors with the real system. Each kernel type has fixed positive gamma `shape` and an initial `mean` in seconds. Every source has latency prior, prior standard deviation, and admissible latency bounds. Source zero is the fixed clock reference. Distinct physical edge delays and source latencies require informative source priors or independent timing evidence; topology alone does not identify the decomposition.

## Fit and diagnose

```bash
tracerca fit --data events.json --graph graph.json --output model.json --epochs 200
tracerca evaluate --data events.json --model model.json --output outputs/full
tracerca evaluate --data events.json --model model.json --output outputs/raw --variant raw
tracerca evaluate --data events.json --model model.json --output outputs/no_missing --variant no_missing
tracerca evaluate --data events.json --model model.json --output outputs/no_reliability --variant no_reliability
```

Fitting minimizes a continuous-time directed gamma-kernel point-process NLL plus Gaussian latency priors and squared local edge corrections. Edge means share a learned type-level component. `--local-penalty` controls local deviation and `--lr` controls Adam. The baseline intensity is learned per node. Latencies are projected to source bounds after each update; the reference source stays fixed.

Diagnosis enumerates root candidates and simple admissible paths up to `--depth`. For each root it searches a finite physical onset-time grid, selects the highest-scoring path to each observed node, and returns their union. Reliability weights timing evidence; hidden intermediate nodes consume unsupported mass and receive `--missing-penalty`. `raw` uses observed times directly; `no_missing` forbids hidden intermediate nodes; `no_reliability` weights all observed evidence equally.

The paper leaves the kernel families and numerical decoder open. This implementation uses fixed-shape gamma kernels, moment-matched gamma approximations for path convolutions, and a bounded path/onset search. Missing mass is a discrete unit per unobserved intermediate node rather than a free continuous time-varying field; the resulting trace exposes every such node. These choices keep root/path inference and missing-middle-event comparisons directly inspectable.

## Timing and missing-source controls

```bash
tracerca evaluate --data events.json --model model.json --output outputs/drop40 --missing 0.4 --seed 0
tracerca evaluate --data events.json --model model.json --output outputs/clock --clock-offset 100
tracerca evaluate --data events.json --model model.json --output outputs/drift --latency-drift 0.5
```

`--missing` removes a seeded fraction of whole sources; `--clock-offset` shifts all observed events and the horizon; `--latency-drift` adds a source-index-scaled logger offset without refitting propagation. Preserve at least one observed source per episode. For logger-swap adaptation, fit a new model with changed priors while retaining the same physical topology and compare fitted means. Missing-middle tests remove only the annotated intermediate events from a copied event file.

`episodes.json` records ranked roots, selected paths, inferred physical times, unsupported nodes, event NLL, and timing. `metrics.json` reports Root Hit@1/3, MRR, path precision/recall/F1, physical-delay MAE, per-event NLL, unsupported mass, decision time relative to injection, and inference latency whenever the corresponding references exist. `likelihood.png` displays per-episode NLL. A horizon denotes the decision time; evaluate several horizons to locate the first successful diagnosis. Group results by missingness and latency interventions, using fault episodes as paired units. No dataset outcomes or reference annotations are fabricated.

## Reproduction logic conventions

Episode start defaults to zero. A global clock-offset control shifts start, horizon and event timestamps together; likelihood exposure and gamma-kernel integrals use that same observation window. Empty training event lists retain integer node/source index tensors and contribute baseline exposure.
