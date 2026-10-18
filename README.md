# TraceRCA

Implementation of **Separating Physical Propagation Delay from Observation Latency for Root-Cause Localization**.

## Installation

Use Python 3.11+. Create and activate a virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

In PowerShell, replace the activation line with `.venv\Scripts\Activate.ps1`.

## Data and preprocessing

Use [SWaT/WADI](https://itrust.sutd.edu.sg/itrust-labs_datasets/) or documented Tennessee Eastman/HAI exports, with independently supplied topology/timing references.

For process samples, create an NPZ with `x` of shape `samples × sources`, `time` in seconds, `episode` identifiers, and `split` strings `train` or `test`. Use episode-relative nonnegative time and split whole fault episodes. An optional boolean `normal` array restricts the training baseline to verified normal samples; otherwise all training samples define channel means and standard deviations. The extraction command emits a threshold-crossing event when an absolute standardized residual first exceeds the chosen threshold:

For existing alarm logs, provide `events.json` directly: a list of episode objects with `id`, `split`, `horizon`, and `events`. Each event has integer `node`, integer `source`, and observed `time`. Events after the diagnosis horizon are ignored. Optional evaluation references are `root` (node index), `path` (edge-index list), `edge_delay` (mapping from edge index to reference physical mean delay), and `injection_time`. Extraction does not invent these annotations; add real references before scoring them.

The supplied `graph.json` illustrates a three-node engineering graph. Replace nodes, typed edges, source reliabilities, and timing priors with the real system. Each kernel type has fixed positive gamma `shape` and an initial `mean` in seconds. Every source has latency prior, prior standard deviation, and admissible latency bounds. Source zero is the fixed clock reference. Distinct physical edge delays and source latencies require informative source priors or independent timing evidence; topology alone does not identify the decomposition.

```bash
tracerca extract --source process.npz --output events.json --threshold 3
```

## Training and inference

```bash
tracerca fit --data events.json --graph graph.json --output model.json --epochs 200
```

## Evaluation

```bash
tracerca evaluate --data events.json --model model.json --output outputs/full --seed 0
```

## Ablations

Supported controls: `raw`, `no_missing`, `no_reliability`. Reuse the same inputs, seed and budget. For retrained controls, evaluate the corresponding saved model.

```bash
tracerca evaluate --data events.json --model model.json --output outputs/raw --seed 0 --variant raw
```

## Outputs

`events.json` contains extracted events; `model.json` stores fitted timing parameters. Evaluation writes `episodes.json`, `metrics.json` and `likelihood.png` in the selected folder.

Source inspection: Python 3.13.5, numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, matplotlib 3.10.0 (installed versions inspected). Checks covered Python syntax, configuration parsing and CLI arguments. No training, full evaluation or CI was run in this update.
