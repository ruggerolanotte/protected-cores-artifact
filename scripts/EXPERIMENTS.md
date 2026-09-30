# Reproducing the incremental governor experiments

The artifact accompanies *Protected Cores Are Not Enough: Certifying
AI-Proposed Revisions of Temporal Specifications*. The paper describes
the formal model and experimental protocols; this document explains how
to run the code and interpret its outputs.

## Requirements and execution

Requires Python 3.10+, NumPy, SciPy, Matplotlib, and scikit-learn.
Run commands from the artifact root.

```bash
python scripts/fac_experiments.py test --out validation
python scripts/fac_experiments.py all --out results
python scripts/fac_experiments.py all --quick --out smoke
```

- `test`: semantic and protocol checks.
- `all`: experiments with manuscript defaults.
- `all --quick`: a small smoke test; its outputs are not the manuscript results.

Individual experiments:

```bash
python scripts/fac_experiments.py rq1 --out results/rq1
python scripts/fac_experiments.py rq2 --workers 4 --out results/rq2
python scripts/fac_experiments.py rq3 --out results/rq3
python scripts/fac_experiments.py case --out results/case
```

To rerun only RQ2's protected-success sweep:

```bash
python scripts/fac_experiments.py rq2 --gradient-only --workers 4 --out results/rq2
```

Parallel workers affect execution time, not seeds or replicate ordering.
Use a separate output directory for each configuration.

## Code organization

All experiments and checks are implemented in `scripts/fac_experiments.py`.

| Component | Responsibility |
| --- | --- |
| `Specification` | Immutable protected/adaptive components and parameter tuple |
| `revision_rejections` | Structural admissibility checks |
| `IncrementalMonitor` | Event-by-event evaluation and obligation queues |
| `Governor` | Candidate selection, certification, and scheduled activation |
| `CertificationEvidence` | Aggregate/core counts and Hoeffding lower bounds |

The implemented fragment uses Boolean components over `A,C,B,D` and bounded
response windows `[a,b]`. Nested temporal operators inside components are
unsupported. Structural equality is syntactic, not Boolean equivalence.

RQ1 uses a prefix-based selector. RQ2, RQ3, and the case study use the frozen
random-forest proposer described in the paper. Drift declarations are
exogenous in RQ2 and the case study; RQ3 uses the two-window detector.

## Outputs and diagnostic fields

RQ1, RQ2, and RQ3 write manuscript outputs and per-run CSV records.
Additional files include:

| File | Contents |
| --- | --- |
| `gradient_runs.csv` | RQ2 protected-success sweep |
| `proposer_audit.csv` | RQ2 independent proposer audit |
| `candidate_ranking.csv` | Case-study structural verdicts and predicted candidate scores |
| `protocol_tests.json` | Semantic and protocol-check results |
| `runtime_environment.json` | Runtime environment, measurement settings, and timing observations |

In RQ3:

- `oracle_composition_unsupported` compares averages of generating
  probabilities for realised sample origins/groups with the tested
  thresholds. It is a descriptive diagnostic, not the lifetime theorem's
  failure event.
- `current_regime_below_core` checks the environment's protected
  `B`-response probability at activation, one event after certification.
- `ai_selections` includes selected candidates still pending at the horizon.

The RQ2 proposer audit measures optimal top-choice agreement and true-margin
regret, not correctness of the entire ranking.

## Verification

The `test` target compares the incremental monitor with an independent
direct finite-trace evaluation on 120 random traces. Additional checks
cover structural rejection, overlapping windows, tied timestamps, zero
delay, strict completion, origin-version retention, selection-prefix
exclusion, nonempty core evidence, the single-attempt rule, and next-event
activation.

These are executable regression checks, not a machine-checked proof.

## Reproducing runtime measurements

The case benchmark excludes trace generation, model training, prefix ranking,
and the separate diagnostic pass. It alternates baseline/governed measurement
order across nine repetitions and reports median total times divided by the
event count. These are average processing costs, not tail latencies.

`tracemalloc` records peak additional working allocations with the input
trace preallocated. Pending-obligation peaks are measured from monitor queues.

Fixed seeds reproduce the simulated data and decisions. Timing and memory
measurements depend on the execution environment; consult
`runtime_environment.json` when comparing runs.