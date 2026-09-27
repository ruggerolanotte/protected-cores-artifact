# Reproducing the synthetic experiments

Install Python 3.10+, NumPy, SciPy, Matplotlib, and scikit-learn. From the archive root run:

    python scripts/fac_experiments.py all --out results

For a brief interface smoke test, which does not regenerate manuscript
numbers:

    python scripts/fac_experiments.py all --out smoke --quick

The public targets are:

| key | location | purpose |
|---|---|---|
| `rq1` | Section 8.1 | optional-continuation and finite-sample latency check |
| `rq2` | Section 8.2 | masked-core experiment, with protected-share and protected-success sweeps |
| `rq3` | Section 8.3 | sequential detector--proposer--governor experiment |
| `case` | Section 8.4 | executable clinical-alarm monitor and reference overhead |

The manuscript artifacts were produced with:

    python scripts/fac_experiments.py rq1 --out rq1
    python scripts/fac_experiments.py rq2 --out rq2
    python scripts/fac_experiments.py rq3 --out rq3
    python scripts/fac_experiments.py case --out case

`rq2` writes both masked-core figures and tables. `rq3` uses the manuscript
defaults: 100 histories, horizon 120000, protected share 0.3, and protected
success schedule 0.99, 0.97, 0.80, 0.55. It runs an automatic regression test
for origin-version semantics before simulation.

RQ3 reports two distinct diagnostics. `sample_target_unsupported` refers to
the predictable-mean target controlled by the lifetime theorem.
`current_regime_below_core` records the synthetic environment probability at
activation and is not the theorem's failure event.

RQ2 and RQ3 use the same frozen `RandomForestRegressor`, trained on 4,000
independent synthetic candidate tasks. It predicts aggregate margin from
aggregate prefix statistics and structural indicators; it never receives the
protected-group success frequency. RQ1 uses prefix-based candidate selection
but intentionally has no learned AI proposer because it isolates
optional-continuation validity. The model, training data, and simulation
seeds are fixed by the command-line defaults.

RQ2 additionally writes `proposer_audit.csv` and
`table_rq2_proposer.tex`. The independent audit ranks four deliberately close
admissible candidates and reports exact rankings, ranking errors, and
true-margin regret; these descriptive quantities are not assumptions of the
governor guarantee.

The clinical case study constructs a Boolean timed trace and evaluates every
bounded-response obligation from the trace itself. It also executes a pending
origin-version regression and compares a non-adaptive incumbent monitor with
parallel incumbent and candidate monitors plus group classification,
confidence-bound updates, and the activation gate. It records total runtime,
time per trace event, Python-tracked peak memory, peak pending obligations,
and the incremental parallel-monitor cost. Absolute timings depend on the
machine; obligation counts, success frequencies, and activation decisions are
deterministic under the fixed seed.
