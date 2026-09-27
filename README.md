# Protected Cores Artifact

Reproducibility artifact for **Protected Cores Are Not Enough: Certifying
AI-Proposed Revisions of Temporal Specifications**.

The artifact contains the executable experiments and the generated tables and
figures used in the paper. The proposer is intentionally untrusted: the
governor checks structural admissibility and certifies aggregate and
protected-trigger outcomes using fresh post-selection evidence.

## Requirements

- Python 3.10 or later
- NumPy
- SciPy
- Matplotlib
- scikit-learn

Install the dependencies with:

```bash
python -m pip install -r scripts/requirements.txt
```

## Reproduce the experiments

Run the complete suite from the repository root:

```bash
python scripts/fac_experiments.py all --out results
```

Run a short interface smoke test:

```bash
python scripts/fac_experiments.py all --out smoke --quick
```

Individual targets are available as `rq1`, `rq2`, `rq3`, and `case`:

```bash
python scripts/fac_experiments.py rq2 --out results/rq2
```

See [scripts/EXPERIMENTS.md](scripts/EXPERIMENTS.md) for the full protocol,
default parameters, and interpretation of the reported diagnostics.

## Repository contents

- `scripts/`: the single reproduction program, requirements, and instructions;
- `rq1/`: data-separation and finite-sample certification results;
- `rq2/`: masked-core experiments and proposer audit;
- `rq3/`: sequential governed-evolution experiment;
- `case/`: monitored clinical-alarm case study and runtime measurements.

All experiments use fixed seeds. Runtime measurements are machine-dependent;
obligation counts, decisions, and other seeded outputs are reproducible.

## License

The artifact is released under the MIT License. See [LICENSE](LICENSE).

