#!/usr/bin/env python3
"""Reproduce the paper's masked-core, optional-continuation, and sequential experiments.

Dependencies: Python 3.10+, numpy, scipy, matplotlib, scikit-learn. No network or data files.
Run: python fac_experiments.py all --out results
     python fac_experiments.py rq2 --out results/rq2 --replicates 200
For an inexpensive smoke run: python fac_experiments.py all --out smoke --quick
The public ``all`` target regenerates only artifacts included in the paper.
"""
from __future__ import annotations
# rq1_simulation.py (embedded standalone implementation)
"""RQ1: adaptive-prefix selection followed by one post-selection certification run.

The four gates see identical selected-candidate outcomes. Requires NumPy,
SciPy, and Matplotlib. Bernoulli fixed-time tests use an exact binomial tail;
the paper's gate uses its stated Hoeffding spending rule.
"""
import argparse
import csv
import math
import time
import tracemalloc
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from sklearn.ensemble import RandomForestRegressor


class AggregateAIProposer:
    """Frozen supervised proposer trained on independent synthetic tasks.

    The model predicts a candidate's future aggregate margin from its prefix
    aggregate frequency, trigger coverage, threshold, and two structural
    indicators.  It deliberately receives no protected-group outcome rate:
    the symbolic governor, not the proposer, is responsible for core safety.
    """

    def __init__(self, seed: int, training_tasks: int = 4000):
        rng = np.random.default_rng(seed)
        features, targets = [], []
        for _ in range(training_tasks):
            w = rng.uniform(0.01, 0.50)
            p_core = rng.uniform(0.45, 1.0)
            p_other = rng.uniform(0.45, 1.0)
            p_adaptive = rng.uniform(0.75, 1.0)
            keep = bool(rng.integers(0, 2))
            require = bool(rng.integers(0, 2))
            threshold = float(rng.choice([0.85, 0.90, 0.92, 0.95]))
            if keep:
                true_rate = w * p_core + (1 - w) * p_other
                coverage = 1.0
            else:
                true_rate = p_core
                coverage = w
            if require:
                true_rate *= p_adaptive
            prefix_n = max(20, int(round(250 * coverage)))
            empirical = rng.binomial(prefix_n, true_rate) / prefix_n
            features.append([empirical, coverage, threshold, float(keep), float(require)])
            targets.append(true_rate - threshold)
        self.model = RandomForestRegressor(
            n_estimators=96, max_depth=8, min_samples_leaf=3,
            random_state=seed, n_jobs=1)
        self.model.fit(np.asarray(features), np.asarray(targets))

    def predict_margin(self, empirical: float, coverage: float,
                       threshold: float, keep: bool, require: bool) -> float:
        row = np.asarray([[empirical, coverage, threshold,
                           float(keep), float(require)]])
        return float(self.model.predict(row)[0])


def _fmt(value, digits=1):
    if value is None or value == "":
        return "--"
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)
rq1_METHODS = ('direct', 'fixed_repeated', 'exact_spending', 'paper_hoeffding')

def rq1_wilson(k, n):
    z = 1.959963984540054
    m = k / n
    d = 1 + z * z / n
    c = (m + z * z / (2 * n)) / d
    h = z * math.sqrt(m * (1 - m) / n + z * z / (4 * n * n)) / d
    return (0 if k == 0 else max(0, c - h), 1 if k == n else min(1, c + h))

def rq1_first_cross(cumulative, critical):
    crosses = cumulative >= critical[None, :]
    yes = crosses.any(axis=1)
    q = np.where(yes, crosses.argmax(axis=1) + 1, 0)
    return (yes, q)

def rq1_run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    q = np.arange(1, args.max_completed + 1)
    delta_j = args.delta / 2
    fixed_critical = binom.isf(delta_j, q, args.threshold) + 1
    allocated = 6 * delta_j / (math.pi ** 2 * q * q)
    exact_spending_critical = binom.isf(allocated, q, args.threshold) + 1
    paper_radius = np.sqrt(np.log(1 / allocated) / (2 * q))
    paper_critical = np.ceil(q * (args.threshold + paper_radius) - 1e-12)
    rows = []
    details = []
    for pi, p in enumerate(args.probabilities):
        rng = np.random.default_rng(args.seed + 1000000 * pi)
        prefix = rng.random((args.replicates, args.candidates, args.warmup)) < p
        scores = prefix.sum(axis=2)
        selected = np.argmax(scores, axis=1)
        post = rng.random((args.replicates, args.max_completed)) < p
        cumulative = np.cumsum(post, axis=1, dtype=np.int32)
        decisions = {}
        direct_yes = np.ones(args.replicates, dtype=bool)
        decisions['direct'] = (direct_yes, np.zeros(args.replicates, dtype=int))
        decisions['fixed_repeated'] = rq1_first_cross(cumulative, fixed_critical)
        decisions['exact_spending'] = rq1_first_cross(cumulative, exact_spending_critical)
        paper_cross = cumulative / q - paper_radius[None, :] >= args.threshold
        paper_yes = paper_cross.any(axis=1)
        paper_q = np.where(paper_yes, paper_cross.argmax(axis=1) + 1, 0)
        decisions['paper_hoeffding'] = (paper_yes, paper_q)
        assert np.array_equal(paper_yes, rq1_first_cross(cumulative, paper_critical)[0])
        for method in rq1_METHODS:
            yes, decision_q = decisions[method]
            activated = int(yes.sum())
            false = activated if p < args.threshold else 0
            lo, hi = rq1_wilson(activated, args.replicates)
            m_q = float(np.median(decision_q[yes])) if activated else None
            median_event = float(args.warmup * 3 + 2 + 3 * m_q) if m_q is not None and method != 'direct' else float(args.warmup * 3 + 3) if method == 'direct' else None
            rows.append(dict(true_p=p, method=method, replicates=args.replicates, activations=activated, false_activations=false, activation_rate=activated / args.replicates, wilson95_low=lo, wilson95_high=hi, median_completed_at_decision=m_q, median_decision_event=median_event, censored=args.replicates - activated, max_completed=args.max_completed))
            for rep in range(args.replicates):
                details.append(dict(true_p=p, replicate=rep, method=method, selected_candidate=int(selected[rep]), selected_prefix_successes=int(scores[rep, selected[rep]]), selected_prefix_best_score=int(scores[rep].max()), activated=bool(yes[rep]), completed_at_decision=int(decision_q[rep]) if yes[rep] else '', decision_event=(args.warmup * 3 + 3 if method == 'direct' else args.warmup * 3 + 2 + 3 * int(decision_q[rep])) if yes[rep] else '', seed=args.seed + 1000000 * pi))
    for name, data in (('table_rq1.csv', rows), ('runs.csv', details)):
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    method_names = {
        'direct': 'Direct proposal',
        'fixed_repeated': 'Fixed-time, repeatedly tested',
        'exact_spending': 'Exact test with spending',
        'paper_hoeffding': "Paper's Hoeffding bound",
    }
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ1 synthetic post-selection certification. Each row has '
            f'{args.replicates} independent replications and {args.max_completed:,} '
            r'completed-obligation opportunities. Wilson intervals measure Monte Carlo '
            r'variation; $q$ is the median completed count at activation.}'),
           r'\label{tab:rq1-certification}', r'\resizebox{\linewidth}{!}{%',
           r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
           r'True $p$ & Procedure & Activations & Wilson 95\% (\%) & Median $q$ & Censored \\',
           r'\midrule']
    for p in args.probabilities:
        rr = [r for r in rows if r['true_p'] == p]
        for i, row in enumerate(rr):
            pcell = f'{p:.3f}' if i == 0 else ''
            interval = f"[{100*row['wilson95_low']:.1f}, {100*row['wilson95_high']:.1f}]"
            tex.append(f"{pcell} & {method_names[row['method']]} & "
                       f"{row['activations']}/{row['replicates']} & {interval} & "
                       f"{_fmt(row['median_completed_at_decision'])} & {row['censored']} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table}'])
    (out / 'table_rq1.tex').write_text('\n'.join(tex) + '\n')
    fig, ax = plt.subplots(figsize=(7.1, 4.2), layout='constrained')
    x = np.arange(len(args.probabilities))
    width = 0.19
    colors = ['#747b80', '#bc583d', '#4591ae', '#224e70']
    labels = ['Direct', 'Fixed-time, peeked', 'Exact, spent', 'Paper, Hoeffding']
    for i, method in enumerate(rq1_METHODS):
        rr = [r for r in rows if r['method'] == method]
        y = [r['activation_rate'] for r in rr]
        ax.bar(x + (i - 1.5) * width, y, width, color=colors[i], label=labels[i])
        ax.errorbar(x + (i - 1.5) * width, y, yerr=[np.maximum(0, np.array(y) - [r['wilson95_low'] for r in rr]), np.maximum(0, [r['wilson95_high'] for r in rr] - np.array(y))], fmt='none', ecolor='#222222', elinewidth=0.7, capsize=2)
    ax.set_xticks(x, [f'{p:.3f}' + (' (null)' if p < args.threshold else ' (valid)') for p in args.probabilities])
    ax.set(ylabel='Activation within horizon', ylim=(-0.03, 1.08), xlabel='True response probability of selected candidate', title='RQ1: selecting from the prefix, then certifying fresh outcomes')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.16))
    fig.savefig(out / 'figure_rq1.pdf', bbox_inches='tight')
    fig.savefig(out / 'figure_rq1.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    for r in rows:
        print(f"p={r['true_p']:.3f} {r['method']:16s} {r['activations']}/{r['replicates']}, median q={r['median_completed_at_decision']}")
    return rows

def rq1_cli(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--replicates', type=int, default=1000)
    ap.add_argument('--candidates', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=200)
    ap.add_argument('--max-completed', type=int, default=6000)
    ap.add_argument('--probabilities', type=float, nargs='+', default=[0.89, 0.899, 0.98])
    ap.add_argument('--seed', type=int, default=20260921)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--threshold', type=float, default=0.9)
    a = ap.parse_args(argv)
    if not (a.replicates > 0 and a.candidates > 0 and (a.warmup > 0) and (a.max_completed > 0) and (0 < a.delta < 1) and (0 < a.threshold < 1) and all((0 < p < 1 for p in a.probabilities))):
        ap.error('invalid experiment parameters')
    rq1_run(a)

# rq2_simulation.py (embedded standalone implementation)
"""RQ2 event-stream experiment for Algorithm 1's single-candidate certification branch.

Requires Python 3.10+, NumPy, Matplotlib. No data or network access is needed.
"""
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class rq2_Run:
    method: str
    w: float
    replicate: int
    seed: int
    activated: bool
    false_activation: bool
    decision_event: int | None
    all_completed: int
    core_completed: int
    operational_origins_after_activation: int
    retained_old_origins_completed: int
    ai_keep_adaptive_trigger: bool
    ai_predicted_margin: float

def rq2_lower(successes: int, count: int, budget: float) -> float:
    """Section 5's time-uniform, one-sided Hoeffding bound, or 0 if empty."""
    if not count:
        return 0.0
    radius = math.sqrt(math.log(math.pi ** 2 * count ** 2 / (6 * budget)) / (2 * count))
    return max(0.0, successes / count - radius)

def rq2_stream(seed: int, horizon: int, w: float, p_core: float, p_other: float):
    """A occurs at indices 3,6,9,...; C is protected; B at n+1 responds to n.

    e_n=(t_n=n,{A_n,C_n,B_n}).  The drift declaration is exogenous; a frozen
    supervised proposer selects an admissible candidate from an independent
    prefix before any certification outcome is observed.
    """
    rng = np.random.default_rng(seed)
    protected = rng.random(horizon + 1) < w
    success_draw = rng.random(horizon + 1)
    response = np.zeros(horizon + 1, dtype=np.bool_)
    origins = (np.arange(horizon) > 0) & (np.arange(horizon) % 3 == 0)
    response[1:] = origins & (success_draw[:-1] < np.where(protected[:-1], p_core, p_other))
    return (protected, response)

def rq2_ai_select(seed: int, w: float, p_core: float, p_other: float,
                  threshold: float, proposer: AggregateAIProposer,
                  prefix_size: int = 400):
    """Choose between the full admissible trigger and its core-only revision."""
    rng = np.random.default_rng(seed + 70000003)
    protected = rng.random(prefix_size) < w
    outcomes = rng.random(prefix_size) < np.where(protected, p_core, p_other)
    candidates = []
    for keep in (True, False):
        mask = np.ones(prefix_size, dtype=bool) if keep else protected
        if not mask.any():
            continue
        empirical = float(outcomes[mask].mean())
        coverage = float(mask.mean())
        score = proposer.predict_margin(empirical, coverage, threshold, keep, False)
        candidates.append((score, keep))
    return max(candidates)[1], max(candidates)[0]


def rq2_proposer_audit(proposer: AggregateAIProposer, seed: int,
                       tasks: int = 500, prefix_size: int = 300):
    """Audit ranking on close, independently generated candidate sets."""
    rng = np.random.default_rng(seed + 91000011)
    exact = 0
    regrets = []
    records = []
    for task in range(tasks):
        w = rng.uniform(0.05, 0.45)
        p_core = rng.uniform(0.72, 0.99)
        p_other = rng.uniform(0.78, 1.0)
        p_adaptive = rng.uniform(0.82, 1.0)
        threshold = float(rng.choice([0.85, 0.88, 0.90, 0.92]))
        protected = rng.random(prefix_size) < w
        base = rng.random(prefix_size) < np.where(protected, p_core, p_other)
        adaptive = rng.random(prefix_size) < p_adaptive
        ranked = []
        for keep in (True, False):
            for require in (True, False):
                mask = np.ones(prefix_size, dtype=bool) if keep else protected
                if not mask.any():
                    continue
                observed = base & (adaptive if require else True)
                empirical = float(observed[mask].mean())
                coverage = float(mask.mean())
                predicted = proposer.predict_margin(
                    empirical, coverage, threshold, keep, require)
                true_rate = (w * p_core + (1 - w) * p_other) if keep else p_core
                if require:
                    true_rate *= p_adaptive
                ranked.append((predicted, true_rate - threshold, keep, require))
        learned = max(ranked, key=lambda x: x[0])
        oracle = max(ranked, key=lambda x: x[1])
        is_exact = learned[2:] == oracle[2:]
        exact += int(is_exact)
        regret = oracle[1] - learned[1]
        regrets.append(regret)
        records.append(dict(task=task, exact_ranking=is_exact,
                            learned_keep=learned[2], learned_require=learned[3],
                            oracle_keep=oracle[2], oracle_require=oracle[3],
                            true_margin_regret=regret))
    return dict(tasks=tasks, exact=exact, errors=tasks - exact,
                accuracy=exact / tasks,
                mean_regret=float(np.mean(regrets))), records

def rq2_run_method(protected, response, method: str, w: float, rep: int,
                   seed: int, horizon: int, delta: float, threshold: float,
                   core_threshold: float, p_core: float,
                   ai_keep: bool, ai_score: float) -> rq2_Run:
    assert method in ('aggregate_only', 'joint')
    delta_j = delta / 2
    budget_all = delta_j if method == 'aggregate_only' else delta_j / 2
    budget_core = delta_j / 2 if method == 'joint' else 0.0
    n_selection = 0
    active_version = 0
    certification_monitor = True
    n_all = s_all = n_core = s_core = 0
    decision = None
    operational_origins_after_activation = 0
    retained_old_origins_completed = 0
    operational_origin_versions: list[int] = []
    candidate_origins: set[int] = set()
    for n in range(horizon + 1):
        if n >= 2:
            k = n - 2
            assert k < len(operational_origin_versions)
            if operational_origin_versions[k] == 0 and active_version == 1:
                retained_old_origins_completed += 1
            if certification_monitor and k in candidate_origins:
                x = int(response[k + 1])
                n_all += 1
                s_all += x
                if protected[k]:
                    n_core += 1
                    s_core += x
                candidate_origins.remove(k)
        is_origin = n > 0 and n % 3 == 0
        operational_origin_versions.append(active_version if is_origin else -1)
        if is_origin and active_version == 1:
            operational_origins_after_activation += 1
        if certification_monitor and is_origin and (ai_keep or protected[n]):
            candidate_origins.add(n)
        if certification_monitor and n > n_selection and n_all:
            pass_all = rq2_lower(s_all, n_all, budget_all) >= threshold
            pass_core = method == 'aggregate_only' or (n_core > 0 and rq2_lower(s_core, n_core, budget_core) >= core_threshold)
            if pass_all and pass_core:
                decision = n
                certification_monitor = False
                candidate_origins.clear()
                active_version = 1
    activated = decision is not None
    return rq2_Run(method, w, rep, seed, activated,
                   activated and p_core < core_threshold, decision, n_all,
                   n_core, operational_origins_after_activation,
                   retained_old_origins_completed, ai_keep, ai_score)

def rq2_wilson(k: int, n: int, z: float=1.959963984540054) -> tuple[float, float]:
    phat = k / n
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return (0.0 if k == 0 else max(0.0, center - half), 1.0 if k == n else min(1.0, center + half))

def rq2_summarize(runs: list[rq2_Run], horizon: int):
    rows = []
    for w in sorted({r.w for r in runs}):
        for method in ('aggregate_only', 'joint'):
            r = [x for x in runs if x.w == w and x.method == method]
            n = len(r)
            k = sum((x.false_activation for x in r))
            lo, hi = rq2_wilson(k, n)
            yes = [x for x in r if x.activated]
            med = lambda vals: float(np.median(vals)) if vals else None
            rows.append(dict(w=w, method=method, replicates=n, false_activations=k, false_activation_rate=k / n, wilson95_low=lo, wilson95_high=hi, activations=len(yes), censored=n - len(yes), median_decision_event=med([x.decision_event for x in yes]), median_all_at_decision=med([x.all_completed for x in yes]), median_core_at_decision=med([x.core_completed for x in yes]), median_all_at_stop=med([x.all_completed for x in r]), median_core_at_stop=med([x.core_completed for x in r]), median_core_by_end=med([x.core_completed for x in r]), ai_full_trigger_selections=sum(x.ai_keep_adaptive_trigger for x in r), mean_ai_predicted_margin=float(np.mean([x.ai_predicted_margin for x in r])), horizon=horizon))
    return rows

def rq2_plot(rows, output: Path, horizon: int, delta: float, threshold: float, core_threshold: float):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout='constrained')
    for method, color, marker, label in (('aggregate_only', '#b04a32', 'o', 'Aggregate only'), ('joint', '#236f92', 's', 'Aggregate + core')):
        rr = [r for r in rows if r['method'] == method]
        x = np.array([100 * r['w'] for r in rr])
        y = np.array([r['false_activation_rate'] for r in rr])
        yerr = [np.maximum(0, y - np.array([r['wilson95_low'] for r in rr])), np.maximum(0, np.array([r['wilson95_high'] for r in rr]) - y)]
        ax.errorbar(x, y, yerr=yerr, fmt=marker + '-', color=color, capsize=3, linewidth=1.9, label=label)
    ax.set(xlabel='Protected-trigger share (%)',
           ylabel='Below-core activation rate', ylim=(-0.035, 1.08),
           title='RQ2: core failure masked by aggregate success')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
              frameon=False, ncol=2)
    fig.text(0.11, -0.005, f'Synthetic streams; horizon={horizon:,} events; δ={delta}; λ={threshold}; λ_core={core_threshold}; Wilson 95% intervals', fontsize=8)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)

def rq2_main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--replicates', type=int, default=200)
    ap.add_argument('--horizon', type=int, default=45000)
    ap.add_argument('--shares', type=float, nargs='+', default=[0.01, 0.03, 0.05, 0.07])
    ap.add_argument('--seed', type=int, default=20260920)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--threshold', type=float, default=0.9)
    ap.add_argument('--core-threshold', type=float, default=0.9)
    ap.add_argument('--p-core', type=float, default=0.0)
    ap.add_argument('--p-other', type=float, default=1.0)
    ap.add_argument('--gradient-replicates', type=int, default=100)
    ap.add_argument('--gradient-horizon', type=int, default=150000)
    ap.add_argument('--gradient-w', type=float, default=0.2)
    ap.add_argument('--gradient-p-core', type=float, nargs='+',
                    default=[0.0, 0.5, 0.8, 0.88, 0.92, 0.95, 0.99])
    ap.add_argument('--gradient-seed', type=int, default=20260923)
    ap.add_argument('--ai-seed', type=int, default=20260930)
    ap.add_argument('--ai-training-tasks', type=int, default=4000)
    args = ap.parse_args(argv)
    if not (args.replicates > 0 and args.horizon > 2 and (0 < args.delta < 1) and (0 < args.threshold < 1) and (0 < args.core_threshold < 1) and (0 <= args.p_core <= 1) and (0 <= args.p_other <= 1) and all((0 < w < 1 for w in args.shares))):
        ap.error('invalid experiment parameter')
    args.out.mkdir(parents=True, exist_ok=True)
    proposer = AggregateAIProposer(args.ai_seed, args.ai_training_tasks)
    audit, audit_records = rq2_proposer_audit(proposer, args.ai_seed)
    with (args.out / 'proposer_audit.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_records[0]))
        writer.writeheader()
        writer.writerows(audit_records)
    runs = []
    for wi, w in enumerate(args.shares):
        for rep in range(args.replicates):
            seed = args.seed + wi * 1000000 + rep
            c, b = rq2_stream(seed, args.horizon, w, args.p_core, args.p_other)
            ai_keep, ai_score = rq2_ai_select(
                seed, w, args.p_core, args.p_other, args.threshold, proposer)
            for method in ('aggregate_only', 'joint'):
                runs.append(rq2_run_method(
                    c, b, method, w, rep, seed, args.horizon, args.delta,
                    args.threshold, args.core_threshold, args.p_core,
                    ai_keep, ai_score))
    rows = rq2_summarize(runs, args.horizon)
    for name, data in (('runs.csv', [vars(x) for x in runs]), ('table_rq2.csv', rows)):
        with (args.out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ2 masked-core experiment with proposals selected by the '
            r'frozen supervised proposer. A below-core activation occurs while '
            r'$p^{\mathrm{core}}<\lambda_{\mathrm{core}}$; brackets give Wilson 95\% intervals.}'),
           r'\label{tab:rq2-masked-core}', r'\begin{tabular}{@{}lrrrr@{}}',
           r'\toprule',
           r'$w$ & Certification rule & Below core & Wilson 95\% & Median event \\',
           r'\midrule']
    for row in rows:
        gov = 'Aggregate only' if row['method'] == 'aggregate_only' else 'Aggregate + core'
        interval = f"[{100*row['wilson95_low']:.1f}, {100*row['wilson95_high']:.1f}]"
        tex.append(f"{row['w']:.2f} & {gov} & {row['false_activations']}/{row['replicates']} & "
                   f"{interval} & {_fmt(row['median_decision_event'])} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (args.out / 'table_rq2.tex').write_text('\n'.join(tex) + '\n')
    audit_tex = [r'\begin{table}[t]', r'\centering', r'\small',
                 r'\caption{Independent ranking audit of the frozen RQ2 proposer.}',
                 r'\label{tab:rq2-proposer-audit}',
                 r'\begin{tabular}{@{}rrrr@{}}', r'\toprule',
                 r'Tasks & Exact rankings & Ranking errors & Mean margin regret \\',
                 r'\midrule',
                 f"{audit['tasks']} & {audit['exact']} & {audit['errors']} & "
                 f"{audit['mean_regret']:.4f} \\\\",
                 r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (args.out / 'table_rq2_proposer.tex').write_text(
        '\n'.join(audit_tex) + '\n')
    rq2_plot(rows, args.out / 'figure_rq2', args.horizon, args.delta, args.threshold, args.core_threshold)
    print(f'Saved {len(runs)} paired method runs to {args.out}; p_core={args.p_core}, p_other={args.p_other}.')
    for row in rows:
        print(f"w={row['w']:.3f} {row['method']:15s}: below-core activations {row['false_activations']}/{row['replicates']}, median decision event {row['median_decision_event']}")
    rq2_gradient(args, proposer)

def rq2_cli(argv):
    rq2_main(argv)

# rq2 second phase: protected-success sweep (same folder, distinct file names)
"""RQ2b: masked-core window as a function of the protected-group success rate."""
import argparse, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rq2_gradient(args, proposer):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pi, pc in enumerate(args.gradient_p_core):
        runs = []
        for rep in range(args.gradient_replicates):
            seed = args.gradient_seed + pi * 1000000 + rep
            c, b = rq2_stream(seed, args.gradient_horizon, args.gradient_w, pc, 1.0)
            ai_keep, ai_score = rq2_ai_select(
                seed, args.gradient_w, pc, 1.0, args.threshold, proposer)
            for m in ('aggregate_only', 'joint'):
                runs.append(rq2_run_method(c, b, m, args.gradient_w, rep, seed, args.gradient_horizon,
                                           args.delta, args.threshold,
                                           args.core_threshold, pc,
                                           ai_keep, ai_score))
        for m in ('aggregate_only', 'joint'):
            r = [x for x in runs if x.method == m]
            yes = [x for x in r if x.activated]
            k = sum(x.false_activation for x in r)
            lo, hi = rq2_wilson(k, len(r))
            med = lambda v: float(np.median(v)) if v else None
            rows.append(dict(p_core=pc, method=m, replicates=len(r),
                             activations=len(yes), censored=len(r) - len(yes),
                             false_activations=k, false_activation_rate=k / len(r),
                             wilson95_low=lo, wilson95_high=hi,
                             median_all_at_decision=med([x.all_completed for x in yes]),
                             median_core_at_decision=med([x.core_completed for x in yes]),
                             horizon=args.gradient_horizon, w=args.gradient_w))
        print(f"p_core={pc:.2f}  aggregate {rows[-2]['activations']}/{args.gradient_replicates}  "
              f"joint {rows[-1]['activations']}/{args.gradient_replicates}  "
              f"joint median core obligations {rows[-1]['median_core_at_decision']}")
    with (args.out / 'table_rq2_gradient.csv').open('w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ2 protected-success sweep at $w=0.20$. A dash means '
            r'that no run activated within the horizon.}'),
           r'\label{tab:rq2-gradient}', r'\begin{tabular}{@{}lrrrr@{}}',
           r'\toprule',
           r'$p^{\mathrm{core}}$ & Aggregate-only activations & Joint activations & Joint median core $q$ & Joint censored \\',
           r'\midrule']
    for pc in args.gradient_p_core:
        agg = next(r for r in rows if r['p_core'] == pc and r['method'] == 'aggregate_only')
        joint = next(r for r in rows if r['p_core'] == pc and r['method'] == 'joint')
        tex.append(f"{pc:.2f} & {agg['activations']}/{agg['replicates']} & "
                   f"{joint['activations']}/{joint['replicates']} & "
                   f"{_fmt(joint['median_core_at_decision'])} & {joint['censored']} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (args.out / 'table_rq2_gradient.tex').write_text('\n'.join(tex) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m, lab, mk in (('aggregate_only', 'aggregate-only', 'o'),
                       ('joint', 'joint (aggregate + core)', 's')):
        rr = [r for r in rows if r['method'] == m]
        x = [r['p_core'] for r in rr]
        ax[0].plot(x, [r['activations'] / r['replicates'] for r in rr], marker=mk, label=lab)
        y = [r['median_core_at_decision'] for r in rr]
        xs = [xi for xi, yi in zip(x, y) if yi is not None]
        ys = [yi for yi in y if yi is not None]
        if xs:
            ax[1].plot(xs, ys, marker=mk, label=lab)
    ax[0].axvline(args.core_threshold, ls=':', c='k')
    ax[0].set_xlabel(r'$p^{\rm core}$'); ax[0].set_ylabel('activation rate')
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(fontsize=7, loc='upper center',
                 bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax[0].set_title(f'Activation vs protected success ($w={args.gradient_w}$)', fontsize=9)
    ax[1].axvline(args.core_threshold, ls=':', c='k')
    ax[1].set_xlabel(r'$p^{\rm core}$')
    ax[1].set_ylabel('median core obligations\nat activation')
    ax[1].set_yscale('log')
    handles, labels = ax[1].get_legend_handles_labels()
    if handles:
        ax[1].legend(fontsize=7, loc='upper center',
                     bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax[1].set_title('Evidence needed to certify', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'figure_rq2_gradient.{ext}', bbox_inches='tight')
    print(f'Saved {len(rows)} summary rows to {args.out}')



# rq3_endtoend.py (embedded standalone implementation)
"""RQ3: multi-transition end-to-end run with a trained AI proposer.

Exercises detector -> proposer -> governor in sequence over several certified
activations, rather than one isolated certification branch.  The environment
degrades the protected-group response across successive regimes while
non-protected responses stay easy. A frozen supervised model ranks the
admissible neighbourhood by predicted aggregate margin. Two governors are
compared: aggregate-only certification and the paper's joint rule.
"""
import argparse, csv, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def rq3_lower(successes, count, budget):
    """Same time-uniform one-sided Hoeffding bound as Section 5."""
    if not count or budget <= 0:
        return 0.0
    radius = math.sqrt(math.log(math.pi ** 2 * count ** 2 / (6 * budget)) / (2 * count))
    return max(0.0, successes / count - radius)

def rq3_stream(seed, horizon, w, regime_edges, p_core_schedule, p_other):
    """A at indices 3,6,...; C protected; response to origin k observed at k+1.

    p_core_schedule[r] is the protected-group success probability in regime r;
    non-protected responses succeed with probability p_other throughout.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(horizon + 1)
    protected = rng.random(horizon + 1) < w
    draw = rng.random(horizon + 1)
    regime = np.searchsorted(np.asarray(regime_edges), idx, side='right')
    regime = np.clip(regime, 0, len(p_core_schedule) - 1)
    p_core_at = np.asarray(p_core_schedule)[regime]
    origins = (idx > 0) & (idx % 3 == 0)
    response = np.zeros(horizon + 1, dtype=np.bool_)
    ok = draw[:-1] < np.where(protected[:-1], p_core_at[:-1], p_other)
    response[1:] = origins[:-1] & ok
    return protected, response, p_core_at

# --- admissible candidate neighbourhood -------------------------------------
# The protected components are fixed by the designer and never revised.  A
# candidate is (lambda, keep_adaptive_trigger, require_adaptive_response):
# keep_adaptive_trigger=False replaces phi1_adapt by bottom, restricting the
# trigger to the protected core; require_adaptive_response=False replaces
# phi2_adapt by top, so the response reduces to the protected core.
def rq3_candidates(envelope):
    return [(lam, keep, req)
            for lam in envelope
            for keep in (True, False)
            for req in (True, False)]

def rq3_group_counts(protected, response, origins, keep_adaptive, adaptive_ok):
    """Counts (n_all, s_all, n_core, s_core) over the given origin indices."""
    n_all = s_all = n_core = s_core = 0
    for k in origins:
        if not keep_adaptive and not protected[k]:
            continue                      # trigger restricted to the core
        x = int(response[k + 1]) and (1 if adaptive_ok[k] else 0)
        n_all += 1; s_all += x
        if protected[k]:
            n_core += 1; s_core += x
    return n_all, s_all, n_core, s_core

@dataclass
class rq3_Run:
    governor: str
    rep: int
    seed: int
    activations: int
    final_lambda: float
    final_protected_prob: float
    sample_target_unsupported_activations: int
    current_regime_below_core_activations: int
    trigger_restricted: bool
    core_dropped: bool
    ai_proposals: int

def rq3_two_window(outcomes, h, eps):
    """Deterministic two-window drift declaration of Section 6."""
    if len(outcomes) < 2 * h:
        return False
    prev = sum(outcomes[-2 * h:-h]) / h
    curr = sum(outcomes[-h:]) / h
    return prev - curr >= eps

def rq3_origin_version_outcome(response_ok, adaptive_ok, origin_require_adaptive):
    """Evaluate a completed incumbent obligation under its origin version."""
    return int(response_ok) and (1 if (not origin_require_adaptive or adaptive_ok) else 0)

def rq3_test_origin_version_semantics():
    """A revision between origin and completion must not change the outcome."""
    old_requires_adaptive = True
    new_requires_adaptive = False
    response_ok, adaptive_ok = True, False
    origin_value = rq3_origin_version_outcome(
        response_ok, adaptive_ok, old_requires_adaptive)
    retroactive_value = rq3_origin_version_outcome(
        response_ok, adaptive_ok, new_requires_adaptive)
    assert origin_value == 0 and retroactive_value == 1

def rq3_run_history(protected, response, p_core_at, governor, rep, seed, args,
                    proposer):
    """One execution: repeated detect -> propose -> certify -> activate.

    Certification counters are maintained incrementally, so the cost per event
    is constant and independent of the number of completed obligations.
    """
    rng = np.random.default_rng(seed + 997)
    adaptive_ok = rng.random(len(protected)) < args.p_adaptive
    horizon = args.horizon
    delta = args.delta
    lam = args.lam0
    keep_adaptive, require_adaptive = True, True
    j = 0
    activations = 0
    sample_target_unsupported = 0
    current_regime_below_core = 0
    ai_proposals = 0
    incumbent_outcomes = []
    prefix_origins = []
    state = 'monitor'
    cand = None
    n_all = s_all = n_core = s_core = 0
    budget_all = budget_core = 0.0
    trace = []
    origin_keep = {}
    origin_require = {}
    mu_all_sum = mu_core_sum = 0.0
    for n in range(horizon + 1):
        # Record the active version at origin. A decision at n activates its
        # successor only at n+1 and cannot reinterpret this obligation.
        if n > 0 and n % 3 == 0:
            origin_keep[n] = keep_adaptive
            origin_require[n] = require_adaptive
        if n >= 2:
            k = n - 2
            if k > 0 and k % 3 == 0:
                prefix_origins.append(k)
                k_keep = origin_keep[k]
                k_require = origin_require[k]
                if k_keep or protected[k]:
                    x = rq3_origin_version_outcome(
                        response[k + 1], adaptive_ok[k], k_require)
                    incumbent_outcomes.append(x)
                    if len(incumbent_outcomes) > 4 * args.window:
                        del incumbent_outcomes[:args.window]
                if state == 'certify':
                    c_lam, c_keep, c_req = cand
                    if c_keep or protected[k]:
                        xc = int(response[k + 1]) and (1 if (not c_req or adaptive_ok[k]) else 0)
                        n_all += 1; s_all += xc
                        mu = (float(p_core_at[k]) if protected[k] else args.p_other)
                        if c_req:
                            mu *= args.p_adaptive
                        mu_all_sum += mu
                        if protected[k]:
                            n_core += 1; s_core += xc
                            mu_core_sum += mu
        if state == 'certify' and n_all:
            c_lam, c_keep, c_req = cand
            pass_all = rq3_lower(s_all, n_all, budget_all) >= c_lam
            pass_core = True if governor == 'aggregate_only' else (
                n_core > 0 and rq3_lower(s_core, n_core, budget_core) >= args.core_threshold)
            if pass_all and pass_core:
                sample_bad = (mu_all_sum / n_all < c_lam)
                if governor == 'joint':
                    sample_bad = sample_bad or (
                        n_core > 0 and mu_core_sum / n_core < args.core_threshold)
                lam, keep_adaptive, require_adaptive = cand
                activations += 1
                if sample_bad:
                    sample_target_unsupported += 1
                if p_core_at[n] < args.core_threshold:
                    current_regime_below_core += 1
                j += 1
                state = 'monitor'
                incumbent_outcomes = []
                n_all = s_all = n_core = s_core = 0
                mu_all_sum = mu_core_sum = 0.0
        if state == 'monitor' and rq3_two_window(incumbent_outcomes, args.window, args.margin):
            best, best_score = None, -1.0
            recent = prefix_origins[-args.prefix:]
            if recent:
                ones = np.ones(len(protected), dtype=bool)
                for c_lam, c_keep, c_req in rq3_candidates(args.envelope):
                    if (c_lam, c_keep, c_req) == (lam, keep_adaptive, require_adaptive):
                        continue
                    structural_changes = int(c_keep != keep_adaptive) + int(c_req != require_adaptive)
                    if structural_changes > 1:
                        continue
                    a_n, a_s, c_n, c_s = rq3_group_counts(
                        protected, response, recent, c_keep,
                        adaptive_ok if c_req else ones)
                    if a_n == 0:
                        continue
                    margin = a_s / a_n - c_lam
                    if margin <= 0:
                        continue
                    coverage = a_n / len(recent)
                    score = proposer.predict_margin(
                        a_s / a_n, coverage, c_lam, c_keep, c_req)
                    if score > best_score:
                        best, best_score = (c_lam, c_keep, c_req), score
            if best is not None:
                cand = best
                ai_proposals += 1
                delta_j = delta / (2 ** (j + 1))
                if governor == 'aggregate_only':
                    budget_all, budget_core = delta_j, 0.0
                else:
                    budget_all = budget_core = delta_j / 2
                state = 'certify'
                n_all = s_all = n_core = s_core = 0
                mu_all_sum = mu_core_sum = 0.0
            else:
                incumbent_outcomes = []
        if n % 3000 == 0:
            trace.append((n, float(p_core_at[n])))
    return rq3_Run(governor, rep, seed, activations, lam, float(p_core_at[horizon]),
                   sample_target_unsupported, current_regime_below_core,
                   not keep_adaptive, not require_adaptive, ai_proposals), trace

def rq3_wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))

def rq3_main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('results'))
    ap.add_argument('--replicates', type=int, default=100)
    ap.add_argument('--horizon', type=int, default=120000)
    ap.add_argument('--w', type=float, default=0.3)
    ap.add_argument('--p-core-schedule', type=float, nargs='+',
                    default=[0.99, 0.97, 0.80, 0.55])
    ap.add_argument('--p-other', type=float, default=1.0)
    ap.add_argument('--p-adaptive', type=float, default=0.97)
    ap.add_argument('--lam0', type=float, default=0.95)
    ap.add_argument('--envelope', type=float, nargs='+', default=[0.95, 0.92, 0.90])
    ap.add_argument('--core-threshold', type=float, default=0.9)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--window', type=int, default=60)
    ap.add_argument('--margin', type=float, default=0.03)
    ap.add_argument('--prefix', type=int, default=400)
    ap.add_argument('--ai-seed', type=int, default=20260930)
    ap.add_argument('--ai-training-tasks', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20260924)
    args = ap.parse_args(argv)
    rq3_test_origin_version_semantics()
    args.out.mkdir(parents=True, exist_ok=True)
    proposer = AggregateAIProposer(args.ai_seed, args.ai_training_tasks)
    edges = [int((i + 1) * args.horizon / len(args.p_core_schedule))
             for i in range(len(args.p_core_schedule) - 1)]
    runs, traces = [], {}
    for rep in range(args.replicates):
        seed = args.seed + rep
        protected, response, p_core_at = rq3_stream(
            seed, args.horizon, args.w, edges, args.p_core_schedule, args.p_other)
        for gov in ('aggregate_only', 'joint'):
            r, tr = rq3_run_history(
                protected, response, p_core_at, gov, rep, seed, args, proposer)
            runs.append(r)
            traces.setdefault(gov, tr)
    rows = []
    for gov in ('aggregate_only', 'joint'):
        rr = [r for r in runs if r.governor == gov]
        k = sum(r.sample_target_unsupported_activations > 0 for r in rr)
        k_regime = sum(r.current_regime_below_core_activations > 0 for r in rr)
        lo, hi = rq3_wilson(k, len(rr))
        rows.append(dict(
            governor=gov, replicates=len(rr),
            mean_activations=float(np.mean([r.activations for r in rr])),
            max_activations=int(np.max([r.activations for r in rr])),
            runs_with_sample_target_unsupported_activation=k,
            sample_target_unsupported_rate=k / len(rr),
            sample_target_wilson95_low=lo, sample_target_wilson95_high=hi,
            runs_with_current_regime_below_core_activation=k_regime,
            current_regime_below_core_rate=k_regime / len(rr),
            mean_sample_target_unsupported_activations=float(np.mean([
                r.sample_target_unsupported_activations for r in rr])),
            mean_current_regime_below_core_activations=float(np.mean([
                r.current_regime_below_core_activations for r in rr])),
            median_final_lambda=float(np.median([r.final_lambda for r in rr])),
            runs_trigger_restricted=sum(r.trigger_restricted for r in rr),
            runs_core_only_response=sum(r.core_dropped for r in rr),
            mean_ai_proposals=float(np.mean([r.ai_proposals for r in rr])),
            horizon=args.horizon, w=args.w))
    for name, data in (('runs.csv', [vars(r) for r in runs]), ('table_rq3.csv', rows)):
        with (args.out / name).open('w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(data[0])); wtr.writeheader(); wtr.writerows(data)
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ3 sequential experiment with the frozen supervised AI '
            r"proposer. `Below core' counts runs with at least one activation "
            r'during a regime where $p^{\mathrm{core}}<\lambda_{\mathrm{core}}$.}'),
           r'\label{tab:rq3-sequential}', r'\resizebox{\linewidth}{!}{%',
           r'\begin{tabular}{@{}lrrrr@{}}',
           r'\toprule',
           r'Certification rule & Mean activations & Mean proposals & Below core & Sample-target failure \\',
           r'\midrule']
    for row in rows:
        gov = 'Aggregate only' if row['governor'] == 'aggregate_only' else 'Aggregate + core'
        tex.append(f"{gov} & {row['mean_activations']:.2f} & {row['mean_ai_proposals']:.2f} & "
                   f"{row['runs_with_current_regime_below_core_activation']}/{row['replicates']} & "
                   f"{row['runs_with_sample_target_unsupported_activation']}/{row['replicates']} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}}', r'\end{table}'])
    (args.out / 'table_rq3.tex').write_text('\n'.join(tex) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    labels = {'aggregate_only': 'aggregate-only governor', 'joint': 'joint governor'}
    x = np.arange(2)
    ax[0].bar(x - 0.18, [r['mean_activations'] for r in rows], 0.36, label='activations (mean)')
    ax[0].bar(x + 0.18, [r['mean_current_regime_below_core_activations'] for r in rows], 0.36,
              label='current regime below core threshold (mean)')
    ax[0].set_xticks(x); ax[0].set_xticklabels([labels[r['governor']] for r in rows], fontsize=7)
    ax[0].set_ylabel('per run')
    ax[0].legend(fontsize=7, loc='upper center',
                 bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax[0].set_title('Certified activations over the history', fontsize=9)
    tr = traces['joint']
    ax[1].step([p[0] for p in tr], [p[1] for p in tr], where='post', label=r'$p^{\rm core}$ (environment)')
    ax[1].axhline(args.core_threshold, ls=':', c='k', label=r'$\lambda_{\rm core}$')
    ax[1].set_xlabel('event index'); ax[1].set_ylabel('protected success prob.')
    ax[1].legend(fontsize=7, loc='upper center',
                 bbox_to_anchor=(0.5, -0.25), frameon=False)
    ax[1].set_title('Environment schedule', fontsize=9)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'figure_rq3.{ext}', bbox_inches='tight')
    print(f'Saved {len(runs)} runs to {args.out}')
    for r in rows:
        print(f"{r['governor']:15s}: activations {r['mean_activations']:.2f} mean, "
              f"sample-target unsupported in {r['runs_with_sample_target_unsupported_activation']}/{r['replicates']} runs, "
              f"current-regime below core in {r['runs_with_current_regime_below_core_activation']}/{r['replicates']} runs, "
              f"median final lambda {r['median_final_lambda']}")

def rq3_cli(argv):
    rq3_main(argv)


# Monitored clinical-alarm case study and reference overhead benchmark.
def case_make_trace(seed: int, triggers: int, protected_share: float,
                    p_core: float, p_other: float, max_delay: int = 8):
    """Create a timed Boolean trace with non-overlapping alarm obligations.

    A is a desaturation alarm, C marks protocol-critical alarms, and B is a
    documented intervention. Alarm origins are spaced beyond the candidate
    response horizon, so every B event belongs to exactly one obligation.
    """
    rng = np.random.default_rng(seed)
    spacing = max_delay + 2
    horizon = triggers * spacing + max_delay + 1
    alarm = np.zeros(horizon, dtype=bool)
    critical = np.zeros(horizon, dtype=bool)
    intervention = np.zeros(horizon, dtype=bool)
    origins = np.arange(0, triggers * spacing, spacing, dtype=int)
    alarm[origins] = True
    groups = rng.random(triggers) < protected_share
    critical[origins] = groups
    success_prob = np.where(groups, p_core, p_other)
    successful = rng.random(triggers) < success_prob
    delays = rng.integers(1, max_delay + 1, size=triggers)
    intervention[origins[successful] + delays[successful]] = True
    return alarm, critical, intervention, origins


def case_monitor(intervention: np.ndarray, origins: np.ndarray,
                 lower: int, upper: int) -> np.ndarray:
    """Evaluate A -> eventually_[lower,upper] B at the supplied origins."""
    return np.fromiter(
        (bool(intervention[k + lower:k + upper + 1].any()) for k in origins),
        dtype=bool, count=len(origins))


def case_first_activation(outcomes, groups, threshold, core_threshold,
                          delta_j, joint):
    all_success = core_success = core_count = 0
    all_budget = delta_j / 2 if joint else delta_j
    core_budget = delta_j / 2
    for q, (x, is_core) in enumerate(zip(outcomes, groups), start=1):
        all_success += int(x)
        if is_core:
            core_count += 1
            core_success += int(x)
        all_ok = rq2_lower(all_success, q, all_budget) >= threshold
        core_ok = (not joint or
                   (core_count > 0 and
                    rq2_lower(core_success, core_count, core_budget) >= core_threshold))
        if all_ok and core_ok:
            return q, core_count
    return None, core_count


def case_peak_pending(origins: np.ndarray, upper: int) -> int:
    """Maximum unresolved obligations for one bounded-response monitor."""
    changes = []
    for k in origins:
        changes.append((int(k), 1))
        changes.append((int(k + upper + 1), -1))
    pending = peak = 0
    for _, change in sorted(changes, key=lambda z: (z[0], z[1])):
        pending += change
        peak = max(peak, pending)
    return peak


def case_peak_bytes(callable_):
    tracemalloc.start()
    result = callable_()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def case_main(argv=None):
    ap = argparse.ArgumentParser(description='Clinical-alarm monitored case study')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--triggers', type=int, default=30000)
    ap.add_argument('--prefix', type=int, default=1000)
    ap.add_argument('--protected-share', type=float, default=0.10)
    ap.add_argument('--p-core', type=float, default=0.60)
    ap.add_argument('--p-other', type=float, default=0.99)
    ap.add_argument('--threshold', type=float, default=0.90)
    ap.add_argument('--core-threshold', type=float, default=0.90)
    ap.add_argument('--delta-j', type=float, default=0.025)
    ap.add_argument('--repetitions', type=int, default=9)
    ap.add_argument('--seed', type=int, default=20260924)
    ap.add_argument('--ai-seed', type=int, default=20260930)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    alarm, critical, intervention, origins = case_make_trace(
        args.seed, args.triggers, args.protected_share,
        args.p_core, args.p_other)
    groups = critical[origins]
    candidate_outcomes = case_monitor(intervention, origins, 1, 8)
    incumbent_outcomes = case_monitor(intervention, origins, 1, 3)

    # A frozen learned proposer ranks the structurally admissible widened
    # response window from prefix aggregate evidence. A second proposal that
    # drops protected response B is rejected before ranking.
    proposer = AggregateAIProposer(args.ai_seed, 4000)
    prefix_rate = float(candidate_outcomes[:args.prefix].mean())
    predicted_margin = proposer.predict_margin(
        prefix_rate, 1.0, args.threshold, True, False)
    selected_window = '[1,8]' if predicted_margin >= 0 else '[1,3]'
    if selected_window != '[1,8]':
        raise RuntimeError('Frozen proposer did not select the manuscript candidate')

    post = candidate_outcomes[args.prefix:]
    post_groups = groups[args.prefix:]
    aggregate_q, aggregate_core_q = case_first_activation(
        post, post_groups, args.threshold, args.core_threshold,
        args.delta_j, False)
    joint_q, joint_core_q = case_first_activation(
        post, post_groups, args.threshold, args.core_threshold,
        args.delta_j, True)

    # Explicit pending-obligation regression: the origin version owns [1,8].
    pending = np.zeros(12, dtype=bool)
    pending[6] = True
    origin_version_outcome = int(pending[1:9].any())
    retroactive_short_window = int(pending[1:4].any())

    def baseline_run():
        return case_monitor(intervention, origins, 1, 3)

    def governed_run():
        operational = case_monitor(intervention, origins, 1, 3)
        certification = case_monitor(intervention, origins, 1, 8)
        decision = case_first_activation(
            certification[args.prefix:], post_groups, args.threshold,
            args.core_threshold, args.delta_j, True)
        return operational, certification, decision

    baseline_times, governed_times = [], []
    for _ in range(args.repetitions):
        tic = time.perf_counter()
        _ = baseline_run()
        baseline_times.append(time.perf_counter() - tic)
        tic = time.perf_counter()
        _ = governed_run()
        governed_times.append(time.perf_counter() - tic)
    _, baseline_peak_bytes = case_peak_bytes(baseline_run)
    _, governed_peak_bytes = case_peak_bytes(governed_run)
    baseline_ms = 1000 * float(np.median(baseline_times))
    governed_ms = 1000 * float(np.median(governed_times))
    overhead = governed_ms / baseline_ms
    trace_events = len(alarm)
    baseline_us_per_event = 1000 * baseline_ms / trace_events
    governed_us_per_event = 1000 * governed_ms / trace_events
    baseline_peak_pending = case_peak_pending(origins, 3)
    governed_peak_pending = baseline_peak_pending + case_peak_pending(origins, 8)

    row = dict(
        triggers=args.triggers,
        protected_share=float(groups.mean()),
        incumbent_success=float(incumbent_outcomes.mean()),
        candidate_success=float(candidate_outcomes.mean()),
        candidate_core_success=float(candidate_outcomes[groups].mean()),
        ai_predicted_margin=predicted_margin,
        structurally_rejected_candidates=1,
        selected_window=selected_window,
        aggregate_activation_completed=aggregate_q,
        joint_activation_completed=joint_q or '',
        joint_final_core_completed=joint_core_q,
        origin_version_outcome=origin_version_outcome,
        retroactive_short_window_outcome=retroactive_short_window,
        baseline_median_ms=baseline_ms,
        governed_median_ms=governed_ms,
        baseline_us_per_event=baseline_us_per_event,
        governed_us_per_event=governed_us_per_event,
        baseline_peak_kib=baseline_peak_bytes / 1024,
        governed_peak_kib=governed_peak_bytes / 1024,
        baseline_peak_pending=baseline_peak_pending,
        governed_peak_pending=governed_peak_pending,
        parallel_monitor_increment_ms=governed_ms - baseline_ms,
        overhead_ratio=overhead)
    with (args.out / 'case_study.csv').open('w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(row)); wtr.writeheader(); wtr.writerow(row)

    agg_text = f'{aggregate_q:,}' if aggregate_q is not None else '--'
    joint_text = f'{joint_q:,}' if joint_q is not None else 'not activated'
    tex = [
        r'\begin{table}[t]', r'\centering', r'\small',
        r'\caption{Monitored clinical-alarm case study and reference runtime.}',
        r'\label{tab:clinical-case}',
        r'\resizebox{\linewidth}{!}{%',
        r'\begin{tabular}{@{}lr@{}}', r'\toprule',
        r'Quantity & Result \\', r'\midrule',
        f'Completed alarm obligations & {args.triggers:,} \\\\',
        f'Observed protected share & {groups.mean():.3f} \\\\',
        f'Candidate aggregate success & {candidate_outcomes.mean():.3f} \\\\',
        f'Candidate protected success & {candidate_outcomes[groups].mean():.3f} \\\\',
        f'Aggregate-only activation sample & {agg_text} \\\\',
        f'Joint activation & {joint_text} \\\\',
        f'Pending outcome: origin / retroactive & {origin_version_outcome} / {retroactive_short_window} \\\\',
        f'Baseline / governed median runtime & {baseline_ms:.1f} / {governed_ms:.1f} ms \\\\',
        f'Baseline / governed time per event & {baseline_us_per_event:.3f} / {governed_us_per_event:.3f} $\\mu$s \\\\',
        f'Baseline / governed peak memory & {baseline_peak_bytes/1024:.1f} / {governed_peak_bytes/1024:.1f} KiB \\\\',
        f'Baseline / governed peak pending & {baseline_peak_pending} / {governed_peak_pending} \\\\',
        f'Parallel-governor increment & {governed_ms-baseline_ms:.1f} ms \\\\',
        f'Reference runtime ratio & {overhead:.2f}$\\times$ \\\\',
        r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (args.out / 'table_case.tex').write_text('\n'.join(tex) + '\n')
    print(f'Saved clinical case study to {args.out}; aggregate={agg_text}, '
          f'joint={joint_text}, overhead={overhead:.2f}x')


def case_cli(argv):
    case_main(argv)

def cli():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('rq1','rq2','rq3','case','all'))
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--quick',action='store_true',help='small smoke run, not manuscript numbers')
    args,unknown=parser.parse_known_args()
    runners={'rq1':rq1_cli,'rq2':rq2_cli,'rq3':rq3_cli,'case':case_cli}
    if args.experiment=='all':
        if unknown:parser.error('experiment-specific flags require one RQ')
        for name,runner in runners.items():
            more={'rq1':['--replicates','2','--max-completed','100'],
                  'rq2':['--replicates','2','--horizon','150','--gradient-replicates','2','--gradient-horizon','600'],
                  'rq3':['--replicates','2','--horizon','3000'],
                  'case':['--triggers','500','--repetitions','2']}[name] if args.quick else []
            print('Running',name,flush=True)
            runner(['--out',str(args.out/name)]+more)
    else:
        if args.quick:parser.error('--quick applies to all; supply explicit RQ parameters for a single RQ')
        runners[args.experiment](['--out',str(args.out)]+unknown)

if __name__=='__main__':
    cli()
