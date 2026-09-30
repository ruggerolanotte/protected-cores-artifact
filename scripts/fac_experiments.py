#!/usr/bin/env python3
"""Reproduce the paper's masked-core, optional-continuation, and sequential experiments.

Dependencies: Python 3.10+, numpy, scipy, matplotlib, scikit-learn. No network or data files.
Run: python fac_experiments.py all --out results
     python fac_experiments.py rq2 --out results/rq2 --replicates 200
For an inexpensive smoke run: python fac_experiments.py all --out smoke --quick
The public ``all`` target regenerates experiment outputs, including ancillary plots.
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


# Executable fragment: Boolean trigger/response formulas and finite [a,b].
# The governor is independent of the learned proposal-ranking mechanism.
from dataclasses import dataclass, field, replace
from collections import deque
from functools import lru_cache

ATOM_BITS = {'A': 1, 'C': 2, 'B': 4, 'D': 8}
TOP, BOTTOM = ('true',), ('false',)
A, C, B, D = (('atom', x) for x in ('A', 'C', 'B', 'D'))
CORE_TRIGGER = ('and', A, C)
ADAPT_TRIGGER = ('and', A, ('not', C))

def boolean_value(formula, mask):
    op, *args = formula
    if op == 'true' and not args: return True
    if op == 'false' and not args: return False
    if op == 'atom' and len(args) == 1: return bool(mask & ATOM_BITS[args[0]])
    if op == 'not' and len(args) == 1: return not boolean_value(args[0], mask)
    if op in ('and', 'or') and len(args) == 2:
        x, y = boolean_value(args[0], mask), boolean_value(args[1], mask)
        return x and y if op == 'and' else x or y
    raise ValueError('unsupported Boolean formula')

@lru_cache(maxsize=None)
def truth_table(formula):
    return tuple(boolean_value(formula, mask) for mask in range(16))

@dataclass(frozen=True, slots=True)
class Specification:
    core_trigger: tuple
    adaptive_trigger: tuple
    core_response: tuple
    adaptive_response: tuple
    threshold: float
    lower: float
    upper: float

    def __post_init__(self):
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)
                and 0 <= self.lower <= self.upper and 0 <= self.threshold <= 1):
            raise ValueError('invalid specification parameters')
        for f in (self.core_trigger, self.adaptive_trigger,
                  self.core_response, self.adaptive_response): truth_table(f)

    @property
    def parameters(self): return self.threshold, self.lower, self.upper

    @property
    def trigger(self): return ('or', self.core_trigger, self.adaptive_trigger)

    @property
    def response(self): return ('and', self.core_response, self.adaptive_response)


def specification(lam=.9, keep=True, require=False, lower=1, upper=1):
    return Specification(CORE_TRIGGER, ADAPT_TRIGGER if keep else BOTTOM,
                         B, D if require else TOP, lam, lower, upper)


def revision_rejections(incumbent, candidate, envelope):
    """All four checks in Definition admissible-revision; syntactic equality."""
    errors = []
    if (candidate.core_trigger != incumbent.core_trigger or
            candidate.core_response != incumbent.core_response):
        errors.append('protected_component_changed')
    if (int(candidate.adaptive_trigger != incumbent.adaptive_trigger) +
            int(candidate.adaptive_response != incumbent.adaptive_response)) > 1:
        errors.append('two_adaptive_components_changed')
    if candidate.parameters not in envelope: errors.append('outside_parameter_envelope')
    if (candidate.adaptive_trigger == incumbent.adaptive_trigger and
        candidate.adaptive_response == incumbent.adaptive_response and
        candidate.parameters == incumbent.parameters): errors.append('no_revision')
    return tuple(errors)


@dataclass(slots=True)
class PendingObligation:
    origin: int
    time: float
    core: bool
    success: bool = False

@dataclass(frozen=True, slots=True)
class CompletedObligation:
    origin: int
    completion: int
    core: bool
    outcome: int
    version: int
    specification: Specification

class IncrementalMonitor:
    """One immutable origin specification; strict t > origin_time + H completion.

    H=b for this propositional response/trigger fragment. Retired monitors
    accept no new origins but keep resolving their existing queue.
    """
    def __init__(self, spec, version=0, after=-1):
        self.spec, self.version, self.after = spec, version, after
        self.triggers, self.responses = truth_table(spec.trigger), truth_table(spec.response)
        self.cores = truth_table(spec.core_trigger)
        self.pending = deque()
        self.accept_origins = True
        self.last_index, self.last_time = -1, -math.inf
        self.peak_pending = self.completed = self.generated = 0

    def advance(self, index, timestamp, mask):
        if index <= self.last_index or timestamp < self.last_time:
            raise ValueError('event indices must increase; timestamps must not decrease')
        self.last_index, self.last_time = index, timestamp
        done = []
        while self.pending and timestamp > self.pending[0].time + self.spec.upper:
            p = self.pending.popleft()
            done.append(CompletedObligation(p.origin, index, p.core, int(p.success),
                                            self.version, self.spec))
            self.completed += 1
        if self.accept_origins and index > self.after and self.triggers[mask]:
            self.pending.append(PendingObligation(index, timestamp, self.cores[mask]))
            self.generated += 1
        if self.responses[mask]:
            for p in self.pending:
                if self.spec.lower <= timestamp - p.time <= self.spec.upper:
                    p.success = True
        self.peak_pending = max(self.peak_pending, len(self.pending))
        return done

class CertificationEvidence:
    def __init__(self, spec, delta, joint, core_threshold):
        if not 0 < delta < 1: raise ValueError('invalid error allocation')
        self.spec, self.core_threshold = spec, core_threshold
        self.core_enabled = joint and spec.core_trigger != BOTTOM
        self.all_budget = delta/2 if self.core_enabled else delta
        self.core_budget = delta/2 if self.core_enabled else 0
        self.n = self.s = self.nc = self.sc = 0
        self.lower = self.core_lower = 0.0

    def add(self, completed):
        for x in completed:
            self.n += 1; self.s += x.outcome
            if x.core: self.nc += 1; self.sc += x.outcome
        if completed:
            self.lower = rq2_lower(self.s, self.n, self.all_budget)
            if self.core_enabled:
                self.core_lower = rq2_lower(self.sc, self.nc, self.core_budget)

    def passes(self):
        return self.n > 0 and self.lower >= self.spec.threshold and (
            not self.core_enabled or (self.nc > 0 and self.core_lower >= self.core_threshold))

class Governor:
    """Single attempt per activated version; decision at n, activation at n+1.

    Caller supplies drift-gated selection from observations already processed.
    A rejected proposal does not consume the version's certification slot.
    """
    def __init__(self, incumbent, envelope, delta=.05, joint=True, core_threshold=.9):
        if incumbent.parameters not in envelope: raise ValueError('incumbent outside envelope')
        self.envelope, self.delta, self.joint = frozenset(envelope), delta, joint
        self.core_threshold = core_threshold
        self.active = IncrementalMonitor(incumbent)
        self.retired = []
        self.version = 0
        self.attempted = False
        self.candidate = self.evidence = None
        self.selection_index = None
        self.scheduled = None
        self.last_index = -1
        self.decisions, self.activations, self.selections = [], [], []
        self.rejections = []
        self.peak_pending = 0

    def admissible(self, candidate):
        return not revision_rejections(self.active.spec, candidate, self.envelope)

    def select(self, candidate, index, drift_declared):
        if index != self.last_index: raise ValueError('selection must follow processing its event')
        if not drift_declared: raise ValueError('selection requires a drift declaration')
        if self.attempted: raise ValueError('only one selected candidate per version')
        reasons = revision_rejections(self.active.spec, candidate, self.envelope)
        if reasons:
            self.rejections.append((index, reasons)); return False
        self.attempted = True
        self.selection_index = index
        self.candidate = IncrementalMonitor(candidate, self.version+1, after=index)
        self.evidence = CertificationEvidence(candidate, self.delta/(2**(self.version+1)),
                                               self.joint, self.core_threshold)
        self.selections.append((index, candidate))
        return True

    def advance(self, index, timestamp, mask):
        if index != self.last_index+1: raise ValueError('process consecutive event indices')
        self.last_index = index
        if self.scheduled is not None:
            at, spec = self.scheduled
            if index != at: raise AssertionError('activation must occur at next event')
            self.active.accept_origins = False
            self.retired.append(self.active)
            self.version += 1
            self.active = IncrementalMonitor(spec, self.version)
            self.activations.append((index, spec))
            self.scheduled = None
            self.attempted = False
            self.candidate = self.evidence = None
        operational = self.active.advance(index, timestamp, mask)
        for old in self.retired: operational.extend(old.advance(index, timestamp, mask))
        self.retired = [x for x in self.retired if x.pending]
        completed = []
        if self.candidate is not None:
            completed = self.candidate.advance(index, timestamp, mask)
            self.evidence.add(completed)
            if self.evidence.passes():
                self.peak_pending = max(self.peak_pending, len(self.active.pending) +
                    sum(len(x.pending) for x in self.retired) + len(self.candidate.pending))
                self.decisions.append(dict(index=index, version=self.version,
                    specification=self.candidate.spec, n=self.evidence.n, nc=self.evidence.nc,
                    lower=self.evidence.lower, core_lower=self.evidence.core_lower))
                self.scheduled = (index+1, self.candidate.spec)
                # Certification obligations are never reclassified as operational.
                self.candidate = None
        pending = len(self.active.pending) + sum(len(x.pending) for x in self.retired)
        if self.candidate is not None: pending += len(self.candidate.pending)
        self.peak_pending = max(self.peak_pending, pending)
        return operational, completed


def event_masks(protected, response, adaptive=None):
    n = np.arange(len(protected))
    masks = ((n > 0) & (n % 3 == 0)).astype(np.uint8)
    masks |= protected.astype(np.uint8) * 2
    masks |= response.astype(np.uint8) * 4
    if adaptive is not None: masks[1:] |= adaptive[:-1].astype(np.uint8) * 8
    return masks


def protocol_tests():
    """Independent finite-trace oracle and boundary/protocol regression tests."""
    s = specification(0, upper=8)
    env = {(0,1,8),(0,1,3)}
    bad = replace(s, core_response=TOP)
    assert 'protected_component_changed' in revision_rejections(s,bad,env)
    assert 'no_revision' in revision_rejections(s,s,env)
    assert 'two_adaptive_components_changed' in revision_rejections(
        s,replace(s,adaptive_trigger=BOTTOM,adaptive_response=D),env)
    assert 'outside_parameter_envelope' in revision_rejections(s,replace(s,upper=9),env)
    g = Governor(s, env, joint=False)
    outputs=[]
    for n in range(12):
        op,_=g.advance(n,n,(1 if n in (0,2,7) else 0)|(4 if n==6 else 0))
        outputs.extend(op)
        if n==0:
            assert not g.select(bad,n,True)
            assert g.select(replace(s,upper=3),n,True)
    x=next(x for x in outputs if x.origin==0)
    assert x.outcome==1 and x.specification.upper==8 and x.version==0
    assert g.decisions[0]['index']==6 and g.activations[0][0]==7
    assert next(x for x in outputs if x.origin==7).specification.upper==3
    # Zero-delay and ties: response visible now, admitted only strictly later.
    z=IncrementalMonitor(specification(0,lower=0,upper=0))
    assert z.advance(0,0,5)==[] and z.advance(1,0,1)==[]
    got=z.advance(2,1,0)
    assert [(a.origin,a.outcome) for a in got]==[(0,1),(1,0)]
    # Overlap, repeated timestamps, and exact interval endpoints vs offline semantics.
    rng=np.random.default_rng(404)
    checks=0
    for lower,upper in [(0,0),(0,2),(1,3),(2,2)]:
        spec=specification(.9,require=True,lower=lower,upper=upper)
        for rep in range(30):
            times=np.cumsum(rng.integers(0,3,40)).tolist()
            masks=rng.integers(0,16,40).tolist()
            times.append(times[-1]+upper+1);masks.append(0)
            mon=IncrementalMonitor(spec);actual=[]
            for i,(t,v) in enumerate(zip(times,masks)): actual.extend(mon.advance(i,t,v))
            expected=[]
            for k,v in enumerate(masks):
                if boolean_value(spec.trigger,v) and any(t>times[k]+upper for t in times[k:]):
                    ok=any(lower<=times[i]-times[k]<=upper and boolean_value(spec.response,masks[i])
                           for i in range(k,len(times)))
                    c=next(i for i in range(k,len(times)) if times[i]>times[k]+upper)
                    expected.append((k,c,int(ok)))
            assert [(x.origin,x.completion,x.outcome) for x in actual]==expected
            assert all(x.core==boolean_value(spec.core_trigger,masks[x.origin]) for x in actual)
            checks+=1
    # No prefix-origin reuse; a noncertifying candidate cannot be replaced.
    s=specification(1,upper=1);t=replace(s,upper=2);g=Governor(s,{s.parameters,t.parameters},joint=False)
    g.advance(0,0,1);g.select(t,0,True)
    for n in range(1,5):g.advance(n,n,0)
    assert g.evidence.n==0
    try:g.select(t,4,True)
    except ValueError:pass
    else:raise AssertionError('second attempt accepted')
    # Joint needs a nonempty core sample even for a zero threshold.
    ev=CertificationEvidence(specification(0),.025,True,0)
    ev.add([CompletedObligation(1,3,False,1,0,s)])
    assert not ev.passes()
    return dict(offline_trace_comparisons=checks, boundary_and_protocol_checks='passed',
                origin_version_outcome=x.outcome, retroactive_short_window_outcome=0,
                regression_decision_event=6, regression_activation_event=7)


class AggregateAIProposer:
    """Frozen supervised proposer trained on independent synthetic tasks.

    The model predicts a candidate's future aggregate margin from its prefix
    aggregate frequency, trigger coverage, threshold, and two structural
    indicators. For a core-only candidate the candidate frequency is the
    protected-group rate; full-trigger candidates have no separate core feature.
    The symbolic governor, not the proposer, is responsible for core safety.
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
        'direct': 'Direct activation',
        'fixed_repeated': 'Repeated exact test',
        'exact_spending': 'Spent exact test',
        'paper_hoeffding': "Hoeffding bound",
    }
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ1 synthetic post-selection certification. Each row has '
            f'{args.replicates} independent replications and {args.max_completed:,} '
            r'completed-obligation opportunities. Wilson intervals measure Monte Carlo '
            r'variation; $q$ is the median completed count at activation.}'),
           r'\label{tab:rq1-certification}', 
           r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
           r'True $p$ & Procedure & Activations & \shortstack{Wilson 95\%\\(\%)} & \shortstack{Median\\$q$} & Censored \\',
           r'\midrule']
    for p in args.probabilities:
        rr = [r for r in rows if r['true_p'] == p]
        for i, row in enumerate(rr):
            pcell = f'{p:.3f}' if i == 0 else ''
            interval = f"[{100*row['wilson95_low']:.1f}, {100*row['wilson95_high']:.1f}]"
            tex.append(f"{pcell} & {method_names[row['method']]} & "
                       f"{row['activations']}/{row['replicates']} & {interval} & "
                       f"{_fmt(row['median_completed_at_decision'])} & {row['censored']} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
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
"""RQ2 event-stream experiment for the governor's single-candidate certification branch.

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
    """The paper's time-uniform, one-sided Hoeffding bound, or 0 if empty."""
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
        records.append(dict(task=task, optimal_choice=is_exact,
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
    # Candidate selection at index 0 from a fixed full-trigger [1,2] incumbent.
    incumbent=specification(threshold,True,False,upper=2)
    candidate=specification(threshold,ai_keep,False)
    g=Governor(incumbent,{incumbent.parameters,candidate.parameters},delta,method=='joint',core_threshold)
    masks=event_masks(protected,response)
    decision=None; n_all=n_core=after=retained=0
    for n,v in enumerate(masks):
        op,cert=g.advance(n,n,int(v))
        if n==0: assert g.select(candidate,n,True)
        after += int(bool(v&1) and g.version==1 and g.active.triggers[int(v)])
        retained += sum(x.version==0 and g.version>0 for x in op)
        if cert: n_all+=len(cert);n_core+=sum(x.core for x in cert)
        if g.decisions and decision is None:decision=g.decisions[0]['index']
    activated=bool(g.activations)
    return rq2_Run(method,w,rep,seed,activated,activated and p_core<core_threshold,
                   decision,n_all,n_core,after,retained,ai_keep,ai_score)

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
    for method, color, marker, label in (('aggregate_only', '#b04a32', 'o', 'Aggregate only'), ('joint', '#236f92', 's', 'Joint')):
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
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--gradient-only',action='store_true')
    args = ap.parse_args(argv)
    if not (args.replicates > 0 and args.horizon > 2 and (0 < args.delta < 1) and (0 < args.threshold < 1) and (0 < args.core_threshold < 1) and (0 <= args.p_core <= 1) and (0 <= args.p_other <= 1) and all((0 < w < 1 for w in args.shares))):
        ap.error('invalid experiment parameter')
    args.out.mkdir(parents=True, exist_ok=True)
    proposer = AggregateAIProposer(args.ai_seed, args.ai_training_tasks)
    if args.gradient_only:
        rq2_gradient(args,proposer); return
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
           r'$w$ & Certification rule & Below-core & Wilson 95\% & \shortstack{Median decision\\index} \\',
           r'\midrule']
    for row in rows:
        gov = 'Aggregate only' if row['method'] == 'aggregate_only' else 'Joint'
        interval = f"[{100*row['wilson95_low']:.1f}, {100*row['wilson95_high']:.1f}]"
        tex.append(f"{row['w']:.2f} & {gov} & {row['false_activations']}/{row['replicates']} & "
                   f"{interval} & {_fmt(row['median_decision_event'])} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (args.out / 'table_rq2.tex').write_text('\n'.join(tex) + '\n')
    audit_tex = [r'\begin{table}[t]', r'\centering', r'\small',
                 r'\caption{Independent top-choice audit of the frozen RQ2 proposer.}',
                 r'\label{tab:rq2-proposer-audit}',
                 r'\begin{tabular}{@{}rrrr@{}}', r'\toprule',
                 r'Tasks & Optimal choices & Other choices & Mean margin regret \\',
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

def rq2_worker_init(proposer):
    global _RQ2_WORKER_PROPOSER
    _RQ2_WORKER_PROPOSER=proposer

def rq2_gradient_job(job):
    rep,seed,pc,args=job
    c,b=rq2_stream(seed,args.gradient_horizon,args.gradient_w,pc,1.0)
    keep,score=rq2_ai_select(seed,args.gradient_w,pc,1.0,args.threshold,_RQ2_WORKER_PROPOSER)
    return [rq2_run_method(c,b,m,args.gradient_w,rep,seed,args.gradient_horizon,
            args.delta,args.threshold,args.core_threshold,pc,keep,score)
            for m in ('aggregate_only','joint')]

def rq2_gradient(args, proposer):
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    all_gradient_runs = []
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    rq2_worker_init(proposer)
    executor=(ProcessPoolExecutor(max_workers=args.workers,
              mp_context=multiprocessing.get_context('spawn'),
              initializer=rq2_worker_init,initargs=(proposer,)) if args.workers>1 else None)
    for pi, pc in enumerate(args.gradient_p_core):
        runs = []
        jobs=[(rep,args.gradient_seed+pi*1000000+rep,pc,args)
              for rep in range(args.gradient_replicates)]
        pairs=executor.map(rq2_gradient_job,jobs) if executor else map(rq2_gradient_job,jobs)
        for pair in pairs:runs.extend(pair)
        all_gradient_runs.extend(vars(x) | {'p_core': pc} for x in runs)
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
    if executor:executor.shutdown()
    with (args.out / 'gradient_runs.csv').open('w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(all_gradient_runs[0])); wtr.writeheader(); wtr.writerows(all_gradient_runs)
    with (args.out / 'table_rq2_gradient.csv').open('w', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ2 protected-success sweep at $w=0.20$. A dash means '
            r'that no run activated within the horizon.}'),
           r'\label{tab:rq2-gradient}', r'\begin{tabular}{@{}lrrrr@{}}',
           r'\toprule',
           r'$p^{\mathrm{core}}$ & \shortstack{Aggregate-only\\activations} & \shortstack{Joint\\activations} & \shortstack{Joint median\\core $q$} & \shortstack{Joint\\censored} \\',
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
                       ('joint', 'joint certification', 's')):
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
    oracle_composition_unsupported_activations: int
    current_regime_below_core_activations: int
    trigger_restricted: bool
    core_dropped: bool
    ai_selections: int

def rq3_two_window(outcomes, h, eps):
    """Deterministic two-window drift declaration of Section 6."""
    if len(outcomes) < 2 * h:
        return False
    prev = sum(outcomes[-2 * h:-h]) / h
    curr = sum(outcomes[-h:]) / h
    return prev - curr >= eps



def rq3_run_history(protected, response, p_core_at, governor, rep, seed, args, proposer):
    rng=np.random.default_rng(seed+997)
    adaptive_ok=rng.random(len(protected))<args.p_adaptive
    masks=event_masks(protected,response,adaptive_ok)
    initial=specification(args.lam0,True,True)
    env={(x,1,1) for x in args.envelope}
    g=Governor(initial,env,args.delta,governor=='joint',args.core_threshold)
    incumbent_outcomes=[];prefix_origins=[];trace=[]
    mu_all=mu_core=0.;n_all=n_core=0
    oracle_bad=regime_bad=0
    pending_oracle_bad=False
    last_version=0
    neighborhood=[specification(lam,keep,req) for lam,keep,req in rq3_candidates(args.envelope)]
    ones=np.ones(len(protected),dtype=bool)
    for n,v in enumerate(masks):
        op,cert=g.advance(n,n,int(v))
        if g.version!=last_version:
            oracle_bad+=int(pending_oracle_bad)
            regime_bad+=int(p_core_at[n]<args.core_threshold)
            incumbent_outcomes=[];last_version=g.version
        incumbent_outcomes.extend(x.outcome for x in op if x.version==g.version)
        if len(incumbent_outcomes)>4*args.window:del incumbent_outcomes[:-3*args.window]
        # All A opportunities complete at k+2 in the [1,1] candidate family.
        k=n-2
        if k>0 and k%3==0:
            prefix_origins.append(k)
            if len(prefix_origins)>args.prefix:del prefix_origins[0]
        for x in cert:
            # Descriptive oracle for realised group/origin, not the theorem estimand.
            mu=float(p_core_at[x.origin]) if x.core else args.p_other
            if x.specification.adaptive_response!=TOP:mu*=args.p_adaptive
            mu_all+=mu;n_all+=1
            if x.core:mu_core+=mu;n_core+=1
        if g.decisions and g.decisions[-1]['index']==n:
            d=g.decisions[-1]
            bad=mu_all/n_all<d['specification'].threshold
            if governor=='joint':bad=bad or mu_core/n_core<args.core_threshold
            pending_oracle_bad=bad
        if not g.attempted and rq3_two_window(incumbent_outcomes,args.window,args.margin):
            candidates=[];features=[]
            for spec in neighborhood:
                if not g.admissible(spec):continue
                keep=spec.adaptive_trigger!=BOTTOM;req=spec.adaptive_response!=TOP
                an,ass,cn,cs=rq3_group_counts(protected,response,prefix_origins,keep,
                                             adaptive_ok if req else ones)
                if an==0 or ass/an<=spec.threshold:continue
                candidates.append(spec)
                features.append([ass/an,an/len(prefix_origins),spec.threshold,float(keep),float(req)])
            if candidates:
                scores=proposer.model.predict(np.asarray(features))
                best=int(np.argmax(scores))
                assert g.select(candidates[best],n,True)
                mu_all=mu_core=0.;n_all=n_core=0
            else:incumbent_outcomes=[]
        if n%3000==0:trace.append((n,float(p_core_at[n])))
    spec=g.active.spec
    return rq3_Run(governor,rep,seed,len(g.activations),spec.threshold,float(p_core_at[-1]),
                   oracle_bad,regime_bad,spec.adaptive_trigger==BOTTOM,
                   spec.adaptive_response==TOP,len(g.selections)),trace

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
    protocol_tests()
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
        k = sum(r.oracle_composition_unsupported_activations > 0 for r in rr)
        k_regime = sum(r.current_regime_below_core_activations > 0 for r in rr)
        lo, hi = rq3_wilson(k, len(rr))
        rows.append(dict(
            governor=gov, replicates=len(rr),
            mean_activations=float(np.mean([r.activations for r in rr])),
            max_activations=int(np.max([r.activations for r in rr])),
            runs_with_oracle_composition_unsupported_activation=k,
            oracle_composition_unsupported_rate=k / len(rr),
            oracle_composition_wilson95_low=lo, oracle_composition_wilson95_high=hi,
            runs_with_current_regime_below_core_activation=k_regime,
            current_regime_below_core_rate=k_regime / len(rr),
            mean_oracle_composition_unsupported_activations=float(np.mean([
                r.oracle_composition_unsupported_activations for r in rr])),
            mean_current_regime_below_core_activations=float(np.mean([
                r.current_regime_below_core_activations for r in rr])),
            median_final_lambda=float(np.median([r.final_lambda for r in rr])),
            runs_trigger_restricted=sum(r.trigger_restricted for r in rr),
            runs_core_only_response=sum(r.core_dropped for r in rr),
            mean_ai_selections=float(np.mean([r.ai_selections for r in rr])),
            horizon=args.horizon, w=args.w))
    for name, data in (('runs.csv', [vars(r) for r in runs]), ('table_rq3.csv', rows)):
        with (args.out / name).open('w', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(data[0])); wtr.writeheader(); wtr.writerows(data)
    tex = [r'\begin{table}[t]', r'\centering', r'\small',
           (r'\caption{RQ3 sequential experiment with the frozen supervised AI '
            r"proposer. `Below-core' counts runs with at least one activation "
            r'during a regime where $p^{\mathrm{core}}<\lambda_{\mathrm{core}}$. '
            r'The oracle-composition column counts histories with at least one activation below the descriptive sample diagnostic defined in the text; it does '
            r'not count the theorem failure event.}'),
           r'\label{tab:rq3-sequential}', 
           r'\begin{tabular}{@{}lrrrr@{}}',
           r'\toprule',
           r'Rule & \shortstack{Mean\\activations} & \shortstack{Mean\\selections} & Below-core & \shortstack{Oracle\\composition} \\',
           r'\midrule']
    for row in rows:
        gov = 'Aggregate only' if row['governor'] == 'aggregate_only' else 'Joint'
        tex.append(f"{gov} & {row['mean_activations']:.2f} & {row['mean_ai_selections']:.2f} & "
                   f"{row['runs_with_current_regime_below_core_activation']}/{row['replicates']} & "
                   f"{row['runs_with_oracle_composition_unsupported_activation']}/{row['replicates']} \\\\")
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
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
              f"oracle-composition unsupported in {r['runs_with_oracle_composition_unsupported_activation']}/{r['replicates']} runs, "
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








def case_peak_bytes(callable_):
    tracemalloc.start()
    result = callable_()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak


def case_write_ranking_table(audit, output):
    tex=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Case-study proposal pool and frozen AI scores on the selection prefix. The selected candidate is marked with an asterisk. The inadmissible candidate is not scored.}',
        r'\label{tab:case-candidates}',r'\begin{tabular}{@{}llllr@{}}',r'\toprule',
        r'Trigger & Response & Window & Structural check & Predicted margin \\',r'\midrule']
    for row in audit:
        legal=str(row['admissible']).lower()=='true'
        full=str(row['full_trigger']).lower()=='true'
        selected=str(row['selected']).lower()=='true'
        score=f"{float(row['predicted_margin']):.4f}" if row['predicted_margin']!='' else '--'
        tex.append(('$A$' if full else r'$A\land C$')+' & '+('$B$' if legal else r'$\top$')+
            ' & $'+row['window']+('^{*}' if selected else '')+'$ & '+
            ('admissible' if legal else 'rejected')+' & '+score+r' \\')
    tex.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (output/'table_case_candidates.tex').write_text('\n'.join(tex)+'\n')

def case_main(argv=None):
    import json, platform, sys, importlib.metadata
    ap=argparse.ArgumentParser(description='Incremental monitored synthetic alarm case')
    for flag,typ,default in [('triggers',int,30000),('prefix',int,1000),
        ('protected-share',float,.1),('p-core',float,.6),('p-other',float,.99),
        ('threshold',float,.9),('core-threshold',float,.9),('delta-j',float,.025),
        ('repetitions',int,9),('seed',int,20260924),('ai-seed',int,20260930)]:
        ap.add_argument('--'+flag,type=typ,default=default)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv);args.out.mkdir(parents=True,exist_ok=True)
    if args.triggers<2:raise ValueError('at least two alarms are required')
    prefix=min(args.prefix,args.triggers//2)
    alarm,critical,intervention,origins=case_make_trace(args.seed,args.triggers,
        args.protected_share,args.p_core,args.p_other)
    masks=alarm.astype(np.uint8)|critical.astype(np.uint8)*2|intervention.astype(np.uint8)*4
    incumbent=specification(args.threshold,upper=3)
    env={(args.threshold,1,b) for b in (3,4,6,8)}
    # Fixed proposal grammar plus an explicitly injected protected-response fault.
    # Model ranking is learned; syntactic protection is enforced by the governor.
    pool=[specification(args.threshold,upper=b) for b in (4,6,8)]
    pool.append(specification(args.threshold,keep=False,upper=8))
    pool.append(replace(pool[2],core_response=TOP))
    legal=[];audit=[]
    for i,cand in enumerate(pool):
        reasons=revision_rejections(incumbent,cand,env)
        audit.append(dict(candidate=i,window=f'[{cand.lower},{cand.upper}]',
            full_trigger=cand.adaptive_trigger!=BOTTOM,admissible=not reasons,
            rejection=';'.join(reasons),prefix_count=0,prefix_frequency='',predicted_margin='',selected=False))
        if not reasons:legal.append((i,cand,IncrementalMonitor(cand)))
    nstar=int(origins[prefix-1]+8+1)
    counts={i:[0,0,0,0] for i,_,_ in legal}
    for n in range(nstar+1):
        for i,_,mon in legal:
            for x in mon.advance(n,n,int(masks[n])):
                c=counts[i];c[0]+=1;c[1]+=x.outcome
                if x.core:c[2]+=1;c[3]+=x.outcome
    proposer=AggregateAIProposer(args.ai_seed,4000)
    features=[];eligible=[]
    for i,cand,_ in legal:
        nn,ss,nc,sc=counts[i]
        if nn:
            features.append([ss/nn,nn/prefix,args.threshold,
                             float(cand.adaptive_trigger!=BOTTOM),0.])
            eligible.append((i,cand))
            audit[i].update(prefix_count=nn,prefix_frequency=ss/nn)
    scores=proposer.model.predict(np.asarray(features))
    for (i,_),score in zip(eligible,scores):audit[i]['predicted_margin']=float(score)
    selected_pos=int(np.argmax(scores));chosen_index,chosen=eligible[selected_pos]
    audit[chosen_index]['selected']=True
    with (args.out/'candidate_ranking.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    case_write_ranking_table(audit,args.out)

    def baseline_run():
        mon=IncrementalMonitor(incumbent);nall=sall=0
        for n,v in enumerate(masks):
            for x in mon.advance(n,n,int(v)):nall+=1;sall+=x.outcome
        return dict(n=nall,s=sall,peak_pending=mon.peak_pending)

    def governed_run(joint=True):
        gov=Governor(incumbent,env,2*args.delta_j,joint,args.core_threshold)
        evidence=None
        for n,v in enumerate(masks):
            gov.advance(n,n,int(v))
            if n==nstar:
                # All untrusted outputs go through the same executable check.
                for cand in pool:
                    reasons=revision_rejections(gov.active.spec,cand,gov.envelope)
                    if reasons:gov.rejections.append((n,reasons))
                assert gov.select(chosen,n,True)
                evidence=gov.evidence
        return gov,evidence

    aggregate,agg_ev=governed_run(False)
    joint,joint_ev=governed_run(True)
    # If joint certification activates, finish the diagnostic candidate monitor
    # separately, without feeding it into later decisions or reusing the prefix.
    diag=IncrementalMonitor(chosen)
    nn=ss=nc=sc=0
    for n,v in enumerate(masks):
        for x in diag.advance(n,n,int(v)):
            nn+=1;ss+=x.outcome
            if x.core:nc+=1;sc+=x.outcome
    baseline_times=[];governed_times=[]
    for rep in range(args.repetitions):
        # Alternate order to reduce systematic warm-cache/order effects.
        order=('base','gov') if rep%2==0 else ('gov','base')
        for kind in order:
            start=time.perf_counter()
            result=baseline_run() if kind=='base' else governed_run(True)
            (baseline_times if kind=='base' else governed_times).append(time.perf_counter()-start)
    base,base_bytes=case_peak_bytes(baseline_run)
    (measured,ev),gov_bytes=case_peak_bytes(governed_run)
    base_ms=1000*float(np.median(baseline_times));gov_ms=1000*float(np.median(governed_times))
    regression=protocol_tests()
    row=dict(triggers=args.triggers,selection_prefix=prefix,selection_event=nstar,
        trace_events=len(masks),protected_share=float(critical[origins].mean()),
        incumbent_success=base['s']/base['n'],candidate_success=ss/nn,
        candidate_core_success=sc/nc if nc else '',
        proposals_generated=len(pool),structurally_rejected_candidates=len(joint.rejections),
        candidates_ranked=len(eligible),ai_predicted_margin=float(scores[selected_pos]),
        selected_window=f'[{chosen.lower},{chosen.upper}]',
        aggregate_activation_completed=aggregate.decisions[0]['n'] if aggregate.decisions else '',
        aggregate_decision_event=aggregate.decisions[0]['index'] if aggregate.decisions else '',
        aggregate_activation_event=aggregate.activations[0][0] if aggregate.activations else '',
        joint_activation_completed=joint.decisions[0]['n'] if joint.decisions else '',
        joint_final_core_completed=joint_ev.nc,
        origin_version_outcome=regression['origin_version_outcome'],
        retroactive_short_window_outcome=regression['retroactive_short_window_outcome'],
        baseline_median_ms=base_ms,governed_median_ms=gov_ms,
        baseline_us_per_event=1000*base_ms/len(masks),governed_us_per_event=1000*gov_ms/len(masks),
        baseline_peak_kib=base_bytes/1024,governed_peak_kib=gov_bytes/1024,
        baseline_peak_pending=base['peak_pending'],governed_peak_pending=measured.peak_pending,
        parallel_monitor_increment_ms=gov_ms-base_ms,overhead_ratio=gov_ms/base_ms)
    with (args.out/'case_study.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    metadata=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        packages={x:importlib.metadata.version(x) for x in ('numpy','scipy','matplotlib','scikit-learn')},
        arguments=vars(args)|{'out':str(args.out)},timing_repetitions=args.repetitions,
        timing_excludes=['trace generation','model training','prefix ranking','diagnostic monitor'],
        peak_memory='tracemalloc allocations during event-loop call; input trace preallocated',
        baseline_seconds=baseline_times,governed_seconds=governed_times)
    (args.out/'runtime_environment.json').write_text(json.dumps(metadata,indent=2))
    (args.out/'protocol_tests.json').write_text(json.dumps(regression,indent=2))
    results=[('Completed alarm obligations',f'{args.triggers:,}'),
        ('Observed protected share',f"{row['protected_share']:.3f}"),
        ('Candidate aggregate frequency',f'{ss/nn:.3f}'),
        ('Candidate protected frequency',f'{sc/nc:.3f}' if nc else '--'),
        ('Proposals / structurally rejected / ranked',f"{len(pool)} / {len(joint.rejections)} / {len(eligible)}"),
        ('Selected response window',row['selected_window']),
        ('Aggregate-only activation sample',f"{row['aggregate_activation_completed']:,}" if aggregate.decisions else '--'),
        ('Joint activation','activated' if joint.activations else 'not activated'),
        ('Pending outcome: origin / retroactive','1 / 0'),
        ('Baseline / governed median runtime',f'{base_ms:.1f} / {gov_ms:.1f} ms'),
        ('Baseline / governed mean time per event',f"{row['baseline_us_per_event']:.3f} / {row['governed_us_per_event']:.3f} $\\mu$s"),
        ('Baseline / governed peak working memory',f'{base_bytes/1024:.1f} / {gov_bytes/1024:.1f} KiB'),
        ('Baseline / governed peak pending',f"{base['peak_pending']} / {measured.peak_pending}"),
        ('Parallel-governor increment',f'{gov_ms-base_ms:.1f} ms'),
        ('Reference runtime ratio',f'{gov_ms/base_ms:.2f}'+r'$\times$')]
    tex=[r'\begin{table}[t]',r'\centering',r'\small',
         r'\caption{Incremental monitored alarm case study and reference runtime.}',
         r'\label{tab:clinical-case}',r'\begin{tabularx}{\linewidth}{@{}Xr@{}}',r'\toprule',
         r'Quantity & Result \\',r'\midrule']
    tex.extend(a+' & '+b+r' \\' for a,b in results)
    tex.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
    (args.out/'table_case.tex').write_text('\n'.join(tex)+'\n')
    print(json.dumps(row,indent=2))


def case_cli(argv):
    case_main(argv)

def cli():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('rq1','rq2','rq3','case','all','test'))
    parser.add_argument('--out',type=Path,default=Path('results'))
    parser.add_argument('--quick',action='store_true',help='small smoke run, not manuscript numbers')
    args,unknown=parser.parse_known_args()
    runners={'rq1':rq1_cli,'rq2':rq2_cli,'rq3':rq3_cli,'case':case_cli}
    if args.experiment=='test':
        import json
        args.out.mkdir(parents=True,exist_ok=True)
        report=protocol_tests()
        (args.out/'protocol_tests.json').write_text(json.dumps(report,indent=2))
        print(report)
    elif args.experiment=='all':
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
