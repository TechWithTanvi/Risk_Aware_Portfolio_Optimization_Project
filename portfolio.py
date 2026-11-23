"""
portfolio.py: the "engine" of the project.

It contains five building blocks. The notebook and the MCP server both use them.

    load()                  read the three data files
    sample_multipliers()    random "what really happens" numbers (Monte Carlo)
    simulate_plan()         test a plan in 5,000 imagined futures
    greedy_by_score()       the common method: fund the highest strategic scores first
    optimize()              the optimizer: math picks the best projects AND start quarters
    optimize_risk_aware()   the optimizer plus a safety rule: at most X% chance of going over budget
    greedy_risk_aware()     the common method held to the same safety rule (for a fair comparison)

Units used everywhere:
    money  = millions of US dollars ($M)
    time   = quarters (1 quarter = 3 months). The plan covers quarters 0..7 (2 years).
    people = FTE (full-time employees)
"""
import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

# ----------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------
SKILLS = ["software", "data_ai", "mechanical", "pmo"]   # the four teams
PLAN_Q = 8          # every project must start AND finish inside quarters 0..7 (2 years)
BENEFIT_H = 12      # we count benefits until quarter 12 (a 3-year view)

# How wrong estimates usually are. Calibrated to Flyvbjerg & Budzier (Harvard Business Review, 2011):
# 1,471 IT projects, average cost overrun of about 27%, and about 1 in 6 projects a "black swan" (huge overrun).
SIGMA  = {"low": 0.10, "medium": 0.18, "high": 0.28}    # normal uncertainty, bigger for complex projects
TAIL_P = {"low": 0.08, "medium": 0.15, "high": 0.25}    # chance of a black swan (about 1 in 6 on average)


def load(path="data"):
    """Read the three CSV files made by data_gen.py."""
    df = pd.read_csv(f"{path}/initiatives.csv").set_index("id")     # one row per project
    deps = pd.read_csv(f"{path}/dependencies.csv")                  # predecessor must finish before successor starts
    cap = pd.read_csv(f"{path}/capacity.csv").set_index("quarter")  # people available per team per quarter
    return df, deps, cap


# ----------------------------------------------------------------------------------------
# 1) MONTE CARLO: imagine thousands of possible futures
# ----------------------------------------------------------------------------------------
def sample_multipliers(df, n_sims=5000, seed=0, corr=0.10):
    """
    For every imagined future (row) and every project (column), draw:
        cost multiplier      actual cost     = estimate x multiplier
        duration multiplier  actual duration = plan     x multiplier
    Three ingredients:
        1. normal uncertainty (lognormal: always positive, longer tail to the right)
        2. black swans: with a small chance, cost x 1.8-3.0 and duration x 1.4-1.9
        3. a shared shock g: one number per future that pushes ALL projects up or down together
           (inflation, a talent shortage...). It makes overruns "correlated", as they are in real life.
    Returns two arrays shaped (n_sims, number of projects), like an Excel sheet of 5,000 rows.
    """
    rng = np.random.default_rng(seed)            # seed = repeatable "random" numbers
    n = len(df)
    sig = df.complexity.map(SIGMA).values
    tail = df.complexity.map(TAIL_P).values
    g = rng.normal(0, corr, size=(n_sims, 1))    # shared shock: same value for every project in a row
    cost_m = np.exp(0.01 + sig * rng.normal(0, 1, size=(n_sims, n)) + g)
    dur_m = np.exp(0.08 + sig * rng.normal(0, 1, size=(n_sims, n)) + g)
    swan = rng.random((n_sims, n)) < tail        # black-swan dice roll for every cell
    cost_m = np.where(swan, cost_m * rng.uniform(1.8, 3.0, (n_sims, n)), cost_m)
    dur_m = np.where(swan, dur_m * rng.uniform(1.4, 1.9, (n_sims, n)), dur_m)
    return cost_m, dur_m


def risk_profile(df, n_sims=5000, seed=0):
    """Per-project summary of the simulated futures: typical (P50), cautious (P80) and average cost and duration."""
    cm, dm = sample_multipliers(df, n_sims, seed)
    out = df[["name", "cost_musd", "duration_q"]].copy()
    out["cost_p50"] = np.percentile(cm, 50, axis=0) * df.cost_musd.values
    out["cost_p80"] = np.percentile(cm, 80, axis=0) * df.cost_musd.values
    out["dur_p80"] = np.ceil(np.percentile(dm, 80, axis=0) * df.duration_q.values - 1e-9).astype(int)
    out["cost_mean"] = cm.mean(axis=0) * df.cost_musd.values          # "realistic" average cost
    out["dur_mean"] = np.ceil(dm.mean(axis=0) * df.duration_q.values - 0.25).astype(int)  # rounded up if > 1/4 quarter over
    out["mean_cost_overrun_%"] = (cm.mean(axis=0) - 1) * 100
    return out


def simulate_plan(df, deps, plan, budget, n_sims=5000, seed=1):
    """
    Test a plan (a list of projects + their start quarters) in 5,000 imagined futures.
    In each future:
      - every project gets a random actual cost and duration
      - a project cannot start before its predecessors ACTUALLY finish (delays ripple down)
      - benefit = benefit per quarter x the number of quarters between its finish and quarter 12
      - net value = total benefit - total actual cost
    Returns a summary:
      prob_over_budget   share of futures where total actual cost > budget
      cost_p50/p80       typical / cautious total cost
      net_value_p10/50/90  bad case / typical / good case value
    """
    ids = list(plan.index)
    sub = df.loc[ids]
    cm, dm = sample_multipliers(sub, n_sims, seed)
    cost = cm * sub.cost_musd.values
    dur = dm * sub.duration_q.values
    pos = {i: k for k, i in enumerate(ids)}
    preds = (deps[deps.successor.isin(ids) & deps.predecessor.isin(ids)]
             .groupby("successor").predecessor.apply(list).to_dict())
    finish = np.zeros_like(dur)
    for i in sorted(ids, key=lambda i: plan.loc[i, "start_q"]):
        k = pos[i]
        start = np.full(n_sims, float(plan.loc[i, "start_q"]))
        for p in preds.get(i, []):                       # wait for predecessors to really finish
            start = np.maximum(start, finish[:, pos[p]])
        finish[:, k] = start + dur[:, k]
    benefit = (np.clip(BENEFIT_H - finish, 0, None) * sub.benefit_musd_per_q.values).sum(axis=1)
    total_cost = cost.sum(axis=1)
    planned_finish = plan.start_q.values + sub.duration_q.values
    late = (finish - planned_finish) > 1.0               # more than one quarter late
    return {
        "n_projects": len(ids),
        "planned_cost": round(float(sub.cost_musd.sum()), 1),
        "cost_p50": round(float(np.percentile(total_cost, 50)), 1),
        "cost_p80": round(float(np.percentile(total_cost, 80)), 1),
        "prob_over_budget": round(float((total_cost > budget).mean()), 3),
        "net_value_p10": round(float(np.percentile(benefit - total_cost, 10)), 1),
        "net_value_p50": round(float(np.percentile(benefit - total_cost, 50)), 1),
        "net_value_p90": round(float(np.percentile(benefit - total_cost, 90)), 1),
        "prob_late_by_project": dict(zip(ids, np.round(late.mean(axis=0), 2).tolist())),
        "_total_cost": total_cost,            # all 5,000 simulated costs (for charts)
        "_net_value": benefit - total_cost,   # all 5,000 simulated values (for charts)
    }


# ----------------------------------------------------------------------------------------
# 2) THE COMMON METHOD: rank by strategic score (a "greedy heuristic")
# ----------------------------------------------------------------------------------------
def greedy_by_score(df, deps, cap, budget):
    """
    How many companies decide today:
      1. fund the legally required projects (and anything they depend on) first
      2. go down the list by strategic score; fund a project if it still fits in the budget
      3. start each one in the earliest quarter where every team has enough free people
    Fast and reasonable, but it looks at one project at a time, never at combinations.
    """
    load_ = {k: np.zeros(PLAN_Q) for k in SKILLS}        # people already booked, per team per quarter
    chosen, spent = {}, 0.0
    preds = deps.groupby("successor").predecessor.apply(list).to_dict()
    must = set(df.index[df.mandatory])                   # required projects + their prerequisites
    stack = list(must)
    while stack:
        for p in preds.get(stack.pop(), []):
            if p not in must:
                must.add(p); stack.append(p)
    order = df.assign(_must=df.index.isin(must)).sort_values(["_must", "strategic_score"], ascending=False).index
    for _ in range(2):                                   # 2nd pass picks up items whose prerequisites came later
        for i in order:
            if i in chosen or spent + df.loc[i, "cost_musd"] > budget:
                continue
            ps = preds.get(i, [])
            if any(p not in chosen for p in ps):
                continue
            d = int(df.loc[i, "duration_q"])
            earliest = max([chosen[p] + int(df.loc[p, "duration_q"]) for p in ps], default=0)
            for s in range(earliest, PLAN_Q - d + 1):
                if all(load_[k][s:s + d].max() + df.loc[i, f"fte_{k}"] <= cap[k].iloc[s:s + d].min() for k in SKILLS):
                    for k in SKILLS:
                        load_[k][s:s + d] += df.loc[i, f"fte_{k}"]
                    chosen[i], spent = s, spent + df.loc[i, "cost_musd"]
                    break
    plan = pd.DataFrame({"start_q": chosen}).rename_axis("id")
    plan["name"] = df.loc[plan.index, "name"]
    plan["planned_cost"] = df.loc[plan.index, "cost_musd"]
    return {"status": "HEURISTIC", "plan": plan.sort_values("start_q")}


# ----------------------------------------------------------------------------------------
# 3) THE OPTIMIZER (integer programming with Google OR-Tools CP-SAT)
# ----------------------------------------------------------------------------------------
def optimize(df, deps, cap, budget, risk_level="plan", strategic_weight=0.3, capacity_factor=None,
             force_in=(), force_out=(), time_limit=20):
    """
    DECISION VARIABLES  x[i, s] = 1 means "fund project i AND start it in quarter s" (yes/no switches)
    CONSTRAINTS (rules)
        1. budget      total cost of funded projects <= budget
        2. capacity    in every quarter, for every team: people needed <= people available
        3. dependency  if B needs A: B funded -> A funded, and B starts after A finishes
        4. mandatory   legally required projects must be funded
        5. time        every project finishes inside the 2-year window
    OBJECTIVE (what "best" means)
        maximize  sum of [benefit per quarter x quarters of benefit before Q12 - cost + 0.3 x strategic score]
        -> starting a valuable project EARLIER earns more quarters of benefit.
    risk_level="plan"      trust the teams' estimates
    risk_level="expected"  use realistic costs and durations (average of the simulated futures)
    capacity_factor        e.g. {"data_ai": 0.75} = the data/AI team is 25% smaller (for what-if scenarios)
    force_in / force_out   project ids that must / must not be in the plan (for what-if scenarios)
    """
    capacity_factor = capacity_factor or {}
    if risk_level == "expected":
        rp = risk_profile(df)
        cost, dur = rp.cost_mean, rp.dur_mean
    else:
        cost, dur = df.cost_musd, df.duration_q
    m = cp_model.CpModel()
    SC = 100   # CP-SAT needs whole numbers: work in units of $0.01M ($10K), like using cents instead of dollars
    x, start, sel = {}, {}, {}
    for i in df.index:
        d = int(dur[i])
        for s in range(0, max(0, PLAN_Q - d) + 1):           # only start quarters that still finish in time (rule 5)
            x[i, s] = m.NewBoolVar(f"x_{i}_{s}")
        starts = [s for (j, s) in x if j == i]
        sel[i] = m.NewBoolVar(f"sel_{i}")                     # sel = 1 if the project is funded at all
        m.Add(sum(x[i, s] for s in starts) == sel[i])         # at most one start quarter per project
        start[i] = m.NewIntVar(0, PLAN_Q, f"start_{i}")
        m.Add(start[i] == sum(s * x[i, s] for s in starts))
        if df.loc[i, "mandatory"] or i in force_in:           # rule 4
            m.Add(sel[i] == 1)
        if i in force_out:
            m.Add(sel[i] == 0)
    # rule 1: budget
    m.Add(sum(int(round(cost[i] * SC)) * sel[i] for i in df.index) <= int(budget * SC))
    # rule 2: capacity, every quarter t and every team k
    for t in range(PLAN_Q):
        for k in SKILLS:
            capk = cap.loc[t, k] * capacity_factor.get(k, 1.0)
            terms = [int(round(df.loc[i, f"fte_{k}"] * 10)) * x[i, s]
                     for (i, s) in x if s <= t < s + int(dur[i])]
            if terms:
                m.Add(sum(terms) <= int(capk * 10))
    # rule 3: dependencies
    for _, r in deps.iterrows():
        p, q = r.predecessor, r.successor
        m.Add(sel[q] <= sel[p])
        m.Add(start[q] >= start[p] + int(dur[p])).OnlyEnforceIf(sel[q])
    # objective
    obj = []
    for (i, s) in x:
        q_benefit = max(0, BENEFIT_H - (s + int(dur[i])))
        v = df.loc[i, "benefit_musd_per_q"] * q_benefit - cost[i] + strategic_weight * df.loc[i, "strategic_score"]
        obj.append(int(round(v * SC)) * x[i, s])
    m.Maximize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "plan": pd.DataFrame()}
    plan = pd.DataFrame([(i, s) for (i, s) in x if solver.Value(x[i, s])], columns=["id", "start_q"]).set_index("id")
    plan["name"] = df.loc[plan.index, "name"]
    plan["planned_cost"] = df.loc[plan.index, "cost_musd"]
    return {"status": solver.StatusName(status), "plan": plan.sort_values("start_q")}


# ----------------------------------------------------------------------------------------
# 4) RISK-AWARE PLANNING: the best plan with at most X% chance of going over budget
# ----------------------------------------------------------------------------------------
def optimize_risk_aware(df, deps, cap, budget, max_overrun_prob=0.20, rounds=9, verbose=False, **kw):
    """
    A "guess the number" search (bisection) for the PLANNING BUDGET:
      1. guess a planning budget (start halfway between 60% and 100% of the real budget)
      2. optimize with REALISTIC costs inside that planning budget
      3. simulate the plan against the REAL budget: over budget in <= max_overrun_prob of futures?
            yes -> PASS, guess higher next time (we can afford more)
            no  -> FAIL, guess lower next time (too risky)
      4. repeat; keep the plan from the highest guess that passed
    Real budget - final planning budget = CONTINGENCY RESERVE (money kept aside for overruns).
    """
    lo, hi, best = 0.6 * budget, budget, None
    for k in range(1, rounds + 1):
        guess = (lo + hi) / 2
        r = optimize(df, deps, cap, guess, risk_level="expected", **kw)
        if r["plan"].empty:
            lo = guess
            continue
        s = simulate_plan(df, deps, r["plan"], budget)
        ok = s["prob_over_budget"] <= max_overrun_prob
        if verbose:
            print(f"Round {k}: plan for ${guess:6.2f}M -> {s['n_projects']:2d} projects | over ${budget:.0f}M in "
                  f"{s['prob_over_budget']:5.1%} of futures -> {'PASS, guess higher' if ok else 'FAIL, guess lower'}")
        if ok:
            best, lo = (r, guess), guess
        else:
            hi = guess
    if best is None:
        return {"status": "INFEASIBLE", "plan": pd.DataFrame()}
    r, guess = best
    r["planning_budget"] = round(guess, 1)
    r["contingency_reserve"] = round(budget - guess, 1)
    return r


def greedy_risk_aware(df, deps, cap, budget, max_overrun_prob=0.20, rounds=8):
    """The common rank-by-score method, held to the SAME safety rule (same guessing game).
    This makes a fair comparison with optimize_risk_aware(): equal safety, different method."""
    lo, hi, best = 0.4 * budget, budget, None
    for _ in range(rounds):
        guess = (lo + hi) / 2
        r = greedy_by_score(df, deps, cap, guess)
        if simulate_plan(df, deps, r["plan"], budget)["prob_over_budget"] <= max_overrun_prob:
            best, lo = r, guess
        else:
            hi = guess
    return best
