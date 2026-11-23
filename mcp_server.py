"""
mcp_server.py (auto-written by the notebook, Step 10): makes the engine usable by AI assistants through MCP (Model Context Protocol).

What is this?
  An AI assistant (an LLM such as Claude) is great at understanding questions, but it should not do
  optimization math in its head: it could make numbers up. MCP is an open standard that lets an AI app
  discover and call real tools. This file turns five Python functions into MCP "tools".
  A manager can then ask in plain English:
      "We just lost $15M of budget. What should we fund, and what's our risk of overrunning?"
  The AI calls optimize_portfolio(budget_musd=85), our solver computes the answer, and the AI explains it.

Run it:     python mcp_server.py        (it waits quietly for an AI app to connect over "stdio")
Test it:    Step 10.4 of the notebook (it acts like an AI app)
"""
import json
import pandas as pd
from mcp.server.fastmcp import FastMCP
from portfolio_engine import load, optimize_risk_aware, simulate_plan, risk_profile

mcp = FastMCP("portfolio-intelligence", log_level="ERROR")   # server name as AI apps see it; only log errors
DF, DEPS, CAP = load()                     # load the portfolio data once when the server starts


def _summary(sim):
    """Remove the big raw arrays (5,000 numbers) before sending results to the AI."""
    return {k: v for k, v in sim.items() if not k.startswith("_")}


# Each @mcp.tool() below becomes a tool. The AI reads the function NAME, the INPUT TYPES and the
# DOCSTRING (the text in triple quotes) to decide when and how to call it, so the docstrings are written for the AI.

@mcp.tool()
def list_initiatives(theme: str | None = None) -> str:
    """List candidate projects with cost ($M), duration (quarters), quarterly benefit ($M),
    strategic score (1-10), theme and whether it is legally required. Optionally filter by theme."""
    d = DF if theme is None else DF[DF.theme.str.contains(theme, case=False)]
    cols = ["name", "theme", "cost_musd", "duration_q", "benefit_musd_per_q", "strategic_score", "mandatory"]
    return d[cols].reset_index().to_json(orient="records")


@mcp.tool()
def project_risk(project_ids: list[str]) -> str:
    """Monte Carlo risk profile for specific projects (ids like "P30"): average overrun %,
    typical (P50) and cautious (P80) cost, and cautious (P80) duration."""
    rp = risk_profile(DF).loc[project_ids, ["name", "cost_musd", "cost_p50", "cost_p80",
                                            "duration_q", "dur_p80", "mean_cost_overrun_%"]]
    return rp.round(2).reset_index().to_json(orient="records")


@mcp.tool()
def simulate_portfolio(project_ids: list[str], start_quarters: list[int], budget_musd: float) -> str:
    """Stress-test a proposed plan (project ids + start quarter for each) in 5,000 simulated futures.
    Returns the chance of exceeding the budget, typical and cautious cost, the value range
    (bad case P10, typical P50, good case P90) and each project's chance of slipping more than a quarter."""
    plan = pd.DataFrame({"start_q": start_quarters}, index=pd.Index(project_ids, name="id"))
    return json.dumps(_summary(simulate_plan(DF, DEPS, plan, budget_musd)))


@mcp.tool()
def optimize_portfolio(budget_musd: float, max_overrun_prob: float = 0.2,
                       must_include: list[str] = [], must_exclude: list[str] = [],
                       capacity_change: dict[str, float] = {}) -> str:
    """Find the most valuable set of projects and their start quarters (OR-Tools CP-SAT solver),
    respecting budget, team capacity (teams: software, data_ai, mechanical, pmo), dependencies and
    legally required projects, with at most `max_overrun_prob` chance that actual cost exceeds the budget.
    capacity_change example: {"data_ai": 0.75} means the data/AI team is 25% smaller."""
    r = optimize_risk_aware(DF, DEPS, CAP, budget_musd, max_overrun_prob,
                            force_in=tuple(must_include), force_out=tuple(must_exclude),
                            capacity_factor=capacity_change)
    if r["plan"].empty:
        return json.dumps({"status": "INFEASIBLE",
                           "hint": "Increase the budget, accept more risk, or remove must-include projects."})
    sim = _summary(simulate_plan(DF, DEPS, r["plan"], budget_musd))
    return json.dumps({"status": r["status"], "contingency_reserve_musd": r["contingency_reserve"],
                       "plan": r["plan"].reset_index()[["id", "name", "start_q", "planned_cost"]].to_dict("records"),
                       "risk": sim})


@mcp.tool()
def compare_scenarios(scenarios: list[dict]) -> str:
    """Run several what-if scenarios side by side. Each scenario is a dict with keys: name, budget_musd and
    optionally max_overrun_prob, must_include, must_exclude, capacity_change.
    Returns, per scenario: number of projects, typical value, bad-case value and chance of going over budget."""
    out = []
    for s in scenarios:
        r = json.loads(optimize_portfolio(s["budget_musd"], s.get("max_overrun_prob", 0.2), s.get("must_include", []),
                                          s.get("must_exclude", []), s.get("capacity_change", {})))
        if r["status"] == "INFEASIBLE":
            out.append({"name": s.get("name"), "status": "INFEASIBLE"})
            continue
        out.append({"name": s.get("name"), "n_projects": len(r["plan"]), "net_value_p50": r["risk"]["net_value_p50"],
                    "net_value_p10": r["risk"]["net_value_p10"], "prob_over_budget": r["risk"]["prob_over_budget"]})
    return json.dumps(out)


if __name__ == "__main__":
    mcp.run()   # start the server and wait for an AI app to connect
