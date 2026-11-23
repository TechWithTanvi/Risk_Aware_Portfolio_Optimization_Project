"""
charts.py: all the project's charts, each designed to answer ONE question.

Colors mean the same thing in every chart:
    orange = the common method (rank by strategic score)
    blue   = the optimizer
    red    = over budget (bad)
"""
import numpy as np
import matplotlib.pyplot as plt

ORANGE, BLUE, RED, GREY = "#eb6834", "#2a78d6", "#e34948", "#8a8984"
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e0"
THEME_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10.5,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# Short display names for the four plans
LABELS = {
    "A": "A · Rank by score",
    "B": "B · Optimizer",
    "A2": "A2 · Rank by score + safety rule",
    "C": "C · Optimizer + safety rule",
}
METHOD_COLOR = {"A": ORANGE, "A2": ORANGE, "B": BLUE, "C": BLUE}


def chart_risk_and_value(sims, risk_limit=0.20, save=None):
    """Question: how risky is each plan, and how much value does it create?
    Left panel: chance of going over budget. Right panel: typical 3-year value. Same plans, same order."""
    order = ["A", "B", "A2", "C"]
    y = np.array([3.6, 2.6, 1.0, 0.0])                    # small gap between the two groups
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    risk = [sims[k]["prob_over_budget"] * 100 for k in order]
    val = [sims[k]["net_value_p50"] for k in order]
    cols = [METHOD_COLOR[k] for k in order]
    ax1.barh(y, risk, color=cols, height=0.7)
    ax1.axvline(risk_limit * 100, color=INK, ls="--", lw=1.2)
    ax1.set_ylim(-0.75, 4.15)
    ax1.text(risk_limit * 100 + 1.5, -0.6, f"safety limit {risk_limit:.0%}", color=INK, fontsize=9)
    for yi, r in zip(y, risk):
        ax1.text(r + 1.5, yi, f"{r:.0f}%", va="center", color=INK, fontweight="bold")
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Chance of going over the budget (% of 5,000 futures)")
    ax1.set_title("How risky is the plan?  (shorter bar = safer)", loc="left")
    ax2.barh(y, val, color=cols, height=0.7)
    for yi, v in zip(y, val):
        ax2.text(v + 0.6, yi, f"${v:.1f}M", va="center", color=INK, fontweight="bold")
    ax2.set_xlim(0, max(val) * 1.25)
    ax2.set_xlabel("Typical 3-year net value, $M (median of 5,000 futures)")
    ax2.set_title("How much value does it create?  (longer bar = better)", loc="left")
    ax1.set_yticks(y, [LABELS[k] for k in order])
    ax1.text(-0.02, 3.1, "No safety rule", transform=ax1.get_yaxis_transform(), ha="right", color=MUTED, fontsize=9, style="italic")
    ax1.text(-0.02, 0.5, "With safety rule", transform=ax1.get_yaxis_transform(), ha="right", color=MUTED, fontsize=9, style="italic")
    for ax in (ax1, ax2):
        ax.axhline(1.8, color=GRID, lw=1.5)
    fig.tight_layout()
    if save: fig.savefig(save, bbox_inches="tight")
    plt.show()


def chart_cost_futures(sims, budget, save=None):
    """Question: in the 5,000 imagined futures, what does each plan REALLY cost?
    One small panel per plan. The red part of each hill = futures that went over budget."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True, sharey=True)
    layout = [("A", 0, 0), ("B", 0, 1), ("A2", 1, 0), ("C", 1, 1)]
    allc = np.concatenate([sims[k]["_total_cost"] for k, _, _ in layout])
    bins = np.linspace(np.percentile(allc, 0.2), np.percentile(allc, 99.5), 70)
    for k, r, c in layout:
        ax = axes[r, c]
        costs = sims[k]["_total_cost"]
        counts, edges = np.histogram(costs, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        under = centers <= budget
        ax.bar(centers[under], counts[under], width=width * 0.92, color=METHOD_COLOR[k])
        ax.bar(centers[~under], counts[~under], width=width * 0.92, color=RED)
        ax.axvline(budget, color=INK, ls="--", lw=1.2)
        med = np.median(costs)
        ax.set_title(LABELS[k], loc="left")
        ax.text(0.98, 0.92, f"{(costs > budget).mean():.0%} of futures over budget", transform=ax.transAxes,
                ha="right", color=RED, fontweight="bold")
        ax.text(0.98, 0.80, f"typical cost ${med:.0f}M", transform=ax.transAxes, ha="right", color=INK)
        ax.grid(axis="x", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Actual total cost in one imagined future ($M)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Number of futures")
    axes[0, 0].text(budget, axes[0, 0].get_ylim()[1] * 0.98, f" ${budget:.0f}M budget", color=INK, va="top", fontsize=9)
    fig.suptitle("What does each plan really cost?  Each hill = 5,000 imagined futures. Red = over budget.",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    if save: fig.savefig(save, bbox_inches="tight")
    plt.show()


def chart_change_vs_base(table, title, save=None):
    """Question: how much typical value does each what-if decision gain or lose compared with the Base plan?"""
    t = table.drop(index=table.index[0])                  # first row is the Base itself
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(t) + 1.4))
    y = np.arange(len(t))[::-1]
    vals = t["change_vs_base ($M)"].values
    ax.barh(y, vals, color=[BLUE if v >= 0 else GREY for v in vals], height=0.6)
    ax.axvline(0, color=INK, lw=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{'+' if v > 0 else ''}{v:.1f}M", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontweight="bold")
    ax.set_yticks(y, t.index)
    lim = max(abs(vals).max() * 1.35, 2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Change in typical 3-year value vs. the Base plan ($M)   ← loses value  |  gains value →")
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    if save: fig.savefig(save, bbox_inches="tight")
    plt.show()


def chart_frontier(fr, current_budget=100, save=None):
    """Question: if we had a bigger or smaller budget, how much value could we get?
    Left: typical value at each budget (line) with the bad-to-good range (shaded).
    Right: the EXTRA value bought by each budget step."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    ax1.fill_between(fr.budget, fr.p10, fr.p90, color=BLUE, alpha=0.15, lw=0, label="range: bad case (P10) to good case (P90)")
    ax1.plot(fr.budget, fr.p50, "o-", color=BLUE, lw=2, ms=7, label="typical value (P50)")
    for b, v in zip(fr.budget, fr.p50):
        ax1.annotate(f"${v:.0f}M", (b, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax1.axvline(current_budget, color=MUTED, ls=":", lw=1.2)
    ax1.text(current_budget + 1, ax1.get_ylim()[0] + 2, "today's budget", color=MUTED, fontsize=9)
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.set_xlabel("Budget ($M)")
    ax1.set_ylabel("3-year net value ($M)")
    ax1.set_title("Value at each budget (all plans keep ≤20% overrun risk)", loc="left")
    ax1.legend(loc="upper left", fontsize=8.5, frameon=False)
    steps = fr.budget.values[1:]
    gain = np.diff(fr.p50.values)
    labels = [f"{a:.0f}→{b:.0f}" for a, b in zip(fr.budget.values[:-1], steps)]
    ax2.bar(labels, gain, color=[BLUE if g > 0 else GREY for g in gain], width=0.65)
    for i, g in enumerate(gain):
        ax2.text(i, g + (0.3 if g >= 0 else -0.3), f"{'+' if g > 0 else ''}{g:.1f}", ha="center",
                 va="bottom" if g >= 0 else "top", fontsize=9, color=INK, fontweight="bold")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.set_xlabel(r"Budget step (\$M → \$M)")
    ax2.set_ylim(min(gain.min(), 0) * 1.35 - 0.5, gain.max() * 1.2)
    ax2.set_ylabel("Extra typical value from this step ($M)")
    ax2.set_title("What does each extra slice of budget buy?", loc="left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    if save: fig.savefig(save, bbox_inches="tight")
    plt.show()


def chart_roadmap(plan, df, deps, save=None):
    """Question: which projects do we do, and when does each one start and finish?"""
    p = plan.join(df[["theme", "duration_q", "mandatory"]]).sort_values(["start_q", "theme"])
    themes = list(df.theme.unique())
    colors = {t: THEME_COLORS[i % len(THEME_COLORS)] for i, t in enumerate(themes)}
    preds = deps.groupby("successor").predecessor.apply(list).to_dict()
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(p) + 1.8))
    for yi, (i, r) in enumerate(p.iterrows()):
        ax.barh(yi, r.duration_q, left=r.start_q, color=colors[r.theme], height=0.62)
        note = []
        if r.mandatory:
            note.append("★ required by law")
        need = [q for q in preds.get(i, []) if q in p.index]
        if need:
            note.append("after " + ", ".join(need))
        label = f"{i}  {r['name']}" + (f"   ({'; '.join(note)})" if note else "")
        ax.text(r.start_q + r.duration_q + 0.08, yi, label, va="center", fontsize=9, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.set_yticks([])
    ax.set_xlim(0, 12)
    ax.set_xticks(np.arange(8) + 0.5, [f"Y{q // 4 + 1} Q{q % 4 + 1}" for q in range(8)])
    for q in range(9):
        ax.axvline(q, color=GRID, lw=0.8, zorder=0)
    ax.axvline(4, color=MUTED, lw=1.2, zorder=0)
    ax.grid(False)
    ax.invert_yaxis()
    ax.set_title("Roadmap: each bar is a project, from the quarter it starts to the quarter it ends", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[t]) for t in themes]
    ax.legend(handles, themes, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=9)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    if save: fig.savefig(save, bbox_inches="tight")
    plt.show()
