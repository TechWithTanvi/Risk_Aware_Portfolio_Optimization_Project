"""
data_gen.py: creates the portfolio data for a fictional global automaker, "Nordhavn Motors".

Why made-up data? Real company portfolio data is confidential, and no public dataset contains projects,
costs, benefits, team needs and dependencies together. So the portfolio is synthetic, but the RISK model
in portfolio.py is calibrated to published research (Flyvbjerg & Budzier, HBR 2011: 1,471 IT projects,
average cost overrun of about 27%, and about 1 in 6 projects a "black swan").

It writes three files into data/:
  initiatives.csv   32 candidate projects (cost, duration, benefit, strategic score, people needed...)
  dependencies.csv  which project must finish before another can start
  capacity.csv      how many people each team has, per quarter
"""
import os
import numpy as np, pandas as pd

os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)
SKILLS = ["software", "data_ai", "mechanical", "pmo"]
THEMES = {
    "Electrification":        ["EV battery mgmt software", "Charging network integration", "EV platform cost-down", "Battery supplier dual-sourcing", "800V architecture pilot"],
    "Software-Defined Vehicle": ["OTA update platform", "ADAS perception model v3", "In-car voice assistant", "Vehicle OS consolidation", "Digital twin of drivetrain", "Cybersecurity (UNECE R155)"],
    "Smart Factory":          ["Predictive maintenance – Plant A", "Predictive maintenance – Plant B", "Vision-based quality inspection", "Factory IoT data lake", "Robotic paint line upgrade", "Energy optimization – plants"],
    "Customer & Digital":     ["Online direct-sales store", "Dealer CRM migration", "Connected-car app 2.0", "Subscription services billing", "Customer 360 data product"],
    "Core IT & Compliance":   ["SAP S/4HANA migration – Finance", "SAP S/4HANA migration – Supply", "EU battery passport (regulatory)", "Data governance & catalog", "Cloud cost optimization", "Identity & access modernization"],
    "Supply Chain":           ["Demand forecasting ML", "Supplier risk early-warning", "Inventory optimization – spare parts", "Logistics route optimization"],
}
MANDATORY = {"Cybersecurity (UNECE R155)", "EU battery passport (regulatory)"}

rows, i = [], 0
for theme, names in THEMES.items():
    for name in names:
        i += 1
        complexity = rng.choice(["low", "medium", "high"], p=[0.3, 0.45, 0.25])
        dur = {"low": rng.integers(1, 3), "medium": rng.integers(2, 4), "high": rng.integers(3, 5)}[complexity]  # quarters
        cost = round(float({"low": rng.uniform(1, 3), "medium": rng.uniform(3, 7), "high": rng.uniform(7, 15)}[complexity]), 1)  # $M
        # quarterly benefit ($M/quarter) once delivered; ROI varies a lot (that's realistic)
        benefit_q = round(float(cost * rng.uniform(0.08, 0.35)), 2)
        strategic = int(rng.integers(3, 11))  # 1-10 alignment score (what execs usually rank by)
        fte = {s: 0 for s in SKILLS}
        main = {"Electrification": "mechanical", "Software-Defined Vehicle": "software", "Smart Factory": "data_ai",
                "Customer & Digital": "software", "Core IT & Compliance": "software", "Supply Chain": "data_ai"}[theme]
        base = cost / dur * 4.0  # FTE per quarter roughly proportional to spend rate
        fte[main] = round(base * 0.6, 1)
        for s in SKILLS:
            if s != main:
                fte[s] = round(base * rng.uniform(0.05, 0.25), 1)
        if "ML" in name or "model" in name or "Predictive" in name or "optimization" in name.lower() or "data" in name.lower():
            fte["data_ai"] = round(fte["data_ai"] + base * 0.3, 1)
        rows.append(dict(id=f"P{i:02d}", name=name, theme=theme, complexity=complexity, duration_q=int(dur),
                         cost_musd=cost, benefit_musd_per_q=benefit_q, strategic_score=strategic,
                         mandatory=name in MANDATORY, **{f"fte_{s}": fte[s] for s in SKILLS}))

df = pd.DataFrame(rows)
# Dependencies (predecessor must finish before successor starts)
deps = [("Factory IoT data lake", "Predictive maintenance – Plant A"), ("Factory IoT data lake", "Predictive maintenance – Plant B"),
        ("Factory IoT data lake", "Energy optimization – plants"), ("OTA update platform", "ADAS perception model v3"),
        ("Vehicle OS consolidation", "In-car voice assistant"), ("Customer 360 data product", "Connected-car app 2.0"),
        ("SAP S/4HANA migration – Finance", "SAP S/4HANA migration – Supply"), ("Data governance & catalog", "Customer 360 data product"),
        ("SAP S/4HANA migration – Supply", "Inventory optimization – spare parts"), ("Battery supplier dual-sourcing", "EU battery passport (regulatory)")]
name2id = dict(zip(df.name, df.id))
pd.DataFrame([(name2id[a], name2id[b]) for a, b in deps], columns=["predecessor", "successor"]).to_csv("data/dependencies.csv", index=False)
df.to_csv("data/initiatives.csv", index=False)

# Quarterly capacity (FTE) per skill pool over an 8-quarter plan horizon
cap = pd.DataFrame({"quarter": range(8), "software": 42, "data_ai": 20, "mechanical": 22, "pmo": 10})
cap.to_csv("data/capacity.csv", index=False)
print(f"\n{len(df)} initiatives, total ask ${df.cost_musd.sum():.0f}M")
