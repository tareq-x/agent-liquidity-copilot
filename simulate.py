import pandas as pd
from rules import recommend_rebalance, LOOKAHEAD_HOURS

# ---------- Settings ----------
OPENING_MULTIPLIER = 6   # opening cash = 6 x the agent's average hourly demand
DELAY_HOURS = 1          # a rebalance ordered now arrives this many hours later (keep >= 1)

# ---------- Load data ----------
preds = pd.read_csv("data/test_predictions.csv", parse_dates=["timestamp"])
full = pd.read_csv("data/synthetic_cashout.csv", parse_dates=["timestamp"])
preds["date"] = preds["timestamp"].dt.date

# Opening cash per agent, based on the training period only
cutoff = preds["timestamp"].min().normalize()
avg_hourly = full[full["timestamp"] < cutoff].groupby("agent_id")["cashout_amount"].mean()


def simulate_day(day, opening_cash, forecast_col):
    """Replay one agent's day. forecast_col=None means no rebalancing."""
    cash = opening_cash
    pending = []  # orders on the way: (arrival_hour_index, amount)
    stockout_hours = 0
    unmet = 0
    ordered = 0
    orders = 0

    demands = day["cashout_amount"].tolist()
    forecasts = day[forecast_col].tolist() if forecast_col else None

    for i, demand in enumerate(demands):
        # 1. Deliveries that arrive now
        cash += sum(a for t, a in pending if t <= i)
        pending = [(t, a) for t, a in pending if t > i]

        # 2. Decide whether to order more cash
        if forecast_col and i + DELAY_HOURS < len(demands):  # no point ordering if it arrives after closing
            in_transit = sum(a for t, a in pending)
            rec = recommend_rebalance(cash + in_transit, forecasts[i : i + LOOKAHEAD_HOURS])
            if rec["amount"] > 0:
                pending.append((i + DELAY_HOURS, rec["amount"]))
                ordered += rec["amount"]
                orders += 1

        # 3. Serve this hour's customers
        if demand > cash:
            stockout_hours += 1
            unmet += demand - cash
            cash = 0
        else:
            cash -= demand

    return stockout_hours, unmet, ordered, orders


strategies = {
    "No rebalancing": None,
    "Rules + baseline forecast": "baseline_pred",
    "Rules + LightGBM forecast": "model_pred",
}

total_hours = len(preds)
total_demand = preds["cashout_amount"].sum()

results = []
for name, col in strategies.items():
    s_total = u_total = o_total = n_total = 0
    for (agent, date), day in preds.groupby(["agent_id", "date"]):
        day = day.sort_values("timestamp")
        s, u, o, n = simulate_day(day, OPENING_MULTIPLIER * avg_hourly[agent], col)
        s_total += s
        u_total += u
        o_total += o
        n_total += n

    results.append({
        "strategy": name,
        "stockout_hours": s_total,
        "stockout_pct": round(s_total / total_hours * 100, 1),
        "unmet_bdt": round(u_total),
        "service_level_pct": round((1 - u_total / total_demand) * 100, 1),
        "cash_moved_bdt": round(o_total),
        "rebalances": n_total,
    })

res = pd.DataFrame(results)
print(res.to_string(index=False))

# ---------- Headline numbers ----------
none = res.loc[0, "stockout_hours"]
base = res.loc[1, "stockout_hours"]
model = res.loc[2, "stockout_hours"]
print()
print(f"LightGBM + rules cuts stockout hours by {(1 - model / none) * 100:.0f}% vs no rebalancing")
if base > 0:
    print(f"LightGBM + rules cuts stockout hours by {(1 - model / base) * 100:.0f}% vs baseline-forecast rules")

res.to_csv("data/simulation_results.csv", index=False)
print("\nSaved data/simulation_results.csv")