import pandas as pd
from rules import recommend_rebalance, LOOKAHEAD_HOURS

df = pd.read_csv("data/test_predictions.csv", parse_dates=["timestamp"])

AGENT = "AG001"
DAY = "2026-09-07"        # a salary-period day, so demand is high
OPENING_CASH = 120_000

day = df[(df["agent_id"] == AGENT) & (df["timestamp"].dt.date.astype(str) == DAY)]
day = day.sort_values("timestamp").reset_index(drop=True)


def run_day(day, opening_cash, use_rules, verbose=False):
    cash = opening_cash
    stockout_hours = 0
    unmet_total = 0

    for i, row in day.iterrows():
        # forecast for this hour and the next ones
        forecast_next = day["model_pred"].iloc[i : i + LOOKAHEAD_HOURS].tolist()
        rec = recommend_rebalance(cash, forecast_next)
        cash_before = cash

        if use_rules and rec["amount"] > 0:
            cash += rec["amount"]  # simplification: arrives instantly (Step 6 adds a delay)

        demand = row["cashout_amount"]
        if demand > cash:
            stockout_hours += 1
            unmet_total += demand - cash
            cash = 0
        else:
            cash -= demand

        if verbose:
            print(f"{int(row['hour']):>2}:00  cash {cash_before:>9,.0f}  "
                  f"{rec['status']:<6}  add {rec['amount']:>8,.0f}  "
                  f"demand {demand:>8,.0f}  left {cash:>9,.0f}")

    return stockout_hours, unmet_total


print(f"Agent {AGENT} on {DAY}, opening cash {OPENING_CASH:,} BDT\n")

print("--- WITH rules (agent approves every recommendation) ---")
s_with, u_with = run_day(day, OPENING_CASH, use_rules=True, verbose=True)

s_without, u_without = run_day(day, OPENING_CASH, use_rules=False)

print("\n--- Summary ---")
print(f"Without rules: {s_without} stockout hours, {u_without:,.0f} BDT of cash-outs missed")
print(f"With rules   : {s_with} stockout hours, {u_with:,.0f} BDT of cash-outs missed")

print("\nExample explanation:")
print(recommend_rebalance(40_000, [30_000, 35_000, 28_000])["reason"])