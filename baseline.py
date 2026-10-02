import pandas as pd

# ---------- 1. Load the data ----------
df = pd.read_csv("data/synthetic_cashout.csv", parse_dates=["timestamp"])
df = df.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)

print("Shape:", df.shape)
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())
print()

# ---------- 2. Explore: do our patterns show up? ----------
print("Average cash-out: salary period (1) vs normal (0)")
print(df.groupby("is_salary_period")["cashout_amount"].mean().round(0))
print()
print("Average cash-out: Friday (1) vs other days (0)")
print(df.groupby("is_friday")["cashout_amount"].mean().round(0))
print()
print("Average cash-out: festival (1) vs normal (0)")
print(df.groupby("is_festival")["cashout_amount"].mean().round(0))
print()

# ---------- 3. Baseline: same hour, same weekday, last week ----------
HOURS_PER_DAY = 13  # 8 AM to 8 PM
df["baseline_pred"] = df.groupby("agent_id")["cashout_amount"].shift(HOURS_PER_DAY * 7)

# ---------- 4. Test set: last 28 days ----------
cutoff = df["timestamp"].max().normalize() - pd.Timedelta(days=27)
test = df[df["timestamp"] >= cutoff].copy()
print("Test set starts:", cutoff.date(), "| rows:", len(test))

# ---------- 5. How wrong is the baseline? ----------
test["abs_error"] = (test["cashout_amount"] - test["baseline_pred"]).abs()
mae = test["abs_error"].mean()
mape = (test["abs_error"] / test["cashout_amount"]).mean() * 100

print()
print(f"Baseline MAE : {mae:,.0f} BDT per agent-hour")
print(f"Baseline MAPE: {mape:.1f}%")
print()
print("Baseline MAE: salary period (1) vs normal (0)")
print(test.groupby("is_salary_period")["abs_error"].mean().round(0))