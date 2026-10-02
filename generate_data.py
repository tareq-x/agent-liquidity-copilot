import os
import numpy as np
import pandas as pd

# ---------- Settings (you can change these later) ----------
SEED = 42
NUM_AGENTS = 20
NUM_DAYS = 90
START_DATE = "2026-07-01"
OPEN_HOURS = range(8, 21)  # 8 AM to 8 PM

# How busy each hour is (1.0 = average). Evening is the peak.
HOUR_WEIGHT = {
    8: 0.4, 9: 0.6, 10: 0.9, 11: 1.1, 12: 1.0, 13: 0.8, 14: 0.8,
    15: 1.0, 16: 1.3, 17: 1.6, 18: 1.4, 19: 0.9, 20: 0.5,
}

# Made-up festival days (synthetic)
FESTIVAL_DAYS = pd.to_datetime(["2026-07-24", "2026-07-25", "2026-08-20"])

# ---------- Create agents ----------
rng = np.random.default_rng(SEED)

agents = pd.DataFrame({
    "agent_id": [f"AG{i:03d}" for i in range(1, NUM_AGENTS + 1)],
    "base_demand": rng.uniform(8000, 25000, NUM_AGENTS).round(0),  # BDT per hour
})

dates = pd.date_range(START_DATE, periods=NUM_DAYS, freq="D")

# ---------- Generate one row per agent per hour ----------
rows = []
for _, agent in agents.iterrows():
    for date in dates:
        is_friday = date.dayofweek == 4
        is_salary_period = 5 <= date.day <= 10
        is_festival = date in FESTIVAL_DAYS

        day_factor = 1.0
        if is_friday:
            day_factor *= 0.8
        if is_salary_period:
            day_factor *= 1.4
        if is_festival:
            day_factor *= 2.0

        for hour in OPEN_HOURS:
            noise = rng.lognormal(mean=0, sigma=0.25)
            amount = agent["base_demand"] * HOUR_WEIGHT[hour] * day_factor * noise
            rows.append({
                "agent_id": agent["agent_id"],
                "timestamp": date + pd.Timedelta(hours=hour),
                "hour": hour,
                "day_of_week": date.dayofweek,
                "day_of_month": date.day,
                "is_friday": int(is_friday),
                "is_salary_period": int(is_salary_period),
                "is_festival": int(is_festival),
                "cashout_amount": round(amount),
            })

df = pd.DataFrame(rows)

# ---------- Save ----------
os.makedirs("data", exist_ok=True)
df.to_csv("data/synthetic_cashout.csv", index=False)

# ---------- Quick check ----------
print("Rows and columns:", df.shape)
print(df.head())
print()
print("Average cash-out by hour:")
print(df.groupby("hour")["cashout_amount"].mean().round(0))