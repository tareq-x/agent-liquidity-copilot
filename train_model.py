import os
import pandas as pd
import lightgbm as lgb

# ---------- 1. Load and prepare ----------
df = pd.read_csv("data/synthetic_cashout.csv", parse_dates=["timestamp"])
df = df.sort_values(["agent_id", "timestamp"]).reset_index(drop=True)

HOURS_PER_DAY = 13  # 8 AM to 8 PM
by_agent = df.groupby("agent_id")["cashout_amount"]
df["lag_week"] = by_agent.shift(HOURS_PER_DAY * 7)  # same hour, last week
df["lag_day"] = by_agent.shift(HOURS_PER_DAY)       # same hour, yesterday
df["baseline_pred"] = df["lag_week"]                # the Step 3 baseline

df["agent_id"] = df["agent_id"].astype("category")  # lets LightGBM treat it as a label

FEATURES = [
    "agent_id", "hour", "day_of_week", "day_of_month",
    "is_salary_period", "is_festival", "lag_week", "lag_day",
]

# ---------- 2. Split: train before Sept 1, test from Sept 1 ----------
cutoff = df["timestamp"].max().normalize() - pd.Timedelta(days=27)
train = df[df["timestamp"] < cutoff].dropna(subset=["lag_week", "lag_day"])
test = df[df["timestamp"] >= cutoff].copy()
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

# ---------- 3. Train ----------
model = lgb.LGBMRegressor(
    n_estimators=300, learning_rate=0.05, num_leaves=31,
    random_state=42, verbose=-1,
)
model.fit(train[FEATURES], train["cashout_amount"])

# ---------- 4. Predict ----------
test["model_pred"] = model.predict(test[FEATURES]).round()

# ---------- 5. Compare ----------
def score(actual, pred):
    err = (actual - pred).abs()
    return err.mean(), (err / actual).mean() * 100

base_mae, base_mape = score(test["cashout_amount"], test["baseline_pred"])
model_mae, model_mape = score(test["cashout_amount"], test["model_pred"])

print()
print(f"{'':10}{'MAE (BDT)':>12}{'MAPE (%)':>12}")
print(f"{'Baseline':10}{base_mae:>12,.0f}{base_mape:>12.1f}")
print(f"{'LightGBM':10}{model_mae:>12,.0f}{model_mape:>12.1f}")
print(f"\nLightGBM reduces MAE by {(1 - model_mae / base_mae) * 100:.1f}%")

test["base_err"] = (test["cashout_amount"] - test["baseline_pred"]).abs()
test["model_err"] = (test["cashout_amount"] - test["model_pred"]).abs()
print("\nMAE by salary period (1 = salary days):")
print(test.groupby("is_salary_period")[["base_err", "model_err"]].mean().round(0))

print("\nFeature importance:")
print(pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False))

# ---------- 6. Save for the next steps ----------
os.makedirs("models", exist_ok=True)
model.booster_.save_model("models/lgbm_model.txt")
test[["agent_id", "timestamp", "hour", "cashout_amount", "baseline_pred", "model_pred"]].to_csv(
    "data/test_predictions.csv", index=False
)
print("\nSaved models/lgbm_model.txt and data/test_predictions.csv")