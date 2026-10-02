import os

import pandas as pd
import streamlit as st
from explain import explain
from rules import recommend_rebalance, LOOKAHEAD_HOURS, SAFETY_MARGIN

st.set_page_config(page_title="Agent Liquidity Copilot", page_icon="💸", layout="wide")


# ---------- Load data (cached so it only loads once) ----------
@st.cache_data
def load_data():
    preds = pd.read_csv("data/test_predictions.csv", parse_dates=["timestamp"])
    full = pd.read_csv("data/synthetic_cashout.csv", parse_dates=["timestamp"])
    sim = pd.read_csv("data/simulation_results.csv")
    cutoff = preds["timestamp"].min().normalize()
    avg_hourly = full[full["timestamp"] < cutoff].groupby("agent_id")["cashout_amount"].mean()
    return preds, sim, avg_hourly


preds, sim, avg_hourly = load_data()

if "log" not in st.session_state:
    st.session_state.log = []

st.title("💸 Agent Liquidity Copilot")
st.caption("Forecast cash-out demand, avoid stockouts. Synthetic data only. The agent approves every action.")

tab1, tab2 = st.tabs(["Agent Copilot", "Results"])

# =====================================================
# TAB 1: the copilot
# =====================================================
with tab1:
    c1, c2, c3 = st.columns(3)
    agent = c1.selectbox("Agent", sorted(preds["agent_id"].unique()))

    agent_rows = preds[preds["agent_id"] == agent].copy()
    agent_rows["date"] = agent_rows["timestamp"].dt.date
    date = c2.selectbox("Day", sorted(agent_rows["date"].unique()))

    day = agent_rows[agent_rows["date"] == date].sort_values("timestamp").reset_index(drop=True)
    hour = c3.selectbox("Current hour", day["hour"].tolist(), index=2, format_func=lambda h: f"{h}:00")

    default_cash = int(round(6 * avg_hourly[agent], -3))
    cash = st.number_input(
        "Agent's current cash (BDT)", min_value=0, value=default_cash, step=5000,
        help="Defaults to 6x this agent's average hourly demand.",
    )
    lang = "bn" if st.radio("Language", ["English", "বাংলা"], horizontal=True) == "বাংলা" else "en"

    # Chart: actual vs forecasts for the day
    st.subheader("Hourly cash-out demand")
    chart = day.set_index("hour")[["cashout_amount", "baseline_pred", "model_pred"]]
    chart.columns = ["Actual", "Baseline forecast", "LightGBM forecast"]
    st.line_chart(chart)

    # Recommendation for the selected hour
    i = day.index[day["hour"] == hour][0]
    forecast_next = day["model_pred"].iloc[i : i + LOOKAHEAD_HOURS].tolist()
    rec = recommend_rebalance(cash, forecast_next)
    text, source = explain(rec, cash, forecast_next, hour, lang, use_llm=bool(os.environ.get("LLM_API_KEY")))

    st.subheader(f"Recommendation at {hour}:00")
    m1, m2, m3 = st.columns(3)
    m1.metric("Current cash", f"{cash:,.0f} BDT")
    m2.metric(f"Forecast need (next {len(forecast_next)}h)", f"{sum(forecast_next):,.0f} BDT")
    m3.metric(f"Cash needed (+{SAFETY_MARGIN:.0%} margin)", f"{sum(forecast_next) * (1 + SAFETY_MARGIN):,.0f} BDT")

    if rec["status"] == "URGENT":
        st.error(f"🔴 URGENT: add {rec['amount']:,.0f} BDT now")
    elif rec["status"] == "WARN":
        st.warning(f"🟠 WARNING: add {rec['amount']:,.0f} BDT soon")
    else:
        st.success("🟢 OK: no action needed")
    st.write(text)
    if source == "llm":
        st.caption("Reworded by an LLM. Numbers checked against the forecast.")

    if rec["amount"] > 0:
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("✅ Approve"):
            st.session_state.log.append(
                {"agent": agent, "day": str(date), "hour": f"{hour}:00",
                 "status": rec["status"], "amount": rec["amount"], "decision": "Approved"})
        if b2.button("❌ Reject"):
            st.session_state.log.append(
                {"agent": agent, "day": str(date), "hour": f"{hour}:00",
                 "status": rec["status"], "amount": rec["amount"], "decision": "Rejected"})

    if st.session_state.log:
        st.subheader("Decision log")
        st.dataframe(pd.DataFrame(st.session_state.log), use_container_width=True)

# =====================================================
# TAB 2: simulation results
# =====================================================
with tab2:
    st.subheader("Stockout simulation: 20 agents x 28 days")
    base_row = sim[sim["strategy"] == "Rules + baseline forecast"].iloc[0]
    lgb_row = sim[sim["strategy"] == "Rules + LightGBM forecast"].iloc[0]
    reduction = (1 - lgb_row["stockout_hours"] / base_row["stockout_hours"]) * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Stockout hours cut (AI vs baseline)", f"{reduction:.0f}%")
    k2.metric("Service level (AI)", f"{lgb_row['service_level_pct']}%")
    k3.metric("Cash moved vs baseline", f"{(lgb_row['cash_moved_bdt'] / base_row['cash_moved_bdt'] - 1) * 100:.1f}%")

    st.dataframe(sim, use_container_width=True)
    st.bar_chart(sim.set_index("strategy")["stockout_hours"])

    with st.expander("Assumptions and responsible AI notes"):
        st.markdown(
            "- All data is **synthetic**. No real customer or agent data is used.\n"
            "- Opening cash = 6x average hourly demand; rebalances arrive 1 hour after ordering.\n"
            "- The simulation assumes the agent approves every recommendation. Real approval rates will differ.\n"
            "- The ML model only forecasts demand. **Transparent rules** make the recommendation, and the **human agent decides**."
        )