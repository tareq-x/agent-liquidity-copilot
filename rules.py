import math

# ---------- Business rules (easy to tune) ----------
LOOKAHEAD_HOURS = 3      # how far ahead we plan
SAFETY_MARGIN = 0.20     # keep 20% extra on top of forecast demand
MIN_REBALANCE = 10_000   # ignore tiny top-ups
MAX_REBALANCE = 200_000  # an agent can't move unlimited cash at once
ROUND_TO = 5_000         # round amounts up to a practical number


def recommend_rebalance(current_cash, forecast_next):
    """Turn a cash balance and forecast demand into a recommendation."""
    expected_need = float(sum(forecast_next))
    required = expected_need * (1 + SAFETY_MARGIN)

    if current_cash >= required:
        status = "OK"
        amount = 0
        reason = (
            f"Cash {current_cash:,.0f} BDT covers the forecast demand of "
            f"{expected_need:,.0f} BDT for the next {len(forecast_next)} hour(s)."
        )
    else:
        shortfall = required - current_cash
        amount = math.ceil(shortfall / ROUND_TO) * ROUND_TO
        amount = min(max(amount, MIN_REBALANCE), MAX_REBALANCE)
        status = "URGENT" if current_cash < expected_need else "WARN"
        reason = (
            f"Forecast demand for the next {len(forecast_next)} hour(s) is "
            f"{expected_need:,.0f} BDT, but cash is only {current_cash:,.0f} BDT. "
            f"Recommend adding {amount:,.0f} BDT to keep a {SAFETY_MARGIN:.0%} safety margin."
        )

    return {
        "status": status,
        "amount": amount,
        "reason": reason,
        "requires_approval": True,  # human oversight: the agent decides
    }