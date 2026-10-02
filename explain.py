import os
import re
from rules import SAFETY_MARGIN

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
EN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

TEMPLATES = {
    ("en", "URGENT"): (
        "Cash may run out around {runout}. The model expects about {need} BDT of cash-outs in the "
        "next {n} hour(s), busiest at {peak} (about {peak_amt} BDT), but you have {cash} BDT. "
        "Top-ups take about an hour to arrive, so order {amount} BDT now. You decide: approve or decline."
    ),
    ("en", "WARN"): (
        "You have enough cash for the forecast demand, but not for the {margin} safety cushion. "
        "About {need} BDT is expected in the next {n} hour(s), busiest at {peak}, and you have {cash} BDT. "
        "Consider adding {amount} BDT within the hour. You decide: approve or decline."
    ),
    ("en", "OK"): (
        "No action needed. About {need} BDT is expected in the next {n} hour(s), "
        "and you have {cash} BDT."
    ),
    ("bn", "URGENT"): (
        "{runout} টার দিকে আপনার ক্যাশ শেষ হয়ে যেতে পারে। পরের {n} ঘণ্টায় আনুমানিক {need} টাকা ক্যাশ-আউট হবে, "
        "সবচেয়ে ব্যস্ত সময় {peak} (প্রায় {peak_amt} টাকা), কিন্তু আপনার কাছে আছে {cash} টাকা। "
        "টাকা আসতে প্রায় এক ঘণ্টা লাগে, তাই এখনই {amount} টাকা অর্ডার করুন। সিদ্ধান্ত আপনার: অনুমোদন করুন বা বাতিল করুন।"
    ),
    ("bn", "WARN"): (
        "আপনার ক্যাশ পূর্বাভাসের চাহিদা মেটাবে, কিন্তু {margin} বাড়তি নিরাপত্তা থাকবে না। "
        "পরের {n} ঘণ্টায় আনুমানিক {need} টাকা লাগবে, সবচেয়ে ব্যস্ত সময় {peak}, আর আপনার কাছে আছে {cash} টাকা। "
        "এক ঘণ্টার মধ্যে {amount} টাকা যোগ করার কথা ভাবুন। সিদ্ধান্ত আপনার।"
    ),
    ("bn", "OK"): (
        "এখন কিছু করার দরকার নেই। পরের {n} ঘণ্টায় আনুমানিক {need} টাকা লাগবে, "
        "আর আপনার কাছে আছে {cash} টাকা।"
    ),
}


def _facts(cash, forecast_next, hour, amount):
    peak_i = max(range(len(forecast_next)), key=lambda i: forecast_next[i])
    running, runout = 0, None
    for i, f in enumerate(forecast_next):
        running += f
        if running > cash:
            runout = f"{hour + i}:00"
            break
    return {
        "n": len(forecast_next),
        "need": f"{sum(forecast_next):,.0f}",
        "cash": f"{cash:,.0f}",
        "amount": f"{amount:,.0f}",
        "peak": f"{hour + peak_i}:00",
        "peak_amt": f"{forecast_next[peak_i]:,.0f}",
        "runout": runout,
        "margin": f"{SAFETY_MARGIN:.0%}",
    }


def _numbers(text):
    plain = text.translate(EN_DIGITS).replace(",", "")
    return sorted(re.findall(r"\d+", plain))


def _llm_reword(text, lang):
    """Optional. Returns reworded text, or None if unavailable or if any number changed."""
    key = os.environ.get("LLM_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        language = "Bangla" if lang == "bn" else "English"
        msg = client.messages.create(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"),
            max_tokens=400,
            system=(
                f"Rewrite the message in simple, friendly {language} for a small shop owner. "
                "Keep every number, time and the final decision exactly as given. "
                "Do not add advice, new facts or new numbers. Reply with the message only."
            ),
            messages=[{"role": "user", "content": text}],
        )
        out = msg.content[0].text.strip()
        return out if _numbers(out) == _numbers(text) else None
    except Exception:
        return None


def explain(rec, cash, forecast_next, hour, lang="en", use_llm=False):
    """Returns (text, source) where source is 'template' or 'llm'."""
    facts = _facts(cash, forecast_next, hour, rec["amount"])
    text = TEMPLATES[(lang, rec["status"])].format(**facts)
    if lang == "bn":
        text = text.translate(BN_DIGITS)
    if use_llm:
        reworded = _llm_reword(text, lang)
        if reworded:
            return reworded, "llm"
    return text, "template"


if __name__ == "__main__":
    from rules import recommend_rebalance

    cases = {"URGENT": 98_355, "WARN": 125_000, "OK": 200_000}
    forecast = [32_000, 41_000, 46_000]
    for name, cash in cases.items():
        rec = recommend_rebalance(cash, forecast)
        for lang in ("en", "bn"):
            print(f"[{name} / {lang}]")
            print(explain(rec, cash, forecast, 15, lang)[0], "\n")