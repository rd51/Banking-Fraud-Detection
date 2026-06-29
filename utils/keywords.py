"""Entity / keyword extraction from a flagged transaction (agentic NLP layer).

Pure regex + rule-based — no external NLP dependency. Extracts amounts,
counterparty codes, country-risk tier, time deltas and balance-error signals,
then returns both a structured entity table and a flat tag list for the UI.
"""

from __future__ import annotations

import re
from typing import Dict, List

# High-risk / monitored jurisdictions (illustrative FATF grey/black-list style).
COUNTRY_RISK = {
    "IR": ("Iran", "BLACKLIST"),
    "KP": ("North Korea", "BLACKLIST"),
    "MM": ("Myanmar", "BLACKLIST"),
    "SY": ("Syria", "HIGH"),
    "AF": ("Afghanistan", "HIGH"),
    "YE": ("Yemen", "HIGH"),
    "PA": ("Panama", "ELEVATED"),
    "KY": ("Cayman Islands", "ELEVATED"),
    "AE": ("UAE", "ELEVATED"),
    "MT": ("Malta", "ELEVATED"),
    "US": ("United States", "STANDARD"),
    "GB": ("United Kingdom", "STANDARD"),
}

STRUCTURING_THRESHOLD = 10000.0  # classic CTR reporting threshold


def _amount_band(amount: float) -> str:
    if amount >= 1_000_000:
        return "VERY_HIGH"
    if amount >= 100_000:
        return "HIGH"
    if amount >= 10_000:
        return "MEDIUM"
    return "LOW"


def extract_entities(txn: Dict) -> Dict[str, object]:
    """Extract a structured entity dictionary from a transaction record.

    ``txn`` is expected to contain raw PaySim fields plus the engineered
    error-balance features. Missing fields degrade gracefully.
    """
    amount = float(txn.get("amount", 0) or 0)
    err_orig = float(txn.get("errorBalanceOrig", 0) or 0)
    err_dest = float(txn.get("errorBalanceDest", 0) or 0)
    old_orig = float(txn.get("oldbalanceOrg", 0) or 0)
    new_orig = float(txn.get("newbalanceOrig", 0) or 0)
    ttype = str(txn.get("type", "TRANSFER"))
    name_dest = str(txn.get("nameDest", ""))
    name_orig = str(txn.get("nameOrig", ""))
    step = int(txn.get("step", 0) or 0)

    # Counterparty class — PaySim encodes merchants as 'M...' and accounts 'C...'
    dest_class = "MERCHANT" if name_dest.startswith("M") else "ACCOUNT"

    # Country tier — derive a pseudo country from a 2-letter code if present,
    # else infer 'US' standard. (Demo: regex any embedded ISO code.)
    code_match = re.search(r"\b([A-Z]{2})\b", name_dest)
    iso = code_match.group(1) if code_match and code_match.group(1) in COUNTRY_RISK else "US"
    country_name, country_tier = COUNTRY_RISK.get(iso, ("United States", "STANDARD"))

    # Structuring signal: amount just under the reporting threshold.
    near_threshold = 0.85 * STRUCTURING_THRESHOLD <= amount < STRUCTURING_THRESHOLD

    # Account drain: origin emptied to (near) zero.
    drained = old_orig > 0 and new_orig <= 0.01 * old_orig

    entities = {
        "transaction_type": ttype,
        "amount": amount,
        "amount_band": _amount_band(amount),
        "origin_account": name_orig,
        "dest_account": name_dest,
        "dest_class": dest_class,
        "country_code": iso,
        "country_name": country_name,
        "country_risk_tier": country_tier,
        "time_step_hours": step,
        "time_delta_label": f"hour {step % 24} of day {step // 24}",
        "balance_error_orig": round(err_orig, 2),
        "balance_error_dest": round(err_dest, 2),
        "structuring_signal": bool(near_threshold),
        "account_drained": bool(drained),
    }
    return entities


def entity_tags(entities: Dict) -> List[str]:
    """Flatten the most salient entities into UI 'pill' tags."""
    tags: List[str] = []
    tags.append(f"{entities['transaction_type']}")
    tags.append(f"${entities['amount']:,.0f}")
    tags.append(f"{entities['amount_band']} value")
    tags.append(f"{entities['dest_class'].title()}")
    tags.append(f"{entities['country_name']} · {entities['country_risk_tier']}")
    if entities.get("structuring_signal"):
        tags.append("⚠ STRUCTURING")
    if entities.get("account_drained"):
        tags.append("⚠ ACCOUNT DRAINED")
    if abs(entities.get("balance_error_orig", 0)) > 1:
        tags.append("BALANCE ERROR (orig)")
    if abs(entities.get("balance_error_dest", 0)) > 1:
        tags.append("BALANCE ERROR (dest)")
    return tags


def extractor_logic_source() -> str:
    """Readable summary of the rule logic, shown in the UI."""
    return (
        "Rule / regex extractor logic\n"
        "----------------------------\n"
        "1. dest_class      = 'M.*'  -> MERCHANT  else ACCOUNT\n"
        "2. country_code    = regex \\b[A-Z]{2}\\b -> COUNTRY_RISK lookup\n"
        "3. structuring     = 0.85*10,000 <= amount < 10,000\n"
        "4. account_drained = newbalanceOrig <= 1% * oldbalanceOrg\n"
        "5. amount_band     = LOW/MEDIUM/HIGH/VERY_HIGH thresholds\n"
        "6. balance errors  = |newbal + amount - oldbal| > 1"
    )
