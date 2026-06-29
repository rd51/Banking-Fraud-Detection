"""Suspicious Activity Report (SAR) narrative generator.

Fills a regulator-style narrative template with extracted entities and the
model's risk decision. Produces deterministic text suitable for analyst review.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from .keywords import extract_entities


DECISION_LANGUAGE = {
    "FREEZE": "high probability of money laundering / fraud",
    "REVIEW": "elevated indicators warranting enhanced due diligence",
    "CLEAR": "low residual risk after automated screening",
}


def generate_sar(txn: Dict, probability: float, decision: str,
                 entities: Dict | None = None) -> str:
    """Return a formatted SAR narrative paragraph."""
    e = entities or extract_entities(txn)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    structuring = (
        " Transaction amount falls immediately below the USD 10,000 reporting "
        "threshold, a pattern consistent with structuring." if e.get("structuring_signal") else ""
    )
    drained = (
        " The originating account was substantially depleted in a single "
        "movement, consistent with bust-out / mule behaviour." if e.get("account_drained") else ""
    )
    balerr = ""
    if abs(e.get("balance_error_orig", 0)) > 1 or abs(e.get("balance_error_dest", 0)) > 1:
        balerr = (
            " Ledger reconciliation detected balance inconsistencies "
            f"(orig error {e.get('balance_error_orig')}, dest error "
            f"{e.get('balance_error_dest')}), indicating possible obfuscation."
        )

    narrative = (
        f"SUSPICIOUS ACTIVITY REPORT — DRAFT (auto-generated {now})\n"
        f"{'='*64}\n\n"
        f"Filing institution model decision: {decision} "
        f"(model risk score {probability:.1%}).\n\n"
        f"On {e.get('time_delta_label', 'the reported period')}, account "
        f"{e.get('origin_account', 'N/A')} executed a "
        f"{e.get('transaction_type')} of USD {e.get('amount', 0):,.2f} "
        f"({e.get('amount_band','').replace('_',' ').lower()} value) to "
        f"{e.get('dest_class','account').lower()} {e.get('dest_account','N/A')} "
        f"associated with {e.get('country_name','United States')} "
        f"(risk tier: {e.get('country_risk_tier','STANDARD')}).\n\n"
        f"The automated sequence-modelling system (Conv1D + LSTM + Attention + "
        f"XGBoost ensemble) assessed the surrounding 90-day behavioural pattern "
        f"and identified {DECISION_LANGUAGE.get(decision, 'risk indicators')}."
        f"{structuring}{drained}{balerr}\n\n"
        f"Recommended action: "
        f"{'File SAR and freeze funds pending investigation.' if decision == 'FREEZE' else ''}"
        f"{'Escalate to L2 analyst for enhanced due diligence.' if decision == 'REVIEW' else ''}"
        f"{'No filing required; retain for audit trail.' if decision == 'CLEAR' else ''}\n\n"
        f"This draft was machine-generated and requires human analyst review "
        f"and sign-off before regulatory submission."
    )
    return narrative
