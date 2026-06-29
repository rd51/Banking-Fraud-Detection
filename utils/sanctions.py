"""Mock OFAC / sanctions + PEP screening.

Entirely offline — a hardcoded list of 20 names stands in for the real OFAC
SDN list. Performs case-insensitive fuzzy matching and returns a risk tier.
"""

from __future__ import annotations

import difflib
from typing import Dict, List

# 20-name mock OFAC SDN list (fictional / illustrative).
MOCK_OFAC: List[Dict[str, str]] = [
    {"name": "Viktor Petrov", "program": "RUSSIA-EO14024", "tier": "CRITICAL"},
    {"name": "Aleksandr Volkov", "program": "RUSSIA-EO14024", "tier": "CRITICAL"},
    {"name": "Global Maritime Holdings", "program": "IRAN-HR", "tier": "CRITICAL"},
    {"name": "Crescent Trading FZE", "program": "IRAN-HR", "tier": "CRITICAL"},
    {"name": "Kim Jong-Ho", "program": "DPRK", "tier": "CRITICAL"},
    {"name": "Pyongyang Star Bank", "program": "DPRK", "tier": "CRITICAL"},
    {"name": "Carlos Mendez", "program": "SDNTK-NARCO", "tier": "HIGH"},
    {"name": "Sinaloa Logistics SA", "program": "SDNTK-NARCO", "tier": "HIGH"},
    {"name": "Dmitri Sokolov", "program": "CYBER-EO13694", "tier": "HIGH"},
    {"name": "Lazarus Digital LLC", "program": "CYBER-EO13694", "tier": "CRITICAL"},
    {"name": "Hassan Al-Rashid", "program": "SDGT", "tier": "CRITICAL"},
    {"name": "Damascus Finance Group", "program": "SYRIA", "tier": "HIGH"},
    {"name": "Wagner Resource Corp", "program": "RUSSIA-EO14024", "tier": "HIGH"},
    {"name": "Maria Gonzalez", "program": "VENEZUELA-EO13850", "tier": "MEDIUM"},
    {"name": "Caracas Oil Brokers", "program": "VENEZUELA-EO13850", "tier": "MEDIUM"},
    {"name": "Yusuf Ibrahim", "program": "SDGT", "tier": "HIGH"},
    {"name": "Eastern Crypto Exchange", "program": "CYBER-EO13694", "tier": "HIGH"},
    {"name": "Ahmed Nasser", "program": "YEMEN", "tier": "MEDIUM"},
    {"name": "Tehran Shipping Lines", "program": "IRAN-HR", "tier": "CRITICAL"},
    {"name": "Boris Kuznetsov", "program": "RUSSIA-EO14024", "tier": "HIGH"},
]

# Mock PEP (Politically Exposed Person) watchlist.
MOCK_PEP = {
    "carlos mendez", "maria gonzalez", "viktor petrov", "ahmed nasser",
}


def screen(name: str, threshold: float = 0.82) -> Dict[str, object]:
    """Screen a name against the mock OFAC list + PEP watchlist.

    Returns a result dict with match status, best candidate, similarity score,
    sanctions program, risk tier and a PEP flag.
    """
    query = (name or "").strip().lower()
    if not query:
        return {
            "status": "NO MATCH", "query": name, "best_match": None,
            "score": 0.0, "program": None, "tier": "NONE", "pep": False,
        }

    best = None
    best_score = 0.0
    for entry in MOCK_OFAC:
        score = difflib.SequenceMatcher(None, query, entry["name"].lower()).ratio()
        # token-level boost: any shared token raises confidence
        q_tokens = set(query.split())
        e_tokens = set(entry["name"].lower().split())
        if q_tokens & e_tokens:
            score = max(score, 0.6 + 0.1 * len(q_tokens & e_tokens))
        if score > best_score:
            best_score, best = score, entry

    is_match = best_score >= threshold
    pep = query in MOCK_PEP or any(tok in MOCK_PEP for tok in [query])

    return {
        "status": "MATCH" if is_match else "NO MATCH",
        "query": name,
        "best_match": best["name"] if best else None,
        "score": round(float(best_score), 3),
        "program": best["program"] if (best and is_match) else None,
        "tier": best["tier"] if (best and is_match) else ("REVIEW" if best_score >= 0.6 else "NONE"),
        "pep": bool(pep),
    }
