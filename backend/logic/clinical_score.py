from __future__ import annotations

from typing import Any

from backend.app.services.risk import compute_clinical_score as _compute
from backend.app.services.risk import risk_tier


def compute_clinical_score(clinical_json: dict[str, Any]) -> dict[str, Any]:
    score = _compute(clinical_json)
    return {
        "clinical_score": score,
        "clinical_risk_tier": risk_tier(score),
    }
