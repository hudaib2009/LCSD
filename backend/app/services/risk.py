from __future__ import annotations

from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_clinical_score(clinical: dict[str, Any]) -> float:
    points = 0

    age = to_float(clinical.get("age"))
    if age is not None and age > 60:
        points += 1

    smoker = clinical.get("isSmoker", clinical.get("smoker"))
    if smoker is True or str(smoker).lower() == "true":
        points += 1

    pack_years = to_float(clinical.get("pack_years"))
    if pack_years is None:
        pack_years = to_float(clinical.get("packYears"))
    if pack_years is not None and pack_years >= 20:
        points += 1

    ecog = to_float(clinical.get("ecog"))
    if ecog is not None and ecog >= 2:
        points += 1

    histology = clinical.get("histology")
    if histology and str(histology).strip().lower() != "unknown":
        points += 1

    return clamp01(points / 5.0)


def compute_ct_stage_risk(probabilities: list[float] | tuple[float, ...]) -> tuple[float, str]:
    if not probabilities:
        return 0.0, "Unknown"

    if len(probabilities) >= 3:
        stage_labels = ["I", "II", "III"]
        stage_index = max(range(len(stage_labels)), key=lambda idx: probabilities[idx])
        imaging_risk = (probabilities[1] * 0.5) + (probabilities[2] * 1.0)
        return clamp01(imaging_risk), stage_labels[stage_index]

    return clamp01(probabilities[0]), "Unknown"


def combine_risk(imaging_risk: float, clinical_risk: float) -> float:
    return clamp01((0.7 * imaging_risk) + (0.3 * clinical_risk))


def risk_bucket(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def risk_tier(score: float) -> str:
    value = risk_bucket(score)
    return value.capitalize()
