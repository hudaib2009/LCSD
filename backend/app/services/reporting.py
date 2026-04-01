from __future__ import annotations

from typing import Any

from backend.app.services.risk import combine_risk, compute_clinical_score, risk_tier, to_float

DISCLAIMER = "This is an educational decision-support output and not a medical diagnosis."
NOT_APPROVED = "This system is not clinically validated or approved for medical use."

PATHOLOGY_MODIFIER = {
    "ACA": 0.9,
    "SCC": 0.8,
    "NORMAL": 0.1,
}


def compute_imaging_risk(imaging_json: dict[str, Any]) -> float:
    terms: list[tuple[float, float]] = []

    ct_prob = to_float(imaging_json.get("ct_prob_mean"))
    if ct_prob is not None:
        terms.append((ct_prob, 0.6))

    ratio_positive = to_float(imaging_json.get("ratio_positive"))
    if ratio_positive is not None:
        terms.append((ratio_positive, 0.2))

    subtype = str(imaging_json.get("path_subtype", "")).upper()
    if subtype in PATHOLOGY_MODIFIER:
        terms.append((PATHOLOGY_MODIFIER[subtype], 0.2))

    if not terms:
        return 0.0

    total_weight = sum(weight for _, weight in terms)
    weighted_sum = sum(value * weight for value, weight in terms)
    return weighted_sum / total_weight


def generate_plan(stage: str, subtype: str, risk: str) -> dict[str, Any]:
    stage_value = str(stage or "").upper()
    subtype_value = str(subtype or "").upper()
    risk_value = str(risk or "").title()

    next_steps: list[str] = []
    warnings: list[str] = [DISCLAIMER, NOT_APPROVED]

    if stage_value in {"I", "II"}:
        next_steps.extend(
            [
                "Recommend surgical evaluation.",
                "Histopathological confirmation.",
                "Consider adjuvant therapy discussion.",
            ]
        )
    elif stage_value == "III":
        next_steps.extend(
            [
                "Multidisciplinary tumor board review.",
                "Consider combined chemoradiotherapy.",
            ]
        )
    elif stage_value == "IV":
        next_steps.extend(
            [
                "Consider systemic therapy.",
                "Discuss palliative/supportive care.",
                "Biomarker testing recommended.",
            ]
        )
    else:
        warnings.append("Stage proxy missing or invalid; using general guidance.")
        next_steps.append("Confirm staging with standard clinical workflows.")

    if subtype_value == "ACA":
        next_steps.append("Biomarker testing (EGFR, ALK, PD-L1) may be relevant.")
    elif subtype_value == "SCC":
        next_steps.append("Assess comorbidities and smoking-related risk factors.")
    elif subtype_value == "NORMAL":
        next_steps.append("Imaging follow-up may be appropriate if symptoms persist.")

    return {
        "summary": (
            f"Stage proxy {stage_value or 'Unknown'} with subtype {subtype_value or 'Unknown'}: "
            f"{risk_value} risk tier based on imaging + clinical inputs."
        ),
        "next_steps": next_steps,
        "warnings": warnings,
    }


def build_final_report(payload: dict[str, Any]) -> dict[str, Any]:
    clinical_json = payload.get("clinical") or {}
    imaging_json = payload.get("imaging") or {}

    clinical_score = compute_clinical_score(clinical_json)
    imaging_risk = compute_imaging_risk(imaging_json)
    final_risk_score = combine_risk(imaging_risk, clinical_score)
    tier = risk_tier(final_risk_score)

    stage_proxy = str(imaging_json.get("stage_proxy", "")).upper() or "Unknown"
    path_subtype = str(imaging_json.get("path_subtype", "")).upper() or "Unknown"
    plan = generate_plan(stage_proxy, path_subtype, tier)

    return {
        "clinical_score": clinical_score,
        "imaging_risk": imaging_risk,
        "final_risk_score": final_risk_score,
        "risk_tier": tier,
        "stage_proxy": stage_proxy,
        "path_subtype": path_subtype,
        "summary": plan["summary"],
        "next_steps": plan["next_steps"],
        "warnings": plan["warnings"],
    }
