import pytest

from backend.app.services.risk import combine_risk, compute_ct_stage_risk


def test_compute_ct_stage_risk_prefers_stage_three_weighting() -> None:
    imaging_risk, stage = compute_ct_stage_risk([0.1, 0.3, 0.6])
    assert stage == "III"
    assert imaging_risk == 0.75


def test_combine_risk_blends_imaging_and_clinical_scores() -> None:
    assert combine_risk(0.8, 0.2) == pytest.approx(0.62)
