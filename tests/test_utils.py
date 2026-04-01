from backend.app.services.risk import compute_clinical_score


def test_compute_clinical_score_handles_frontend_shape() -> None:
    score = compute_clinical_score(
        {
            "age": 65,
            "isSmoker": True,
            "packYears": 30,
            "ecog": 2,
            "histology": "Adenocarcinoma",
        }
    )
    assert score == 1.0


def test_compute_clinical_score_handles_missing_values() -> None:
    score = compute_clinical_score({})
    assert score == 0.0
