from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class ClinicalInput(BaseModel):
    age: int
    sex: Literal["M", "F"]
    smoker: bool
    pack_years: float
    ecog: int
    weight_loss: bool

    @field_validator("age")
    def _age_range(cls, value: int) -> int:
        if value < 18 or value > 100:
            raise ValueError("age must be between 18 and 100")
        return value

    @field_validator("pack_years")
    def _pack_years_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("pack_years must be >= 0")
        return value

    @field_validator("ecog")
    def _ecog_range(cls, value: int) -> int:
        if value < 0 or value > 4:
            raise ValueError("ecog must be between 0 and 4")
        return value

    @model_validator(mode="after")
    def _apply_smoker_rule(self) -> "ClinicalInput":
        if self.smoker is False:
            self.pack_years = 0.0
        return self
