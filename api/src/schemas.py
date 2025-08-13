from datetime import datetime
from typing import Any, Dict

from bentoml import IODescriptor
from pydantic import BaseModel, field_validator


class Features(BaseModel):
    customer_id: int
    age: int
    occupation: str
    annual_income: float
    monthly_inhand_salary: float
    num_bank_accounts: float
    num_credit_card: float
    interest_rate: float
    num_of_loan: int
    type_of_loan: str
    delay_from_due_date: float
    num_of_delayed_payment: float
    changed_credit_limit: float
    num_credit_inquiries: float
    credit_mix: str
    outstanding_debt: float
    credit_utilization_ratio: float
    credit_history_age: float
    payment_of_min_amount: str
    total_emi_per_month: float
    amount_invested_monthly: float
    payment_behaviour: str
    monthly_balance: float

    @field_validator("age", "credit_history_age")
    @classmethod
    def validate_age(cls, value):
        if value > 0:
            return value
        raise ValueError("0보다 커야합니다.")

    @field_validator("credit_mix")
    @classmethod
    def validate_credit_mix(cls, value):
        if value in ["Good", "Bad", "Standard"]:
            return value
        raise ValueError(
            "'Good', 'Bad', 'Standard' 중 하나의 값을 가져야 합니다."
        )

    @field_validator("payment_of_min_amount")
    @classmethod
    def validate_payment_of_min_amount(cls, value):
        if value in ["NM", "Yes", "No"]:
            return value
        raise ValueError("'NM', 'Yes', 'No' 중 하나의 값을 가져야 합니다.")


class Response(BaseModel):
    customer_id: int
    predict: str
    confidence: float


class MetadataResponse(IODescriptor):
    model_name: str
    model_version: str
    params: Dict[str, Any]
    creation_time: datetime
