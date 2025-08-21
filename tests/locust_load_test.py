# How to run this script:
# 1. Install locust: pip install locust
# 2. Run the BentoML server in a separate terminal: bentoml serve api/services.py:CreditScoreClassifier --reload
# 3. Run locust: locust -f locust_load_test.py --host http://localhost:3000

import random

from locust import HttpUser, between, task


def get_random_features():
    return {
        "customer_id": random.randint(1, 100000),
        "age": random.randint(18, 70),
        "occupation": random.choice(
            [
                "Developer",
                "Journalist",
                "Scientist",
                "Engineer",
                "Architect",
                "Lawyer",
                "Doctor",
                "Manager",
                "Musician",
                "Teacher",
                "Entrepreneur",
                "Mechanic",
                "Writer",
                "Media_Manager",
                "Accountant",
            ]
        ),
        "annual_income": random.uniform(5000, 200000),
        "monthly_inhand_salary": random.uniform(400, 15000),
        "num_bank_accounts": float(random.randint(0, 10)),
        "num_credit_card": float(random.randint(0, 10)),
        "interest_rate": float(random.randint(1, 35)),
        "num_of_loan": random.randint(0, 10),
        "type_of_loan": "Personal Loan",
        "delay_from_due_date": float(random.randint(0, 60)),
        "num_of_delayed_payment": float(random.randint(0, 25)),
        "changed_credit_limit": float(random.uniform(0, 30)),
        "num_credit_inquiries": float(random.randint(0, 15)),
        "credit_mix": random.choice(["Good", "Standard", "Bad"]),
        "outstanding_debt": float(random.uniform(0, 5000)),
        "credit_utilization_ratio": random.uniform(20, 50),
        "credit_history_age": float(random.randint(10, 400)),
        "payment_of_min_amount": random.choice(["Yes", "No", "NM"]),
        "total_emi_per_month": float(random.uniform(0, 1000)),
        "amount_invested_monthly": float(random.uniform(0, 1000)),
        "payment_behaviour": "Low_spent_Small_value_payments",
        "monthly_balance": float(random.uniform(0, 1200)),
    }


class CreditScoreUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict(self):
        headers = {"Content-Type": "application/json"}
        features = get_random_features()
        payload = {"data": features}
        self.client.post("/predict", json=payload, headers=headers)
