-- 1. 일주일 전 날짜 이전 데이터 삭제
DELETE FROM mlops.credit_score_features_target
WHERE base_dt <= DATE_FORMAT(
        DATE_ADD(
            STR_TO_DATE(
                '{{ data_interval_start.in_timezone("Asia/Seoul").to_date_string() }}',
                '%Y-%m-%d'
            ),
            INTERVAL -7 DAY
        ),
        '%Y-%m-%d'
    )
    OR base_dt = STR_TO_DATE(
        '{{ data_interval_start.in_timezone("Asia/Seoul").to_date_string() }}',
        '%Y-%m-%d'
    );
-- 2. 새로운 데이터 삽입
INSERT INTO mlops.credit_score_features_target (
        base_dt,
        id,
        customer_id,
        date,
        age,
        occupation,
        annual_income,
        monthly_inhand_salary,
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        num_of_loan,
        type_of_loan,
        delay_from_due_date,
        num_of_delayed_payment,
        changed_credit_limit,
        num_credit_inquiries,
        credit_mix,
        outstanding_debt,
        credit_utilization_ratio,
        credit_history_age,
        payment_of_min_amount,
        total_emi_per_month,
        amount_invested_monthly,
        payment_behaviour,
        monthly_balance,
        credit_score
    )
SELECT STR_TO_DATE(
        '{{ data_interval_start.in_timezone("Asia/Seoul").to_date_string() }}',
        '%Y-%m-%d'
    ) AS base_dt,
    b.id,
    b.customer_id,
    b.date,
    a.age,
    a.occupation,
    a.annual_income,
    a.monthly_inhand_salary,
    a.num_bank_accounts,
    a.num_credit_card,
    a.interest_rate,
    a.num_of_loan,
    a.type_of_loan,
    a.delay_from_due_date,
    a.num_of_delayed_payment,
    a.changed_credit_limit,
    a.num_credit_inquiries,
    a.credit_mix,
    a.outstanding_debt,
    a.credit_utilization_ratio,
    a.credit_history_age,
    a.payment_of_min_amount,
    a.total_emi_per_month,
    a.amount_invested_monthly,
    a.payment_behaviour,
    a.monthly_balance,
    b.credit_score
FROM mlops.credit_score_features a
    INNER JOIN (
        SELECT *
        FROM mlops.credit_score
        WHERE date BETWEEN DATE_ADD(
                STR_TO_DATE(
                    '{{ data_interval_start.in_timezone("Asia/Seoul").to_date_string() }}',
                    '%Y-%m-%d'
                ),
                INTERVAL -1 MONTH
            )
            AND STR_TO_DATE(
                '{{ data_interval_start.in_timezone("Asia/Seoul").to_date_string() }}',
                '%Y-%m-%d'
            )
    ) b ON a.id = b.id
    AND a.customer_id = b.customer_id;