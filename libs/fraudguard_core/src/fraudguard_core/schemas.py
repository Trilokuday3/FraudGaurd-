"""Pandera schemas for every entity in the data model.

Single source of truth for "what a clean row looks like" — used by the DQ
suite (`tests/dq`) against generator output, and importable by any downstream
job that wants to validate its own input.
"""

from pandera.pandas import Check, Column, DataFrameSchema

from fraudguard_core.enums import (
    COUNTRIES,
    CURRENCY,
    CUSTOMER_SEGMENT,
    DEVICE_TYPE,
    MERCHANT_CATEGORY,
    PAYMENT_METHOD,
)

customers_schema = DataFrameSchema(
    {
        "customer_id": Column(str, unique=True, checks=Check.str_startswith("CUST")),
        "account_created_at": Column("datetime64[ns]"),
        "customer_segment": Column(str, checks=Check.isin(CUSTOMER_SEGMENT)),
        "home_country": Column(str, checks=Check.isin(COUNTRIES)),
        "home_city": Column(str),
    },
    strict=True,
)

merchants_schema = DataFrameSchema(
    {
        "merchant_id": Column(str, unique=True, checks=Check.str_startswith("MERC")),
        "merchant_category": Column(str, checks=Check.isin(MERCHANT_CATEGORY)),
        "country": Column(str, checks=Check.isin(COUNTRIES)),
        "created_at": Column("datetime64[ns]"),
    },
    strict=True,  # base_fraud_rate must NEVER appear here — see leakage guards
)

devices_schema = DataFrameSchema(
    {
        "device_id": Column(str, unique=True, checks=Check.str_startswith("DEV")),
        "customer_id": Column(str),
        "device_type": Column(str, checks=Check.isin(DEVICE_TYPE)),
        "first_seen_at": Column("datetime64[ns]"),
    },
    strict=True,
)

transactions_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True, checks=Check.str_startswith("TXN")),
        "customer_id": Column(str),
        "merchant_id": Column(str),
        "device_id": Column(str),
        "timestamp": Column("datetime64[ns]"),
        "amount": Column(float, checks=Check.gt(0)),
        "currency": Column(str, checks=Check.isin(CURRENCY)),
        "payment_method": Column(str, checks=Check.isin(PAYMENT_METHOD)),
        "country": Column(str, checks=Check.isin(COUNTRIES)),
        "city": Column(str),
        "ip_country": Column(str, checks=Check.isin(COUNTRIES)),
    },
    strict=True,  # fraud_label / fraud_probability_true must NEVER appear here
)

ground_truth_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True),
        "fraud_probability_true": Column(float, checks=Check.in_range(0, 1)),
        "fraud_label": Column(int, checks=Check.isin({0, 1})),
        "confirmed_fraud_at": Column("datetime64[ns]", nullable=True),
    },
    strict=True,
)

features_schema = DataFrameSchema(
    {
        "transaction_id": Column(str, unique=True, checks=Check.str_startswith("TXN")),
        "customer_id": Column(str),
        "merchant_id": Column(str),
        "timestamp": Column("datetime64[ns]"),
        "amount": Column(float, checks=Check.gt(0)),
        "payment_method": Column(str, checks=Check.isin(PAYMENT_METHOD)),
        "hour_of_day": Column(int, checks=Check.in_range(0, 23)),
        "is_night": Column(int, checks=Check.isin({0, 1})),
        "is_cross_border": Column(int, checks=Check.isin({0, 1})),
        "amount_vs_customer_p95": Column(float, checks=Check.gt(0)),
        "txn_count_1h": Column(int, checks=Check.ge(0)),
        "txn_count_24h": Column(int, checks=Check.ge(0)),
        "is_new_device": Column(int, checks=Check.isin({0, 1})),
        "device_age_days": Column(float, checks=Check.ge(0)),
        "customer_device_count_so_far": Column(int, checks=Check.ge(0)),
        "merchant_fraud_rate_hist": Column(float, checks=Check.in_range(0, 1)),
        "is_new_country_for_customer": Column(int, checks=Check.isin({0, 1})),
        "ip_country_mismatch": Column(int, checks=Check.isin({0, 1})),
    },
    strict=True,
)

ALL_SCHEMAS = {
    "customers": customers_schema,
    "merchants": merchants_schema,
    "devices": devices_schema,
    "transactions": transactions_schema,
    "ground_truth": ground_truth_schema,
    "features": features_schema,
}
