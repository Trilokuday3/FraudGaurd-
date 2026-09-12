"""Canonical value sets for every categorical column in the data model.

These frozensets are the single source of truth for "what values are allowed".
The pandera schemas (fraudguard_core.schemas) turn each into an `isin` check,
and the generator draws only from these. Change a category here and every
validation and generator picks it up.
"""

COUNTRIES = frozenset({"US", "GB", "IN", "CA", "AU", "DE", "SG"})
COUNTRY_WEIGHTS = {
    "US": 0.45,
    "GB": 0.15,
    "IN": 0.15,
    "CA": 0.08,
    "AU": 0.07,
    "DE": 0.06,
    "SG": 0.04,
}
CITIES_BY_COUNTRY = {
    "US": ["New York", "Chicago", "Austin", "Seattle", "Miami"],
    "GB": ["London", "Manchester", "Bristol"],
    "IN": ["Bengaluru", "Mumbai", "Hyderabad", "Pune"],
    "CA": ["Toronto", "Vancouver"],
    "AU": ["Sydney", "Melbourne"],
    "DE": ["Berlin", "Munich"],
    "SG": ["Singapore"],
}

CUSTOMER_SEGMENT = frozenset({"retail", "premium", "business"})

MERCHANT_CATEGORY = frozenset(
    {
        "grocery",
        "electronics",
        "travel",
        "gaming",
        "crypto",
        "fashion",
        "dining",
        "other",
    }
)
# Latent per-category fraud-rate multipliers (generator-internal only; never
# serialized to the merchants table — see leakage guards in the sub-project 1
# design doc).
MERCHANT_CATEGORY_RISK_MULTIPLIER = {
    "crypto": 4.0,
    "gaming": 2.2,
    "electronics": 1.8,
    "travel": 1.4,
    "fashion": 1.0,
    "other": 1.0,
    "dining": 0.6,
    "grocery": 0.4,
}

DEVICE_TYPE = frozenset({"mobile", "desktop", "tablet"})

CURRENCY = frozenset({"USD", "GBP", "INR", "EUR"})
PAYMENT_METHOD = frozenset({"card", "wallet", "bank_transfer"})
