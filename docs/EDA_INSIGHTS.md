# EDA Insights — Sub-project 2

Numbered business insights from `notebooks/01-eda.ipynb`, copied here so
they're readable without opening the notebook. Regenerate by re-running the
notebook top to bottom against a fresh `make seed` and updating both files.

1. Overall fraud prevalence is **1.51%** (7,833 of 519,876 transactions).
2. Fraud transactions have a **1.41x** higher median amount than legitimate ones ($61.41 vs $43.43).
3. The riskiest hour of day is **02:00**, at **2.01%** fraud rate vs the overall **1.51%**.
4. The riskiest merchant category is **crypto**, at **4.82%** fraud rate.
5. New-device transactions are **2.70x** more likely to be fraud than established-device transactions (**3.71%** vs **1.37%**), using the real `is_new_device` feature from `features/device.py` rather than a timestamp-proximity proxy.
6. Cross-border transactions are **2.04x** more likely to be fraud than domestic ones (**2.98%** vs **1.46%**).
