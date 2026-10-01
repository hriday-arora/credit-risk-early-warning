"""Data cleaning and feature engineering.

1. Standardise free-text EMI bounce reasons into a small set of categories.
2. Build behavioural features per loan from the bounce history.
3. Define NPA labels from days-past-due (DPD) snapshots (RBI 90+ DPD rule).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw reason (lower-cased, stripped) -> standard category
REASON_MAP = {
    # insufficient balance family
    "insufficient balance": "low_balance",
    "balance insufficient": "low_balance",
    "insufficient funds": "low_balance",
    "fund insufficient": "low_balance",
    "fund insuffieient": "low_balance",
    "4-balance insufficient": "low_balance",
    # not presented family
    "not presented": "not_presented",
    "not present": "not_presented",
    "not to present": "not_presented",
    # account issues
    "account closed": "account_issue",
    "a/c blocked or frozen": "account_issue",
    "a/c blocked / frozen / closed": "account_issue",
    "a/c blocked / frozen": "account_issue",
    "68-a/c blocked or frozen": "account_issue",
    # mandate / NACH issues
    "mandate umrn is not present |": "mandate_issue",
    "nach not active": "mandate_issue",
    "99-mandate before start date": "mandate_issue",
    # technical / other
    "status awaited": "technical_issue",
    "registration not in active status": "technical_issue",
    "exceeds arrangements": "technical_issue",
    "exhausted": "technical_issue",
}


def standardise_bounce_reasons(bounces: pd.DataFrame) -> pd.DataFrame:
    """Add a `bounce_reason` column with standard categories ('other' if unmapped)."""
    out = bounces.copy()
    cleaned = out["bounce_reason_raw"].astype(str).str.strip().str.lower()
    out["bounce_reason"] = cleaned.map(REASON_MAP).fillna("other")
    return out


def max_consecutive_streak(months: list[int]) -> int:
    """Longest run of consecutive months with a bounce, e.g. [4, 5, 6, 8] -> 3."""
    if len(months) == 0:
        return 0
    months = sorted(set(months))
    best = current = 1
    for prev, nxt in zip(months, months[1:]):
        current = current + 1 if nxt == prev + 1 else 1
        best = max(best, current)
    return best


def behavioural_features(bounces: pd.DataFrame, months: list[int]) -> pd.DataFrame:
    """Per-loan bounce features using ONLY the given months (the observation window).

    Restricting the window matters: if bounces from the same months as the default
    outcome are used, the model 'sees' the default as it happens rather than
    predicting it in advance (target leakage).
    """
    window = bounces[bounces["month"].isin(months)]
    grouped = window.groupby("loan_id")
    feats = pd.DataFrame({
        "bounce_frequency": grouped["month"].nunique(),
        "max_streak": grouped["month"].apply(lambda m: max_consecutive_streak(list(m))),
        "low_balance_bounces": grouped["bounce_reason"].apply(lambda r: (r == "low_balance").sum()),
        "bounced_last_month": grouped["month"].max().eq(max(months)).astype(int),
    })
    return feats


def add_npa_labels(loans: pd.DataFrame) -> pd.DataFrame:
    """NPA = 90+ days past due.

    existing_npa : already 90+ DPD in April (excluded from modelling - already defaulted)
    new_npa      : below 90 DPD in April, 90+ DPD by August (the event we predict)
    """
    out = loans.copy()
    out["existing_npa"] = (out["dpd_apr"] >= 90).astype(int)
    out["new_npa"] = ((out["dpd_apr"] < 90) & (out["dpd_aug"] >= 90)).astype(int)
    return out


def clean_origination(loans: pd.DataFrame) -> pd.DataFrame:
    """Origination features: handle bureau no-hit (-1) and log-transform income."""
    out = loans.copy()
    out["cibil_no_hit"] = (out["cibil"] < 0).astype(int)
    median_cibil = out.loc[out["cibil"] > 0, "cibil"].median()
    out["cibil_clean"] = out["cibil"].where(out["cibil"] > 0, median_cibil)
    out["log_income"] = np.log(out["income"])
    return out


def build_model_dataset(loans: pd.DataFrame, bounces: pd.DataFrame,
                        observation_months: list[int]) -> pd.DataFrame:
    """Join everything into one modelling table of loans that were performing in April."""
    bounces = standardise_bounce_reasons(bounces)
    feats = behavioural_features(bounces, observation_months)

    data = add_npa_labels(clean_origination(loans))
    data = data.merge(feats, left_on="loan_id", right_index=True, how="left")
    feat_cols = ["bounce_frequency", "max_streak", "low_balance_bounces", "bounced_last_month"]
    data[feat_cols] = data[feat_cols].fillna(0).astype(int)

    return data[data["existing_npa"] == 0].reset_index(drop=True)
