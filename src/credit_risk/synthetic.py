"""Generate a synthetic MSME / retail loan book with the same structure as the
original (confidential) portfolio data.

No real borrower data is used anywhere in this repository. Every value below is
simulated from a latent "true risk" variable so that the modelling pipeline
behaves realistically, but the numbers will not match the original portfolio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = [4, 5, 6, 7, 8]  # April-August observation window

# Raw bounce reasons exactly as messy as they arrive from bank / NACH files
RAW_REASONS = {
    "low_balance": [
        "Insufficient Balance", "BALANCE INSUFFICIENT", "Insufficient Funds",
        "fund insufficient", "fund insuffieient", "4-Balance Insufficient",
    ],
    "not_presented": ["Not Presented", "not to present", "Not Present"],
    "account_issue": [
        "A/C Blocked or Frozen", "a/c blocked / frozen / closed",
        "68-A/C Blocked or Frozen", "Account Closed",
    ],
    "mandate_issue": [
        "Mandate UMRN is not present |", "NACH not active",
        "99-Mandate before start date",
    ],
    "technical_issue": ["Status Awaited", "Exceeds arrangements", "Exhausted"],
}


def generate_portfolio(n_loans: int = 811, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (loans, bounces) DataFrames.

    loans   : one row per loan with origination attributes and DPD snapshots
    bounces : one row per failed EMI collection (loan_id, month, raw reason)
    """
    rng = np.random.default_rng(seed)

    # Latent borrower risk (unobserved in practice) ~ N(0, 1)
    risk = rng.normal(0, 1, n_loans)

    product = rng.choice(["HL", "LAP", "PL"], n_loans, p=[0.45, 0.35, 0.20])
    profession = rng.choice(["Salaried", "Self-Employed"], n_loans, p=[0.55, 0.45])
    state = rng.choice(
        ["Haryana", "Uttar Pradesh", "Uttarakhand", "Delhi", "Rajasthan"],
        n_loans, p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )
    end_use = rng.choice(
        ["Home Purchase", "Construction", "Business Expansion", "Working Capital"],
        n_loans, p=[0.35, 0.20, 0.25, 0.20],
    )

    # Origination attributes: weakly correlated with latent risk (as in reality)
    cibil = np.clip(np.round(740 - 35 * risk + rng.normal(0, 45, n_loans)), 550, 880)
    no_hit = rng.random(n_loans) < 0.06                      # new-to-credit borrowers
    cibil = np.where(no_hit, -1, cibil)                      # bureau sends -1 for no-hit

    ltv = np.clip(0.50 + 0.06 * risk + rng.normal(0, 0.12, n_loans), 0.10, 0.85).round(3)
    foir = np.clip(0.45 + 0.05 * risk + rng.normal(0, 0.10, n_loans), 0.10, 0.80).round(3)
    income = np.round(np.exp(rng.normal(11.2, 0.45, n_loans) - 0.08 * risk), -2)
    tenure = rng.choice([60, 84, 120, 180, 240], n_loans, p=[0.2, 0.2, 0.3, 0.2, 0.1])
    roi = np.clip(13.5 + 0.8 * risk + rng.normal(0, 1.0, n_loans), 10.5, 20.0).round(2)
    mob = rng.integers(3, 30, n_loans)                       # months on book
    exposure = np.round(np.exp(rng.normal(13.0, 0.5, n_loans)), -2)  # INR outstanding

    # Days-past-due snapshots and the (future) default event
    dpd_apr = np.where(risk > 1.55, rng.integers(90, 400, n_loans),
                       np.where(risk > 1.0, rng.integers(0, 60, n_loans), 0))
    p_new_npa = 1 / (1 + np.exp(-(-3.7 + 1.4 * risk)))
    becomes_npa = (dpd_apr < 90) & (rng.random(n_loans) < p_new_npa)
    dpd_aug = np.where(dpd_apr >= 90, dpd_apr + 120,
                       np.where(becomes_npa, rng.integers(90, 200, n_loans),
                                np.clip(dpd_apr + rng.integers(-30, 30, n_loans), 0, 89)))

    # Monthly EMI bounces: baseline driven by latent risk, plus a rising
    # deterioration signal in the months before a loan tips into NPA
    rows = []
    reason_mix = {"low_balance": 0.50, "not_presented": 0.44, "account_issue": 0.03,
                  "mandate_issue": 0.02, "technical_issue": 0.01}
    for i in range(n_loans):
        for m_idx, month in enumerate(MONTHS):
            deterioration = (0.9 + 0.5 * m_idx) if becomes_npa[i] else 0.0
            p = 1 / (1 + np.exp(-(-1.3 + 0.9 * risk[i] + deterioration)))
            if rng.random() < p:
                cat = rng.choice(list(reason_mix), p=list(reason_mix.values()))
                if becomes_npa[i] and rng.random() < 0.3:
                    cat = "low_balance"
                rows.append({
                    "loan_id": f"SYN-{i:05d}",
                    "month": month,
                    "emi": round(float(exposure[i]) / tenure[i] * 1.6, 0),
                    "bounce_reason_raw": rng.choice(RAW_REASONS[cat]),
                })
    bounces = pd.DataFrame(rows)

    loans = pd.DataFrame({
        "loan_id": [f"SYN-{i:05d}" for i in range(n_loans)],
        "product": product, "profession": profession, "state": state, "end_use": end_use,
        "ltv": ltv, "foir": foir, "income": income, "tenure": tenure, "roi": roi,
        "mob": mob, "cibil": cibil, "exposure": exposure,
        "dpd_apr": dpd_apr, "dpd_aug": dpd_aug,
    })
    return loans, bounces


if __name__ == "__main__":
    loans, bounces = generate_portfolio()
    loans.to_csv("data/synthetic/loans.csv", index=False)
    bounces.to_csv("data/synthetic/bounces.csv", index=False)
    print(f"loans: {loans.shape}, bounces: {bounces.shape}")
