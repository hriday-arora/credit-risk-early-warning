"""Two-stage credit risk framework.

Model 1 - Origination screening (ordinal logistic regression)
    Uses only information known at the time of the loan application (CIBIL, LTV,
    FOIR, income, ROI, tenure, product, profession, state, end use) to place each
    borrower on a 0-4 risk ladder. Used to approve / review / reject new loans.

Model 2 - Behavioural early warning (logistic regression)
    Uses repayment behaviour in an observation window (EMI bounce frequency,
    longest bounce streak, low-balance bounces, bounce in the latest month) to
    estimate the probability that a currently performing loan becomes an NPA
    within the next few months. Used to prioritise collections.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel

SCREENING_NUMERIC = ["ltv", "foir", "tenure", "roi", "log_income", "mob",
                     "cibil_clean", "cibil_no_hit"]
SCREENING_CATEGORICAL = ["product", "profession", "state", "end_use"]
MONITORING_FEATURES = ["bounce_frequency", "max_streak", "low_balance_bounces",
                       "bounced_last_month"]

RISK_TIERS = ["Healthy", "Low", "Moderate", "High", "Severe"]
BANDS = ["Very Low", "Low", "Moderate", "High", "Very High"]


def risk_tier(data: pd.DataFrame, full_bounce_freq: pd.Series,
              full_max_streak: pd.Series) -> pd.Series:
    """Ordinal 0-4 target for Model 1, built from what actually happened to the loan.

    4 Severe   : became NPA
    3 High     : 4+ bounces or 3+ month streak AND DPD worsened by 30+ days
    2 Moderate : 2+ bounces or a 2-month streak
    1 Low      : a single bounce
    0 Healthy  : no bounces
    """
    dpd_change = data["dpd_aug"] - data["dpd_apr"]
    conditions = [
        data["new_npa"] == 1,
        ((full_bounce_freq >= 4) | (full_max_streak >= 3)) & (dpd_change >= 30),
        (full_bounce_freq >= 2) | (full_max_streak == 2),
        full_bounce_freq == 1,
    ]
    return pd.Series(np.select(conditions, [4, 3, 2, 1], default=0), index=data.index)


def score_to_band(score_0_100: np.ndarray | float) -> np.ndarray:
    """Map a 0-100 score to five equal-width risk bands."""
    idx = np.clip((np.asarray(score_0_100) // 20).astype(int), 0, 4)
    return np.array(BANDS)[idx]


class ScreeningModel:
    """Model 1: ordinal logit on origination data, output rescaled to 0-100."""

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "ScreeningModel":
        X = self._encode(data, fit=True)
        self.scaler = StandardScaler().fit(X)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        self.result = OrderedModel(target, Xs, distr="logit").fit(method="bfgs", disp=False)
        raw = self._expected_tier(Xs)
        # Store training range so new applicants are scaled consistently
        self.raw_min, self.raw_max = float(raw.min()), float(raw.max())
        return self

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """0-100 risk score (higher = riskier)."""
        X = self._encode(data, fit=False)
        Xs = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
        raw = self._expected_tier(Xs)
        return np.clip((raw - self.raw_min) / (self.raw_max - self.raw_min) * 100, 0, 100)

    def coefficients(self) -> pd.Series:
        n_thresholds = self.result.model.k_levels - 1
        return self.result.params.iloc[:-n_thresholds].sort_values()

    # -- helpers ---------------------------------------------------------------
    def _encode(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = pd.get_dummies(data[SCREENING_NUMERIC + SCREENING_CATEGORICAL],
                           columns=SCREENING_CATEGORICAL, drop_first=False, dtype=float)
        if fit:
            # drop one reference level per category (avoids perfect collinearity,
            # which is what broke the Hessian inversion in the first version)
            refs = [f"{c}_{data[c].mode()[0]}" for c in SCREENING_CATEGORICAL]
            self.columns = [c for c in X.columns if c not in refs]
            self.medians = X[self.columns].median()
        X = X.reindex(columns=self.columns, fill_value=0.0)
        return X.fillna(self.medians)

    def _expected_tier(self, Xs: pd.DataFrame) -> np.ndarray:
        probs = self.result.model.predict(self.result.params, exog=Xs)
        return probs @ np.arange(probs.shape[1])


class MonitoringModel:
    """Model 2: logistic regression on behavioural features -> P(new NPA)."""

    def fit(self, data: pd.DataFrame, target: pd.Series) -> "MonitoringModel":
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.model.fit(data[MONITORING_FEATURES], target)
        return self

    def probability(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(data[MONITORING_FEATURES])[:, 1]

    def score(self, data: pd.DataFrame) -> np.ndarray:
        """0-100 score: probability expressed as a percentage."""
        return self.probability(data) * 100

    def coefficients(self) -> pd.Series:
        return pd.Series(self.model.coef_[0], index=MONITORING_FEATURES).sort_values()


def score_borrower(screening: ScreeningModel, monitoring: MonitoringModel,
                   applicant: dict, behaviour: dict | None = None) -> dict:
    """Score one borrower. `behaviour` is optional (new applicants have no history)."""
    row = pd.DataFrame([applicant])
    row["cibil_no_hit"] = int(row.loc[0, "cibil"] < 0)
    row["cibil_clean"] = row["cibil"].where(row["cibil"] > 0, np.nan)
    row["log_income"] = np.log(row["income"])
    s1 = float(screening.score(row)[0])
    result = {"screening_score": round(s1, 1), "screening_band": str(score_to_band(s1))}
    if behaviour is not None:
        s2 = float(monitoring.score(pd.DataFrame([behaviour]))[0])
        result.update({"early_warning_score": round(s2, 1),
                       "early_warning_band": str(score_to_band(s2))})
    return result
