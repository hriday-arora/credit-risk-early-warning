"""Model validation metrics used in credit risk.

AUC  : probability the model ranks a random defaulter above a random non-defaulter
KS   : maximum separation between the score distributions of defaulters and
       non-defaulters (industry-standard scorecard metric; 0.4+ is considered strong)
Capture rate @ flag rate : if collections review only the riskiest X% of the book,
       what share of loans that later defaulted would they have caught?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    return float(ks_2samp(score[y_true == 1], score[y_true == 0]).statistic)


def capture_rate(y_true: np.ndarray, score: np.ndarray, flag_rate: float) -> float:
    """Share of defaulters inside the top `flag_rate` fraction of scores."""
    y_true = np.asarray(y_true)
    n_flag = int(np.ceil(flag_rate * len(score)))
    flagged = np.argsort(-np.asarray(score))[:n_flag]
    return float(y_true[flagged].sum() / y_true.sum())


def summary(y_true: np.ndarray, score: np.ndarray,
            flag_rates: tuple[float, ...] = (0.10, 0.20, 0.30)) -> dict:
    out = {
        "auc": round(float(roc_auc_score(y_true, score)), 3),
        "ks": round(ks_statistic(y_true, score), 3),
        "n_loans": int(len(y_true)),
        "n_defaults": int(np.sum(y_true)),
    }
    for fr in flag_rates:
        out[f"capture_at_{int(fr * 100)}pct_flagged"] = round(capture_rate(y_true, score, fr), 3)
    return out


def default_rate_by_band(bands: np.ndarray, y_true: np.ndarray, order: list[str]) -> pd.DataFrame:
    """A well-behaved scorecard shows default rates rising band by band."""
    df = pd.DataFrame({"band": bands, "default": y_true})
    table = df.groupby("band")["default"].agg(loans="size", defaults="sum", default_rate="mean")
    return table.reindex(order).fillna(0)
