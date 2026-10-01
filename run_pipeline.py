"""End-to-end pipeline: data -> features -> both models -> out-of-fold validation -> report.

Usage
-----
    python run_pipeline.py                 # synthetic data (default)
    python run_pipeline.py --loans my_loans.csv --bounces my_bounces.csv

Validation is done with 5-fold stratified cross-validation: every loan is scored by
a model that never saw it during training, so the reported AUC / KS / capture rates
are out-of-sample rather than in-sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
from credit_risk import evaluation as ev
from credit_risk.features import (behavioural_features, build_model_dataset,
                                  standardise_bounce_reasons)
from credit_risk.models import (BANDS, MonitoringModel, ScreeningModel, risk_tier,
                                score_borrower, score_to_band)
from credit_risk.synthetic import generate_portfolio

ALL_MONTHS = [4, 5, 6, 7, 8]
OBSERVATION_MONTHS = [4, 5, 6]   # Model 2 only sees Apr-Jun behaviour ...
                                  # ... to predict NPAs by the August snapshot
FIG_DIR = Path("reports/figures")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.loans and args.bounces:
        return pd.read_csv(args.loans), pd.read_csv(args.bounces)
    syn = Path("data/synthetic")
    if not (syn / "loans.csv").exists():
        syn.mkdir(parents=True, exist_ok=True)
        loans, bounces = generate_portfolio()
        loans.to_csv(syn / "loans.csv", index=False)
        bounces.to_csv(syn / "bounces.csv", index=False)
    return pd.read_csv(syn / "loans.csv"), pd.read_csv(syn / "bounces.csv")


def out_of_fold_scores(data: pd.DataFrame, tier: pd.Series, seed: int = 7):
    """Score every loan with models trained on the other 4 folds."""
    s1 = np.zeros(len(data))
    s2 = np.zeros(len(data))
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(data, data["new_npa"]):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        s1[test_idx] = ScreeningModel().fit(train, tier.iloc[train_idx]).score(test)
        s2[test_idx] = MonitoringModel().fit(train, train["new_npa"]).score(test)
    return s1, s2


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(y, s1, s2, band_table, monitoring_coefs):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. ROC curves
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for score, colour, name in [(s1, BLUE, "Model 1 - screening"),
                                (s2, ORANGE, "Model 2 - early warning")]:
        fpr, tpr, _ = roc_curve(y, score)
        auc = ev.summary(y, score)["auc"]
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=f"{name} (AUC {auc:.2f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Random")
    style(ax, "ROC curve, out-of-fold", "False positive rate", "True positive rate")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=160)
    plt.close(fig)

    # 2. Capture curve: share of defaults caught vs share of book reviewed
    fig, ax = plt.subplots(figsize=(6, 4.5))
    rates = np.linspace(0.01, 1, 100)
    for score, colour, name in [(s1, BLUE, "Model 1 - screening"),
                                (s2, ORANGE, "Model 2 - early warning")]:
        ax.plot(rates * 100, [ev.capture_rate(y, score, r) * 100 for r in rates],
                color=colour, linewidth=2, label=name)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle="--", label="Random")
    style(ax, "Defaults caught vs. share of book reviewed",
          "Riskiest % of loans flagged", "% of new NPAs captured")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "capture_curve.png", dpi=160)
    plt.close(fig)

    # 3. Default rate by Model 1 band (should rise left to right)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(band_table.index, band_table["default_rate"] * 100, color=BLUE, width=0.6)
    for i, (rate, n) in enumerate(zip(band_table["default_rate"], band_table["loans"])):
        ax.text(i, rate * 100, f"{rate:.1%}\n(n={n})", ha="center", va="bottom",
                fontsize=8, color=MUTED)
    style(ax, "Model 1: new-NPA rate by risk band", "Risk band", "New-NPA rate (%)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_rate_by_band.png", dpi=160)
    plt.close(fig)

    # 4. Model 2 coefficients (fitted, not hand-picked weights)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = [c.replace("_", " ") for c in monitoring_coefs.index]
    ax.barh(labels, monitoring_coefs.values, color=ORANGE, height=0.5)
    style(ax, "Model 2: fitted logistic-regression coefficients",
          "Coefficient (log-odds per unit)", "")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "early_warning_coefficients.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loans")
    parser.add_argument("--bounces")
    args = parser.parse_args()

    loans, bounces = load_data(args)
    data = build_model_dataset(loans, bounces, OBSERVATION_MONTHS)

    # Model 1 target uses the full Apr-Aug history (it is the *outcome* being learned)
    full = behavioural_features(standardise_bounce_reasons(bounces), ALL_MONTHS)
    full = full.reindex(data["loan_id"]).fillna(0).reset_index(drop=True)
    tier = risk_tier(data, full["bounce_frequency"], full["max_streak"])

    y = data["new_npa"].to_numpy()
    s1, s2 = out_of_fold_scores(data, tier)

    metrics = {
        "portfolio": {"loans_total": int(len(loans)),
                      "existing_npa_excluded": int((loans["dpd_apr"] >= 90).sum()),
                      "loans_modelled": int(len(data)),
                      "new_npa": int(y.sum())},
        "model_1_screening": ev.summary(y, s1),
        "model_2_early_warning": ev.summary(y, s2),
    }
    band_table = ev.default_rate_by_band(score_to_band(s1), y, BANDS)

    # Final models trained on all data (for scoring new borrowers)
    screening = ScreeningModel().fit(data, tier)
    monitoring = MonitoringModel().fit(data, data["new_npa"])
    make_figures(y, s1, s2, band_table, monitoring.coefficients())

    example = score_borrower(
        screening, monitoring,
        applicant={"ltv": 0.60, "foir": 0.45, "tenure": 120, "roi": 13.5,
                   "income": 90000, "mob": 18, "cibil": 720, "product": "HL",
                   "profession": "Salaried", "state": "Delhi", "end_use": "Home Purchase"},
        behaviour={"bounce_frequency": 2, "max_streak": 2,
                   "low_balance_bounces": 1, "bounced_last_month": 1},
    )
    metrics["example_borrower"] = example

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2))
    band_table.to_csv("reports/default_rate_by_band.csv")
    print(json.dumps(metrics, indent=2))
    print("\nModel 1 default rate by band:\n", band_table)


if __name__ == "__main__":
    main()
