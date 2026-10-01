import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from credit_risk.features import (add_npa_labels, behavioural_features,
                                  max_consecutive_streak, standardise_bounce_reasons)


def test_streak():
    assert max_consecutive_streak([4, 5, 6, 8]) == 3
    assert max_consecutive_streak([6, 8]) == 1
    assert max_consecutive_streak([]) == 0
    assert max_consecutive_streak([5, 4, 4, 6]) == 3   # unordered + duplicates


def test_reason_mapping():
    df = pd.DataFrame({"bounce_reason_raw": [" Insufficient Balance ", "NOT TO PRESENT",
                                             "68-A/C Blocked or Frozen", "something new"]})
    out = standardise_bounce_reasons(df)["bounce_reason"].tolist()
    assert out == ["low_balance", "not_presented", "account_issue", "other"]


def test_observation_window_excludes_later_months():
    df = pd.DataFrame({"loan_id": ["A"] * 4, "month": [4, 5, 7, 8],
                       "bounce_reason_raw": ["insufficient funds"] * 4})
    feats = behavioural_features(standardise_bounce_reasons(df), [4, 5, 6])
    assert feats.loc["A", "bounce_frequency"] == 2


def test_npa_labels():
    df = pd.DataFrame({"dpd_apr": [0, 120, 30], "dpd_aug": [95, 240, 60]})
    out = add_npa_labels(df)
    assert out["new_npa"].tolist() == [1, 0, 0]
    assert out["existing_npa"].tolist() == [0, 1, 0]
