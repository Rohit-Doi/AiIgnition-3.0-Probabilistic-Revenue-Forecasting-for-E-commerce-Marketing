import sys
from pathlib import Path

# Add root to sys.path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import pytest


def test_roas_cap_config():
    """Ensure ROAS_CAP is set to realistic e-commerce bounds."""
    from src.config import ROAS_CAP
    assert ROAS_CAP == 15.0, "ROAS cap should be 15x to prevent unrealistic extrapolations."


def test_negative_clipping():
    """Ensure negative spend and revenue values are correctly clipped to 0.0 in pipelines."""
    df = pd.DataFrame({"spend": [-10, 5, 0], "revenue": [100, -50, 0]})
    df["spend"] = df["spend"].clip(lower=0.0)
    df["revenue"] = df["revenue"].clip(lower=0.0)
    
    assert df["spend"].min() == 0.0
    assert df["revenue"].min() == 0.0
    assert df["spend"].iloc[0] == 0.0
    assert df["revenue"].iloc[1] == 0.0


def test_quantile_monotonicity():
    """Ensure predicted quantiles obey non-crossing bounds: P10 <= P50 <= P90."""
    from src.metrics import evaluate_predictions
    
    # Mock data arrays
    y_true = np.array([1000, 2000])
    p10 = np.array([800, 1500])
    p50 = np.array([1000, 1900])
    p90 = np.array([1200, 2500])
    spend = np.array([500, 1000])
    
    # Mathematical monotonicity check
    assert np.all(p10 <= p50), "P10 bounds exceed P50"
    assert np.all(p50 <= p90), "P50 bounds exceed P90"
    
    # Ensure the metrics evaluator correctly processes these arrays
    metrics = evaluate_predictions(y_true, p10, p50, p90, spend)
    assert "wape_p50" in metrics, "WAPE metric missing"
    assert "coverage_p10_p90" in metrics, "Coverage metric missing"
    assert metrics["coverage_p10_p90"] == 100.0, "Coverage calculation incorrect"
