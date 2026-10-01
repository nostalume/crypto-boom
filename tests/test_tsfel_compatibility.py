"""Pinned optional library compatibility, not a feature-selection experiment."""

import numpy as np
import pandas as pd
import pytest
import tsfel

pytestmark = pytest.mark.research


def test_bounded_tsfel_batch_matches_scalar_and_numpy():
    config = tsfel.get_features_by_domain()
    selection = {
        "statistical": [
            "Mean",
            "Standard deviation",
            "Interquartile range",
            "Skewness",
            "Kurtosis",
        ],
        "temporal": ["Mean absolute diff", "Autocorrelation", "Zero crossing rate"],
    }
    config = {
        d: {name: config[d][name] for name in names} for d, names in selection.items()
    }
    t = np.arange(120, dtype=float)
    windows = np.stack(
        [
            np.column_stack((np.sin(t / 9), np.zeros(120))),
            np.column_stack((t / 120, np.ones(120))),
            np.column_stack(((-1.0) ** t, np.cos(t / 7))),
        ]
    )
    # TSFEL 0.2.0 uses a process pool even for n_jobs=1; None is serial.
    batch = tsfel.time_series_features_extractor(
        config, windows, fs=1 / 60, verbose=0, n_jobs=None
    )
    assert batch.shape == (3, 16)
    assert batch.loc[:1, "1_Skewness"].isna().all()
    assert batch.loc[:1, "1_Kurtosis"].isna().all()
    for i, window in enumerate(windows):
        single = tsfel.time_series_features_extractor(
            config, window, fs=1 / 60, verbose=0, n_jobs=None
        )
        pd.testing.assert_frame_equal(batch.iloc[[i]].reset_index(drop=True), single)
        for channel in range(2):
            np.testing.assert_allclose(
                batch.iloc[i][f"{channel}_Mean"], window[:, channel].mean()
            )
            np.testing.assert_allclose(
                batch.iloc[i][f"{channel}_Standard deviation"], window[:, channel].std()
            )
    repeat = tsfel.time_series_features_extractor(
        config, windows[::-1].copy(), fs=1 / 60, verbose=0, n_jobs=None
    )
    pd.testing.assert_frame_equal(batch.iloc[::-1].reset_index(drop=True), repeat)
