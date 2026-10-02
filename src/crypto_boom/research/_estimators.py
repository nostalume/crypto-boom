"""Fixed quantile-tree recipe shared by the three path research workflows."""

from sklearn.ensemble import HistGradientBoostingRegressor


def path_regressor(quantile: float, *, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        learning_rate=0.08,
        early_stopping=False,
        random_state=0,
    )
