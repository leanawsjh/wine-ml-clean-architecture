import numpy as np
from sklearn.ensemble import RandomForestClassifier


class WineModel:
    def __init__(self, n_estimators: int, max_depth: int | None):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self._model.predict(X)
