import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class NumericScaler:
    def __init__(self):
        self._scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        self._scaler.fit(X)
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("NumericScaler must be fitted first")
        return self._scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class CategoricalEncoder:
    def __init__(self):
        self._encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> None:
        self._encoder.fit(X)
        self._is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("CategoricalEncoder must be fitted first")
        return self._encoder.transform(X)


class FeaturePipeline:
    def __init__(
        self,
        numeric_scaler: NumericScaler,
        categorical_encoder: CategoricalEncoder | None = None,
    ):
        self.numeric_scaler = numeric_scaler
        self.categorical_encoder = categorical_encoder

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray | None = None) -> None:
        self.numeric_scaler.fit(X_num)
        if self.categorical_encoder and X_cat is not None:
            self.categorical_encoder.fit(X_cat)

    def transform(
        self, X_num: np.ndarray, X_cat: np.ndarray | None = None
    ) -> np.ndarray:
        X_num_t = self.numeric_scaler.transform(X_num)

        if self.categorical_encoder and X_cat is not None:
            X_cat_t = self.categorical_encoder.transform(X_cat)
            return np.hstack([X_num_t, X_cat_t])

        return X_num_t
