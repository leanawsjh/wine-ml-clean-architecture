import numpy as np
from sklearn.metrics import accuracy_score
from wine_ml.features import FeaturePipeline
from wine_ml.model import WineModel


class Trainer:
    def __init__(self, model: WineModel, features: FeaturePipeline):
        self.model = model
        self.features = features
        self._is_trained = False
        self.metrics: dict[str, float] = {}

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.features.fit(X_train)
        X_train_t = self.features.transform(X_train)
        X_test_t = self.features.transform(X_test)

        self.model.fit(X_train_t, y_train)
        preds = self.model.predict(X_test_t)

        self.metrics["accuracy"] = accuracy_score(y_test, preds)
        self._is_trained = True

    def get_metrics(self) -> dict[str, float]:
        if not self._is_trained:
            raise RuntimeError("Trainer has not been run")
        return self.metrics
