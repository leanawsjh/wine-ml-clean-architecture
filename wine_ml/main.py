from wine_ml.config import TrainConfig
from wine_ml.data import load_data, split_data
from wine_ml.features import NumericScaler, FeaturePipeline
from wine_ml.model import WineModel
from wine_ml.trainer import Trainer


def main():
    config = TrainConfig()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(
        X, y, config.test_size, config.random_state
    )

    features = FeaturePipeline(
        numeric_scaler=NumericScaler()
    )

    model = WineModel(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
    )

    trainer = Trainer(model, features)
    trainer.train(X_train, y_train, X_test, y_test)

    print("Metrics:", trainer.get_metrics())


if __name__ == "__main__":
    main()
