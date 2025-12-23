from wine_ml.config import TrainConfig
from wine_ml.data import load_data, split_data
from wine_ml.features import NumericScaler, FeaturePipeline
from wine_ml.model import WineModel
from wine_ml.trainer import Trainer
import mlflow


def main():
    config = TrainConfig()

    mlflow.set_experiment("wine_classification_experiments")

    with mlflow.start_run():
        # log parameters 
        mlflow.log_params(config.__dict__)

        # data
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, config.test_size, config.random_state
        )

        # features
        features = FeaturePipeline(
            numeric_scaler=NumericScaler()
        )

        # model
        model = WineModel(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
        )

        # training
        trainer = Trainer(model, features)
        trainer.train(X_train, y_train, X_test, y_test)

        # log metrics (outcome)
        metrics = trainer.get_metrics()
        mlflow.log_metrics(metrics)

        print("Metrics:", metrics)

        print("Metrics:", trainer.get_metrics())



if __name__ == "__main__":
    main()
