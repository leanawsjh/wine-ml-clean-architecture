from dataclasses import dataclass

@dataclass
class TrainConfig:
    test_size: float = 0.2
    randon_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = None


if __name__ == "__main__":
    TrainConfig