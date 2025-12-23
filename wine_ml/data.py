from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from typing import Tuple
import numpy as np


def load_data() -> Tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    return data.data, data.target

def split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
        random_state: int 
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)