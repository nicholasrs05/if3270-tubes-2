import pandas as pd
from typing import Tuple
import numpy as np


def load_nusax_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(filepath)
    texts = df["text"].values

    label_map = {"neutral": 0, "negative": 1, "positive": 2}
    labels = np.array([label_map[label] for label in df["label"].values])

    return texts, labels
