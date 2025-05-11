from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
import json


def load_mnist_data():
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

    return X, y


def one_hot_encode(labels, num_classes=10):
    labels = labels.astype(int)
    return np.eye(num_classes)[labels]


def split_data(X, y, val_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


def batch_iterator(X, y, batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_comparison(results, metric):
    for label, data in results.items():
        plt.plot(data["epochs"], data[metric], label=label)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def get_pyvis_config():
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "levelSeparation": 150,
                "nodeSpacing": 100,
                "treeSpacing": 0,
                "direction": "LR",
                "sortMethod": "directed",
            }
        },
        "edges": {
            "arrows": {"to": {"enabled": True}},
            "font": {"size": 12, "align": "middle"},
            "color": {"inherit": "source"},
        },
    }

    return json.dumps(options)


def accuracy_score(predictions, y_true):
    return np.mean(predictions == y_true)
