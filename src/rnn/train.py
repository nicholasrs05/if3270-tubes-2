from src.rnn.preprocessing import TextPreprocessor
from src.rnn.keras import RNNKerasModel
from src.rnn.model import RNNModel
from src.rnn.utils import load_nusax_data
import json
import os

FOLDER_PATH = "data/nusax"
OUTPUT_PATH = "src/rnn/output"
WEIGHTS_FILE = "model_weights.npz"


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    train_texts, train_labels = load_nusax_data(FOLDER_PATH + "/train.csv")
    valid_texts, valid_labels = load_nusax_data(FOLDER_PATH + "/valid.csv")
    test_texts, test_labels = load_nusax_data(FOLDER_PATH + "/test.csv")

    preprocessor = TextPreprocessor(
        max_tokens=5000, output_sequence_length=54, embedding_dim=100
    )

    preprocessor.fit(train_texts)

    train_sequences = preprocessor.preprocess(train_texts)
    valid_sequences = preprocessor.preprocess(valid_texts)
    test_sequences = preprocessor.preprocess(test_texts)

    keras_model = RNNKerasModel(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=100,
        rnn_units=128,
        dropout_rate=0.2,
        bidirectional=True,
    )

    history = keras_model.model.fit(
        train_sequences,
        train_labels,
        validation_data=(valid_sequences, valid_labels),
        epochs=10,
        batch_size=32,
        verbose=0,
    )

    weights_path = os.path.join(OUTPUT_PATH, WEIGHTS_FILE)
    keras_model.save_weights(weights_path)

    keras_f1 = keras_model.evaluate(test_sequences, test_labels)

    scratch_model = RNNModel(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=100,
        rnn_units=128,
        dropout_rate=0.2,
        bidirectional=True,
    )

    scratch_model.load_weights(weights_path)
    scratch_f1 = scratch_model.evaluate(test_sequences, test_labels)

    results = {
        "keras_f1_score": float(keras_f1),
        "scratch_f1_score": float(scratch_f1),
        "training_history": {
            "loss": [float(x) for x in history.history["loss"]],
            "accuracy": [float(x) for x in history.history["accuracy"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
            "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        },
    }

    with open(os.path.join(OUTPUT_PATH, "training_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Keras F1 Score: {keras_f1:.4f}")
    print(f"From-scratch F1 Score: {scratch_f1:.4f}")


if __name__ == "__main__":
    for i in range(10):
        main()
