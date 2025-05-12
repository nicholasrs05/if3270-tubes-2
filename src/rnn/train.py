from src.rnn.preprocessing import TextPreprocessor
from src.rnn.keras import RNNKerasModel
from src.rnn.model import RNNModel
from src.rnn.utils import load_nusax_data
import json
import os

FOLDER_PATH = "data"
OUTPUT_PATH = "src/rnn/output"


def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    train_texts, train_labels = load_nusax_data(FOLDER_PATH + "/train.csv")
    test_texts, test_labels = load_nusax_data(FOLDER_PATH + "/valid.csv")

    preprocessor = TextPreprocessor(
        max_tokens=10000, output_sequence_length=200, embedding_dim=100
    )

    preprocessor.fit(train_texts)

    train_sequences = preprocessor.preprocess(train_texts)
    test_sequences = preprocessor.preprocess(test_texts)

    keras_model = RNNKerasModel(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=100,
        rnn_units=128,
        dropout_rate=0.2,
        bidirectional=True,
    )

    history = keras_model.model.fit(
        train_sequences, train_labels, validation_split=0.1, epochs=10, batch_size=32
    )

    keras_model.save_weights(OUTPUT_PATH + "/keras_model.weights.h5")
    keras_model.save_dense_layer_weights(OUTPUT_PATH + "/dense_layer_weights.npz")
    keras_f1 = keras_model.evaluate(test_sequences, test_labels)

    scratch_model = RNNModel(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=100,
        rnn_units=128,
        dropout_rate=0.2,
        bidirectional=True,
    )

    scratch_model.load_dense_layer_weights(OUTPUT_PATH + "/dense_layer_weights.npz")
    scratch_model.save_weights(OUTPUT_PATH + "/scratch_model.weights.npy")
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

    with open(OUTPUT_PATH + "/training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Keras F1 Score: {keras_f1:.4f}")
    print(f"From-scratch F1 Score: {scratch_f1:.4f}")


if __name__ == "__main__":
    main()
