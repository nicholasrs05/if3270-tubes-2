from src.ffnn.layer import Layer
from src.ffnn.utils import batch_iterator, get_pyvis_config
from src.ffnn.loss import LossFunction, CategoricalCrossentropy
from src.ffnn.activation import Activation, Softmax
from src.ffnn.initializer import Initializer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pyvis.network import Network
from tqdm import tqdm


class FFNNModel:
    def __init__(
        self,
        layers: list[int],
        activation_functions: list[Activation],
        loss_function: LossFunction,
        weight_initializer: list[Initializer],
        learning_rate: float = 0.01,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.1,
    ):
        self.layers: list[Layer] = []
        self.activation_functions = activation_functions
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.weight_initializer = weight_initializer
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        if (isinstance(activation_functions[-1], Softmax)) and (
            not isinstance(loss_function, CategoricalCrossentropy)
        ):
            print(
                "Warning: Not pairing Softmax with CategoricalCrossentropy on the last layer may lead to incorrect results."
            )
        elif (isinstance(loss_function, CategoricalCrossentropy)) and (
            not isinstance(activation_functions[-1], Softmax)
        ):
            print(
                "Warning: Not pairing CategoricalCrossentropy with Softmax on the last layer may lead to incorrect results."
            )
        elif (layers[-1] != 1) and (not isinstance(activation_functions[-1], Softmax)):
            print(
                "Warning: Not using softmax on output layer with more than 1 neuron may lead to incorrect results."
            )
        elif (layers[-1] != 1) and (
            not isinstance(loss_function, CategoricalCrossentropy)
        ):
            print(
                "Warning: Not using CategoricalCrossentropy on output layer with more than 1 neuron may lead to incorrect results."
            )

        for i in range(1, len(layers)):
            self.add_layer(
                layers[i - 1],
                layers[i],
                activation_functions[i - 1],
                weight_initializer[i - 1],
            )

        self.training_history = {"train_loss": [], "val_loss": []}

    def add_layer(
        self,
        input_dim: int,
        neuron_count: int,
        activation: Activation,
        weight_initializer: Initializer,
    ):
        layer = Layer(
            input_dim,
            neuron_count,
            activation,
            weight_initializer,
        )

        self.layers.append(layer)

    def forward(self, X_train):
        activations = [X_train]

        for layer in self.layers:
            layer.Z = np.dot(activations[-1], layer.weights) + layer.biases
            layer.A = layer.activation.forward(layer.Z)
            activations.append(layer.A)

        return activations

    def backward(self, X_train, y_train, activations):
        batch_size = X_train.shape[0]
        dE_do = self.loss_function.backward(activations[-1], y_train)
        gradients = []

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if (
                i == len(self.layers) - 1
                and isinstance(layer.activation, Softmax)
                and isinstance(self.loss_function, CategoricalCrossentropy)
            ):
                dE_dnet = activations[-1] - y_train
            elif i == len(self.layers) - 1 and isinstance(layer.activation, Softmax):
                jacobian = layer.activation.backward(activations[-1])
                dE_dnet = np.einsum("ijk,ik->ij", jacobian, dE_do)
            else:
                do_dnet = layer.activation.backward(layer.A)
                dE_dnet = dE_do * do_dnet

            dnet_dw = activations[i].T
            dE_dw = np.dot(dnet_dw, dE_dnet) / batch_size
            dE_db = np.sum(dE_dnet, axis=0, keepdims=True) / batch_size

            combined_gradients = np.concatenate((dE_dw.flatten(), dE_db.flatten()))
            layer.gradients = combined_gradients

            gradients.append((dE_dw, dE_db))

            if i > 0:
                dE_do = np.dot(dE_dnet, layer.weights.T)

        gradients.reverse()
        return gradients

    def update_weights(self, gradients):
        for i, (dE_dw, dE_db) in enumerate(gradients):
            dE_dw += self.l1_lambda * np.sign(self.layers[i].weights)
            dE_dw += 2 * self.l2_lambda * self.layers[i].weights

            dE_dw = np.clip(dE_dw, -5, 5)
            dE_db = np.clip(dE_db, -5, 5)

            self.layers[i].weights -= self.learning_rate * dE_dw
            self.layers[i].biases -= self.learning_rate * dE_db.squeeze()

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        verbose=1,
    ):
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = len(X_train) // batch_size

            with tqdm(
                total=num_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit=" batch"
            ) as pbar:
                for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
                    activations = self.forward(X_batch)
                    gradients = self.backward(X_batch, y_batch, activations)
                    self.update_weights(gradients)

                    batch_loss = (
                        self.loss_function.compute(activations[-1], y_batch)
                        / batch_size
                    )
                    epoch_loss += batch_loss

                    pbar.update(1)

            epoch_loss /= num_batches
            self.training_history["train_loss"].append(epoch_loss)

            val_loss = None
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val, training=True)
                val_loss = self.loss_function.compute(val_predictions, y_val)
                self.training_history["val_loss"].append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - train_loss: {epoch_loss} "
                    + (f"- val_loss: {val_loss}" if val_loss is not None else "")
                )

    def predict(self, X, training=False):
        if training:
            return self.forward(X)[-1]

        probabilities = self.forward(X)[-1]

        if isinstance(self.loss_function, CategoricalCrossentropy):
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = (probabilities > 0.5).astype(int)

        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self.loss_function(y, predictions)
        return loss

    def save(self, file_path):
        import pickle

        with open(file_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_path):
        import pickle

        with open(file_path, "rb") as file:
            return pickle.load(file)

    def plot_model(self):
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            notebook=True,
            cdn_resources="remote",
        )

        net.toggle_physics(False)

        layer_colors = [
            mpl.colors.rgb2hex(mpl.colormaps["tab10"](step))
            for step in (np.linspace(0, 1, 10))
        ]

        net.set_options(get_pyvis_config())
        layer_positions = {}

        layer_positions[0] = []
        for i in range(self.layers[0].weights.shape[0]):
            node_id = f"L0N{i}"
            layer_positions[0].append(node_id)
            net.add_node(
                node_id, label=node_id, title=f"Input Layer Neuron {i}", level=0
            )

        for i, layer in enumerate(self.layers):
            layer_positions[i + 1] = []

            num_neurons = layer.weights.shape[1]
            layer_color = layer_colors[(i + 1) % len(layer_colors)]

            for j in range(num_neurons):
                node_id = f"L{i + 1}N{j}"
                layer_positions[i + 1].append(node_id)
                net.add_node(
                    node_id,
                    label=node_id,
                    title=(
                        (
                            f"Layer {i + 1}"
                            if i < len(self.layers) - 1
                            else "Output Layer"
                        )
                        + f" Neuron {j}"
                    ),
                    level=i + 1,
                    color=layer_color,
                )

        for i in range(len(self.layers)):
            weights = self.layers[i].weights

            for src_idx, src in enumerate(layer_positions[i]):
                for dst_idx, dst in enumerate(layer_positions[i + 1]):
                    weight = weights[src_idx][dst_idx]
                    net.add_edge(src, dst, title=f"Weight: {weight:.3f}")

        return net.show("../output/output.html")

    def plot_model_matplotlib(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        layer_x_positions = np.linspace(0, 1, len(self.layers) + 1)
        layer_y_positions = {}

        for i, layer in enumerate(self.layers):
            num_neurons = (
                layer.weights.shape[1]
                if i < len(self.layers)
                else self.layers[-1].weights.shape[1]
            )
            layer_y_positions[i + 1] = np.linspace(0, 1, num_neurons)

        layer_y_positions[0] = np.linspace(0, 1, self.layers[0].weights.shape[0])

        for i, layer in enumerate(self.layers):
            weights = layer.weights
            for src_idx, src_y in enumerate(layer_y_positions[i]):
                for dst_idx, dst_y in enumerate(layer_y_positions[i + 1]):
                    weight = weights[src_idx, dst_idx]
                    ax.plot(
                        [layer_x_positions[i], layer_x_positions[i + 1]],
                        [src_y, dst_y],
                        color="gray",
                        alpha=0.5,
                        lw=abs(weight) * 2,
                    )

        for i in range(len(self.layers) + 1):
            for y in layer_y_positions[i]:
                ax.scatter(
                    layer_x_positions[i],
                    y,
                    color="blue",
                    s=100,
                    edgecolors="black",
                    zorder=3,
                )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.show()

    def plot_layer_weights_distribution(self, layer_idx):
        weights = self.layers[layer_idx].weights.flatten()
        plt.hist(weights, bins=30)
        plt.title(f"Layer {layer_idx} Weights Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.show()

    def plot_all_layers_weights_distribution(self):
        for i in range(len(self.layers)):
            self.plot_layer_weights_distribution(i)

    def plot_layer_gradients_distribution(self, layer_idx):
        gradients = self.layers[layer_idx].gradients.flatten()
        plt.hist(gradients, bins=30)
        plt.title(f"Layer {layer_idx + 1} Gradients Distribution")
        plt.xlabel("Gradient")
        plt.ylabel("Frequency")
        plt.show()

    def plot_all_layers_gradients_distribution(self):
        for i in range(len(self.layers)):
            self.plot_layer_gradients_distribution(i)

    def plot_training_history(self):
        plt.plot(self.training_history["train_loss"], label="Training Loss")
        if "val_loss" in self.training_history:
            plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
