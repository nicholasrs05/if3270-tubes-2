from cnn.layer import LayerScratch, InputScratch, Conv2DScratch, MaxPooling2DScratch, FlattenScratch, DenseScratch
import numpy as np

class CNNScratch:
    def __init__(self, layers: list[LayerScratch]):
        self.layers = layers

    def transfer_weights(self, keras_model):
        keras_layers = [layer for layer in keras_model.layers if len(layer.get_weights()) > 0]
        keras_idx = 0
        for layer in self.layers:
            if isinstance(layer, (Conv2DScratch, DenseScratch)):
                keras_weights = keras_layers[keras_idx].get_weights()
                weights = keras_weights[0]
                bias = keras_weights[1]
                layer.set_weights((weights, bias))
                keras_idx += 1

    def predict(self, input, batch_size=32):
        outputs = []
        for i in range(0, len(input), batch_size):
            print(f'Processing {i // batch_size + 1} / {len(input) // batch_size + 1}')
            x_batch = input[i:i+batch_size]
            x = x_batch
            for layer in self.layers:
                x = layer.forward(x)
            outputs.append(x)

        return np.concatenate(outputs, axis=0)