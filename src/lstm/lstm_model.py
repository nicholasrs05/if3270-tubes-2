import numpy as np
import os
import sys
import tensorflow as tf
from typing import List, Union, Optional, Dict, Any


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .embedding_layer import EmbeddingLayer
from .lstm_layer import LSTMLayer, BidirectionalLSTM
from .dropout_layer import DropoutLayer
from .text_preprocessor import TextPreprocessor


try:
    from ffnn.layer import DenseLayer
except ImportError:
    print("Warning: Could not import FFNN DenseLayer. Using simple implementation.")
    
    class DenseLayer:
        def __init__(self, units, activation='linear'):
            self.units = units
            self.activation = activation
            self.weights = None
            self.bias = None
            
        def load_weights(self, weights, bias):
            self.weights = weights
            self.bias = bias
            
        def forward(self, inputs):
            output = np.dot(inputs, self.weights) + self.bias
            
            if self.activation == 'relu':
                output = np.maximum(0, output)
            elif self.activation == 'softmax':
                exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
                output = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
                
            return output

class LSTMModel:
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.layers = []
        self.preprocessor = preprocessor
        self.label_mapping = None
        
    def add_embedding(self, vocab_size, embedding_dim, input_length=None):
        layer = EmbeddingLayer(vocab_size, embedding_dim, input_length)
        self.layers.append(layer)
        
    def add_lstm(self, units, return_sequences=False, bidirectional=False):
        if bidirectional:
            layer = BidirectionalLSTM(units, return_sequences)
        else:
            layer = LSTMLayer(units, return_sequences)
        self.layers.append(layer)
        
    def add_dropout(self, rate):
        layer = DropoutLayer(rate)
        self.layers.append(layer)
        
    def add_dense(self, units, activation='linear'):
        layer = DenseLayer(units, activation)
        self.layers.append(layer)
        
    def set_preprocessor(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor
        
    def set_label_mapping(self, label_mapping: Dict[str, int]) -> None:
        self.label_mapping = label_mapping
        
    def load_weights_from_keras(self, keras_model):
        keras_layers = keras_model.layers
        layer_idx = 0
        
        print(f"Custom model has {len(self.layers)} layers")
        print(f"Keras model has {len(keras_layers)} layers")
        
        for i, our_layer in enumerate(self.layers):
            print(f"Processing layer {i}: {type(our_layer).__name__}")
            
            while layer_idx < len(keras_layers) and len(keras_layers[layer_idx].get_weights()) == 0:
                print(f"Skipping Keras layer {layer_idx} as it has no weights")
                layer_idx += 1
                
            if layer_idx >= len(keras_layers):
                print("Ran out of Keras layers with weights")
                break
                
            print(f"Using Keras layer {layer_idx} for weights")
            
            if isinstance(our_layer, EmbeddingLayer):
                weights = keras_layers[layer_idx].get_weights()
                print(f"Found {len(weights)} weight arrays")
                if len(weights) >= 1:
                    our_layer.load_weights(weights[0])
                layer_idx += 1
                
            elif isinstance(our_layer, LSTMLayer):
                weights = keras_layers[layer_idx].get_weights()
                print(f"Found {len(weights)} weight arrays for LSTM")
                if len(weights) > 0:
                    our_layer.load_weights(weights)
                layer_idx += 1
                
            elif isinstance(our_layer, BidirectionalLSTM):
                keras_layer = keras_layers[layer_idx]
                print(f"Loading bidirectional LSTM weights from layer type: {type(keras_layer).__name__}")
                
                if isinstance(keras_layer, tf.keras.layers.Bidirectional):
                    all_weights = keras_layer.get_weights()
                    print(f"Bidirectional layer weights: {len(all_weights)} arrays")
                    
                    # In TensorFlow 2.x, bidirectional weights are typically split as
                    # [forward_kernel, forward_recurrent_kernel, forward_bias, 
                    #  backward_kernel, backward_recurrent_kernel, backward_bias]
                    num_weights_per_lstm = len(all_weights) // 2
                    
                    forward_weights = all_weights[:num_weights_per_lstm]
                    backward_weights = all_weights[num_weights_per_lstm:]
                    
                    print(f"Split into forward ({len(forward_weights)} arrays) and backward ({len(backward_weights)} arrays)")
                    print(f"Forward shapes: {[w.shape for w in forward_weights]}")
                    print(f"Backward shapes: {[w.shape for w in backward_weights]}")
                    
                    our_layer.load_weights(forward_weights, backward_weights)
                    print("Bidirectional weights loaded successfully")
                else:
                    print(f"WARNING: Expected Bidirectional layer but got {type(keras_layer).__name__}")
                    # Fallback for unexpected layer type
                    
                layer_idx += 1
                
            elif isinstance(our_layer, DenseLayer):
                weights = keras_layers[layer_idx].get_weights()
                print(f"Found {len(weights)} weight arrays for Dense")
                if len(weights) == 2:
                    our_layer.load_weights(weights[0], weights[1])
                else:
                    print(f"Warning: Expected 2 weight arrays for Dense layer but got {len(weights)}")
                layer_idx += 1
                
            else:
                print(f"Skipping layer type: {type(our_layer).__name__}")
    
    def save(self, model_path: str, save_preprocessor: bool = True) -> None:

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        config = {
            'layer_types': [layer.__class__.__name__ for layer in self.layers],
            'layer_configs': []
        }
        
        for layer in self.layers:
            if isinstance(layer, EmbeddingLayer):
                layer_config = {
                    'vocab_size': layer.vocab_size,
                    'embedding_dim': layer.embedding_dim,
                    'input_length': layer.input_length
                }
            elif isinstance(layer, LSTMLayer):
                layer_config = {
                    'units': layer.units,
                    'return_sequences': layer.return_sequences,
                    'go_backwards': layer.go_backwards
                }
            elif isinstance(layer, BidirectionalLSTM):
                layer_config = {
                    'units': layer.units,
                    'return_sequences': layer.return_sequences
                }
            elif isinstance(layer, DropoutLayer):
                layer_config = {
                    'rate': layer.rate
                }
            elif isinstance(layer, DenseLayer):
                layer_config = {
                    'units': layer.units,
                    'activation': layer.activation
                }
            else:
                layer_config = {}
                
            config['layer_configs'].append(layer_config)
            
        if self.label_mapping:
            config['label_mapping'] = self.label_mapping
            
        np.save(f"{model_path}.config", config, allow_pickle=True)
        
        if save_preprocessor and self.preprocessor:
            self.preprocessor.save(f"{model_path}.preprocessor")
            
        
    @classmethod
    def load(cls, model_path: str, load_preprocessor: bool = True) -> 'LSTMModel':
       
        config = np.load(f"{model_path}.config", allow_pickle=True).item()
        
        preprocessor = None
        if load_preprocessor and os.path.exists(f"{model_path}.preprocessor"):
            preprocessor = TextPreprocessor.load(f"{model_path}.preprocessor")
            
        model = cls(preprocessor=preprocessor)
        
        if 'label_mapping' in config:
            model.label_mapping = config['label_mapping']
            
        for layer_type, layer_config in zip(config['layer_types'], config['layer_configs']):
            if layer_type == 'EmbeddingLayer':
                model.add_embedding(
                    layer_config['vocab_size'],
                    layer_config['embedding_dim'],
                    layer_config['input_length']
                )
            elif layer_type == 'LSTMLayer':
                model.add_lstm(
                    layer_config['units'],
                    layer_config['return_sequences'],
                    bidirectional=False
                )
            elif layer_type == 'BidirectionalLSTM':
                model.add_lstm(
                    layer_config['units'],
                    layer_config['return_sequences'],
                    bidirectional=True
                )
            elif layer_type == 'DropoutLayer':
                model.add_dropout(layer_config['rate'])
            elif layer_type == 'DenseLayer':
                model.add_dense(
                    layer_config['units'],
                    layer_config['activation']
                )
                
        
        return model
                
    def predict(self, inputs):
        
        if self.preprocessor and isinstance(inputs, (str, list)) and (isinstance(inputs, str) or 
                                                                      isinstance(inputs[0], str)):
            inputs = self.preprocessor.preprocess(inputs)
        
        if hasattr(inputs, 'numpy'):
            inputs = inputs.numpy()
        
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
            
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
        
    def predict_classes(self, inputs):
      
        predictions = self.predict(inputs)
        return np.argmax(predictions, axis=-1)
        
    def predict_labels(self, inputs):
       
        class_indices = self.predict_classes(inputs)
        
        if self.label_mapping:
            reverse_mapping = {v: k for k, v in self.label_mapping.items()}
            return [reverse_mapping[idx] for idx in class_indices]
        
        return class_indices
    
    def count_params(self) -> int:
       
        total_params = 0
        
        for layer in self.layers:
            if isinstance(layer, EmbeddingLayer):
                total_params += layer.vocab_size * layer.embedding_dim
            elif isinstance(layer, LSTMLayer):
                input_dim = 0
                for prev_layer in self.layers:
                    if isinstance(prev_layer, EmbeddingLayer):
                        input_dim = prev_layer.embedding_dim
                        break
                    elif isinstance(prev_layer, LSTMLayer) or isinstance(prev_layer, BidirectionalLSTM):
                        if isinstance(prev_layer, BidirectionalLSTM):
                            input_dim = prev_layer.units * 2
                        else:
                            input_dim = prev_layer.units
                        break
                
                if input_dim > 0:
                    total_params += 4 * ((input_dim + layer.units + 1) * layer.units)
            elif isinstance(layer, BidirectionalLSTM):
                input_dim = 0
                for prev_layer in self.layers:
                    if isinstance(prev_layer, EmbeddingLayer):
                        input_dim = prev_layer.embedding_dim
                        break
                    elif isinstance(prev_layer, LSTMLayer) or isinstance(prev_layer, BidirectionalLSTM):
                        if isinstance(prev_layer, BidirectionalLSTM):
                            input_dim = prev_layer.units * 2
                        else:
                            input_dim = prev_layer.units
                        break
                
                if input_dim > 0:
                    total_params += 2 * 4 * ((input_dim + layer.units + 1) * layer.units)

            elif isinstance(layer, DenseLayer):
                input_dim = 0
                for prev_layer in self.layers:
                    if isinstance(prev_layer, LSTMLayer):
                        input_dim = prev_layer.units
                        break
                    elif isinstance(prev_layer, BidirectionalLSTM):
                        input_dim = prev_layer.units * 2
                        break
                    elif isinstance(prev_layer, DenseLayer):
                        input_dim = prev_layer.units
                        break
                
                if input_dim > 0:
                    total_params += (input_dim + 1) * layer.units
        
        return total_params
