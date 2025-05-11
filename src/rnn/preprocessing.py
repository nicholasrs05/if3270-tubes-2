import tensorflow as tf
import numpy as np
from typing import List

class TextPreprocessor:
    def __init__(
        self,
        max_tokens: int = 10000,
        output_sequence_length: int = 200,
        embedding_dim: int = 100
    ):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.embedding_dim = embedding_dim
        self.vectorization_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            standardize='lower_and_strip_punctuation'
        )
        
    def fit(self, texts: List[str]) -> None:
        """Fit the vectorization layer on the training texts."""
        self.vectorization_layer.adapt(texts)
        
    def preprocess(self, texts: List[str]) -> np.ndarray:
        """Convert texts to sequences of integers."""
        return self.vectorization_layer(texts).numpy()
    
    def get_vocabulary(self) -> List[str]:
        """Get the vocabulary learned during fitting."""
        return self.vectorization_layer.get_vocabulary()
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.get_vocabulary()) 