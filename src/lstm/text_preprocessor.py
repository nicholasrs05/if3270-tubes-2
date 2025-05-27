import numpy as np
import tensorflow as tf
from typing import List, Union, Tuple

class TextPreprocessor:
   
    def __init__(
        self,
        max_tokens: int = 10000,
        output_sequence_length: int = 100,
        standardize: str = 'lower_and_strip_punctuation'
    ):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.standardize = standardize
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_sequence_length=output_sequence_length,
            output_mode='int',
            standardize=standardize
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]) -> None:
        self.vectorizer.adapt(texts)
        self.is_fitted = True
        
    def preprocess(self, texts: Union[str, List[str]]) -> np.ndarray:
       
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before preprocessing. Call fit() first.")
            
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            
        # Vectorize texts
        sequences = self.vectorizer(texts)
        
        # Convert to numpy array if it's a TensorFlow tensor
        if hasattr(sequences, 'numpy'):
            sequences = sequences.numpy()
            
        return sequences
    
    def get_vocabulary(self) -> List[str]:
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before getting vocabulary.")
        return self.vectorizer.get_vocabulary()
    
    def get_vocab_size(self) -> int:
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before getting vocabulary size.")
        return len(self.get_vocabulary())
        
    def save(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("TextPreprocessor must be fitted before saving.")
            
        config = {
            'max_tokens': self.max_tokens,
            'output_sequence_length': self.output_sequence_length,
            'standardize': self.standardize,
            'vocabulary': self.get_vocabulary()
        }
        np.save(filepath, config, allow_pickle=True)
        
    @classmethod
    def load(cls, filepath: str) -> 'TextPreprocessor':
        config = np.load(filepath, allow_pickle=True).item()
        
        preprocessor = cls(
            max_tokens=config['max_tokens'],
            output_sequence_length=config['output_sequence_length'],
            standardize=config['standardize']
        )
        
        # Manually set the vocabulary
        vocab = config['vocabulary']
        preprocessor.vectorizer.set_vocabulary(vocab)
        preprocessor.is_fitted = True
        
        return preprocessor
