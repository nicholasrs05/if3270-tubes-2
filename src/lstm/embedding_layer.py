import numpy as np

class EmbeddingLayer:
    """
    Embedding layer implementation for converting token sequences to embeddings
    """
    
    def __init__(self, vocab_size, embedding_dim, input_length=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        
        self.weights = None
        
    def load_weights(self, weights):
        self.weights = weights
        
    def forward(self, inputs):
       
        if self.weights is None:
            raise ValueError("Weights not loaded. Call load_weights() first.")
            
        # Handle single sequence input
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Convert inputs to integers if they're not already
        inputs = inputs.astype(np.int32)
        
      
        batch_size, seq_length = inputs.shape
        
        # Clip token IDs to ensure they're within vocabulary range
        clipped_inputs = np.clip(inputs, 0, self.vocab_size - 1)
        
        # Use advanced indexing for faster embedding lookup
        embedded = self.weights[clipped_inputs]
        
        return embedded
