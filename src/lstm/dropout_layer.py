import numpy as np

class DropoutLayer:
    """
    Dropout layer implementation (inference mode only)
    """
    
    def __init__(self, rate=0.5):
        self.rate = rate
        
    def forward(self, inputs, training=False):
        """
        Forward pass through dropout layer
        
        Args:
            inputs: Input tensor
            training: Whether in training mode (always False for inference)
            
        Returns:
            Input tensor (unchanged during inference)
        """
        # During inference, dropout is not applied
        # We multiply by (1-rate) to maintain the expected value
        return inputs * (1 - self.rate) if training else inputs
