import numpy as np

class DropoutLayer:
   
    def __init__(self, rate=0.5):
        self.rate = rate
        
    def forward(self, inputs, training=False):
        return inputs * (1 - self.rate) if training else inputs
