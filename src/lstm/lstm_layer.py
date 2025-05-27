import numpy as np

class LSTMLayer:
    """
    LSTM layer implementation from scratch
    """
    
    def __init__(self, units, return_sequences=False, go_backwards=False):
        self.units = units
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        
        # Weight matrices (will be loaded from Keras model)
        self.W_i = None  # Input gate weights
        self.W_f = None  # Forget gate weights  
        self.W_c = None  # Cell state weights
        self.W_o = None  # Output gate weights
        
        # Recurrent weights
        self.U_i = None
        self.U_f = None
        self.U_c = None
        self.U_o = None
        
        # Biases
        self.b_i = None
        self.b_f = None
        self.b_c = None
        self.b_o = None
        
    def load_weights(self, weights):
        """Load weights from trained Keras model"""
        # Keras LSTM weights format: [W, U, b]
        # W contains [W_i, W_f, W_c, W_o] concatenated
        # U contains [U_i, U_f, U_c, U_o] concatenated
        # b contains [b_i, b_f, b_c, b_o] concatenated
        
        W, U, b = weights
        
        # Split concatenated weight matrices
        self.W_i = W[:, :self.units]
        self.W_f = W[:, self.units:2*self.units]
        self.W_c = W[:, 2*self.units:3*self.units]
        self.W_o = W[:, 3*self.units:]
        
        self.U_i = U[:, :self.units]
        self.U_f = U[:, self.units:2*self.units]
        self.U_c = U[:, 2*self.units:3*self.units]
        self.U_o = U[:, 3*self.units:]
        
        self.b_i = b[:self.units]
        self.b_f = b[self.units:2*self.units]
        self.b_c = b[2*self.units:3*self.units]
        self.b_o = b[3*self.units:]
        
    def sigmoid(self, x):
        """Sigmoid activation function optimized for numerical stability"""
        # Using clip for numerical stability and vectorized operations
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        """Tanh activation function optimized for numerical stability"""
        # Using clip for numerical stability and vectorized operations
        return np.tanh(np.clip(x, -250, 250))
        
    def forward(self, inputs):
        """
        Forward pass through LSTM layer
        
        Args:
            inputs: Input sequences of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            If return_sequences=True: (batch_size, sequence_length, units)
            If return_sequences=False: (batch_size, units)
        """
        if self.W_i is None:
            raise ValueError("Weights not loaded. Call load_weights() first.")
            
        batch_size, seq_length, input_dim = inputs.shape
        
        # Initialize hidden state and cell state
        h = np.zeros((batch_size, self.units))
        c = np.zeros((batch_size, self.units))
        
        # Pre-allocate memory for outputs if return_sequences is True
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_length, self.units))
        
        # Determine sequence order
        time_steps = range(seq_length)
        if self.go_backwards:
            time_steps = reversed(time_steps)
            
        for t in time_steps:
            x_t = inputs[:, t, :]
            
            # Compute all gate inputs at once for better vectorization
            i_gate = np.dot(x_t, self.W_i) + np.dot(h, self.U_i) + self.b_i
            f_gate = np.dot(x_t, self.W_f) + np.dot(h, self.U_f) + self.b_f
            c_gate = np.dot(x_t, self.W_c) + np.dot(h, self.U_c) + self.b_c
            o_gate = np.dot(x_t, self.W_o) + np.dot(h, self.U_o) + self.b_o
            
            # Apply activation functions
            i_t = self.sigmoid(i_gate)
            f_t = self.sigmoid(f_gate)
            c_tilde = self.tanh(c_gate)
            o_t = self.sigmoid(o_gate)
            
            # Update cell state
            c = f_t * c + i_t * c_tilde
            
            # Update hidden state
            h = o_t * self.tanh(c)
            
            if self.return_sequences:
                if self.go_backwards:
                    # If going backwards, we need to fill the array from the end
                    outputs[:, seq_length - 1 - t, :] = h
                else:
                    outputs[:, t, :] = h
                
        if self.return_sequences:
            return outputs
        else:
            return h

class BidirectionalLSTM:
    """
    Bidirectional LSTM implementation
    """
    
    def __init__(self, units, return_sequences=False):
        self.units = units
        self.return_sequences = return_sequences
        
        self.forward_lstm = LSTMLayer(units, return_sequences=True, go_backwards=False)
        self.backward_lstm = LSTMLayer(units, return_sequences=True, go_backwards=True)
        
    def load_weights(self, forward_weights, backward_weights):
        """Load weights for both forward and backward LSTM"""
        self.forward_lstm.load_weights(forward_weights)
        self.backward_lstm.load_weights(backward_weights)
        
    def forward(self, inputs):
        """
        Forward pass through bidirectional LSTM
        
        Args:
            inputs: Input sequences of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Concatenated forward and backward outputs
        """
        forward_out = self.forward_lstm.forward(inputs)
        backward_out = self.backward_lstm.forward(inputs)
        
        # Concatenate along feature dimension
        output = np.concatenate([forward_out, backward_out], axis=-1)
        
        if not self.return_sequences:
            # Return last timestep
            output = output[:, -1, :]
            
        return output
