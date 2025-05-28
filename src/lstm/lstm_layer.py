import numpy as np

class LSTMLayer:
    
    def __init__(self, units, return_sequences=False, go_backwards=False):
        self.units = units
        self.return_sequences = return_sequences
        self.go_backwards = go_backwards
        
        self.W_i = None  
        self.W_f = None   
        self.W_c = None  
        self.W_o = None  
        
        self.U_i = None
        self.U_f = None
        self.U_c = None
        self.U_o = None
        
        self.b_i = None
        self.b_f = None
        self.b_c = None
        self.b_o = None
        
        self.weights_loaded = False
        
    def load_weights(self, weights):
        if len(weights) != 3:
            raise ValueError(f"Expected 3 weight arrays but got {len(weights)}")
            
        W, U, b = weights
        
        print(f"LSTM weights shapes: W={W.shape}, U={U.shape}, b={b.shape}")
        
        actual_units = U.shape[0] if U.shape[0] <= U.shape[1] else U.shape[1]
        
        if actual_units != self.units:
            print(f"WARNING: Using detected unit size {actual_units} instead of configured {self.units}")
            self.units = actual_units  
        
        input_slice = slice(0, self.units)
        forget_slice = slice(self.units, 2*self.units)
        cell_slice = slice(2*self.units, 3*self.units)
        output_slice = slice(3*self.units, 4*self.units)
        
        self.W_i = W[:, input_slice]
        self.W_f = W[:, forget_slice]
        self.W_c = W[:, cell_slice]
        self.W_o = W[:, output_slice]
        
        if U.shape[0] == self.units and U.shape[1] == 4*self.units:
            print("Format: Keras standard (units, 4*units)")
            self.U_i = U[:, :self.units].T
            self.U_f = U[:, self.units:2*self.units].T
            self.U_c = U[:, 2*self.units:3*self.units].T
            self.U_o = U[:, 3*self.units:].T
            
        elif U.shape[0] == 4*self.units and U.shape[1] == self.units:
            print("Format: (4*units, units)")
            self.U_i = U[:self.units, :]
            self.U_f = U[self.units:2*self.units, :]
            self.U_c = U[2*self.units:3*self.units, :]
            self.U_o = U[3*self.units:, :]
            
        elif U.shape[1] == 4*self.units:
            print(f"Format: Adapting ({U.shape[0]}, 4*{self.units})")
            self.U_i = U[:, :self.units].T
            self.U_f = U[:, self.units:2*self.units].T
            self.U_c = U[:, 2*self.units:3*self.units].T
            self.U_o = U[:, 3*self.units:].T
            
        elif U.shape[0] == 4*self.units:
            print(f"Format: Adapting (4*{self.units}, {U.shape[1]})")
            self.U_i = U[:self.units, :]
            self.U_f = U[self.units:2*self.units, :]
            self.U_c = U[2*self.units:3*self.units, :]
            self.U_o = U[3*self.units:, :]
            
        elif U.shape[0] % 4 == 0:
            gate_size = U.shape[0] // 4
            print(f"Format: Dividing first dimension ({U.shape[0]}) into 4 gates of size {gate_size}")
            self.U_i = U[:gate_size, :]
            self.U_f = U[gate_size:2*gate_size, :]
            self.U_c = U[2*gate_size:3*gate_size, :]
            self.U_o = U[3*gate_size:, :]
            
        elif U.shape[1] % 4 == 0:
            gate_size = U.shape[1] // 4
            print(f"Format: Dividing second dimension ({U.shape[1]}) into 4 gates of size {gate_size}")
            self.U_i = U[:, :gate_size].T
            self.U_f = U[:, gate_size:2*gate_size].T
            self.U_c = U[:, 2*gate_size:3*gate_size].T
            self.U_o = U[:, 3*gate_size:].T
            
        else:
            U_t = U.T
            print(f"Format: Last resort - transposing to {U_t.shape}")
            
            if U_t.shape[0] % 4 == 0:
                gate_size = U_t.shape[0] // 4
                self.U_i = U_t[:gate_size, :].T
                self.U_f = U_t[gate_size:2*gate_size, :].T
                self.U_c = U_t[2*gate_size:3*gate_size, :].T
                self.U_o = U_t[3*gate_size:, :].T
            elif U_t.shape[1] % 4 == 0:
                gate_size = U_t.shape[1] // 4
                self.U_i = U_t[:, :gate_size]
                self.U_f = U_t[:, gate_size:2*gate_size]
                self.U_c = U_t[:, 2*gate_size:3*gate_size]
                self.U_o = U_t[:, 3*gate_size:]
            else:
                raise ValueError(f"Cannot process recurrent kernel with shape {U.shape}")
        
        if len(b) >= 4*self.units:
            self.b_i = b[input_slice]
            self.b_f = b[forget_slice]
            self.b_c = b[cell_slice]
            self.b_o = b[output_slice]
        else:
            bias_unit_size = len(b) // 4
            self.b_i = b[:bias_unit_size]
            self.b_f = b[bias_unit_size:2*bias_unit_size]
            self.b_c = b[2*bias_unit_size:3*bias_unit_size]
            self.b_o = b[3*bias_unit_size:]
        
        self.weights_loaded = True
        
        print(f"W_i shape: {self.W_i.shape}, U_i shape: {self.U_i.shape}, b_i shape: {self.b_i.shape}")
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def tanh(self, x):
        return np.tanh(np.clip(x, -250, 250))
        
    def forward(self, inputs):
        if not self.weights_loaded or self.W_i is None or self.U_i is None:
            raise ValueError("Weights not loaded. Call load_weights() first.")
            
        batch_size, seq_length, input_dim = inputs.shape
        
        h = np.zeros((batch_size, self.units))
        c = np.zeros((batch_size, self.units))
        
        if self.return_sequences:
            outputs = np.zeros((batch_size, seq_length, self.units))
        
        time_steps = range(seq_length)
        if self.go_backwards:
            time_steps = reversed(time_steps)
            
        for t in time_steps:
            x_t = inputs[:, t, :]
            
            i_gate = np.dot(x_t, self.W_i) + np.dot(h, self.U_i) + self.b_i
            f_gate = np.dot(x_t, self.W_f) + np.dot(h, self.U_f) + self.b_f
            c_gate = np.dot(x_t, self.W_c) + np.dot(h, self.U_c) + self.b_c
            o_gate = np.dot(x_t, self.W_o) + np.dot(h, self.U_o) + self.b_o
            
            i_t = self.sigmoid(i_gate)
            f_t = self.sigmoid(f_gate)
            c_tilde = self.tanh(c_gate)
            o_t = self.sigmoid(o_gate)
            
            c = f_t * c + i_t * c_tilde
            
            h = o_t * self.tanh(c)
            
            if self.return_sequences:
                if self.go_backwards:
                    outputs[:, seq_length - 1 - t, :] = h
                else:
                    outputs[:, t, :] = h
                
        if self.return_sequences:
            return outputs
        else:
            return h


class BidirectionalLSTM:
    def __init__(self, units, return_sequences=False):
        self.units = units
        self.return_sequences = return_sequences
        
        self.forward_lstm = LSTMLayer(units, return_sequences=True, go_backwards=False)
        self.backward_lstm = LSTMLayer(units, return_sequences=True, go_backwards=True)
        self.weights_loaded = False
        
    def load_weights(self, forward_weights, backward_weights):
        print(f"Loading bidirectional LSTM weights:")
        print(f"  Forward weights: {len(forward_weights)} arrays")
        print(f"  Forward weights shapes: {[w.shape for w in forward_weights]}")
        print(f"  Backward weights: {len(backward_weights)} arrays")
        print(f"  Backward weights shapes: {[w.shape for w in backward_weights]}")
        
        try:
            self.forward_lstm.load_weights(forward_weights)
            self.backward_lstm.load_weights(backward_weights)
            
            self.units = self.forward_lstm.units
            
            self.weights_loaded = True
            print("Bidirectional weights loaded successfully!")
        except ValueError as e:
            print(f"Error loading weights: {e}")
            raise
        
    def forward(self, inputs):
     
        if not self.weights_loaded:
            raise ValueError("Bidirectional LSTM weights not loaded. Call load_weights() first.")
       
        forward_out = self.forward_lstm.forward(inputs)
        backward_out = self.backward_lstm.forward(inputs)
        
        output = np.concatenate([forward_out, backward_out], axis=-1)
        
        if not self.return_sequences:
            output = output[:, -1, :]
            
        return output
    
    def self_weightloaded(self):
        return self.weights_loaded