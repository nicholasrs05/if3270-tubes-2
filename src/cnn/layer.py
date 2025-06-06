import numpy as np
from abc import ABC, abstractmethod

def image_to_col(input, kh, kw, sh, sw, out_h, out_w):
    batch, h, w, c = input.shape
    cols = np.zeros((batch, out_h * out_w, kh * kw * c))
    
    for y in range(out_h):
        for x in range(out_w):
            patch = input[:, y*sh:y*sh+kh, x*sw:x*sw+kw, :]
            cols[:, y * out_w + x, :] = patch.reshape(batch, -1)
    
    return cols

class LayerScratch(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, input):
        pass

class InputScratch(LayerScratch):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, input):
        return input

class Conv2DScratch(LayerScratch):
    def __init__(self, filters, kernel_size, strides=(1, 1), activation=None, padding='valid'):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.weights = None
        self.bias = None

    def set_weights(self, weights):
        w, b = weights

        if w.ndim != 4:
            raise ValueError(f"Expected Conv2D weights to be 4D, got shape {w.shape}")
        if b.ndim != 1:
            raise ValueError(f"Expected bias to be 1D, got shape {b.shape}")
        if w.shape[3] != self.filters:
            raise ValueError(f"Weight filter mismatch: expected {self.filters}, got {w.shape[3]}")
        if b.shape[0] != self.filters:
            raise ValueError(f"Bias shape mismatch: expected ({self.filters},), got {b.shape}")

        self.weights = w
        self.bias = b

    def forward(self, input):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights must be set before calling forward")
        
        batch, h, w, c = input.shape
        kh, kw = self.kernel_size
        sh, sw = self.strides

        if self.padding == 'same':
            pad_h = ((h - 1) * sh + kh - h) // 2
            pad_w = ((w - 1) * sw + kw - w) // 2
            input = np.pad(input, ((0,0), (pad_h, pad_h), (pad_w, pad_w), (0,0)), mode='constant') # ngasih padding

        out_h = (input.shape[1] - kh) // sh + 1 # ini basically V = (W - F + 2P) / S + 1, tpi 2P-nya udh ditambahin waktu cek padding
        out_w = (input.shape[2] - kw) // sw + 1

        cols = image_to_col(input, kh, kw, sh, sw, out_h, out_w) # gambarnya dijadiin kolom spy komputasinya cepat
        w_col = self.weights.reshape(-1, self.filters)

        out = np.matmul(cols, w_col) + self.bias # kernel * input + bias
        out = out.reshape(batch, out_h, out_w, self.filters) # ubah spy V * V * k

        if self.activation == 'relu':
            out = np.maximum(0, out)
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-out))
        elif self.activation == 'softmax':
            exp_output = np.exp(out - np.max(out, axis=-1, keepdims=True))
            out = exp_output / np.sum(exp_output, axis=-1, keepdims=True)

        return out

class MaxPooling2DScratch(LayerScratch):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input for MaxPooling2D, got shape {input.shape}")
        
        batch, h, w, c = input.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        if self.padding == 'same':
            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))
            pad_h = max((out_h - 1) * sh + ph - h, 0)
            pad_w = max((out_w - 1) * sw + pw - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            input = np.pad(input, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            out_h = (h - ph) // sh + 1
            out_w = (w - pw) // sw + 1

        output = np.zeros((batch, out_h, out_w, c)) # nyiapin tempat output

        for y in range(out_h):
            for x in range(out_w):
                region = input[:, y*sh:y*sh+ph, x*sw:x*sw+pw, :] # ini ambil area buat di-pool
                output[:, y, x, :] = np.max(region, axis=(1, 2)) # max dari areanya

        return output


class AvgPooling2DScratch(LayerScratch): # basically sama kayak MaxPooling2D, cuma ganti max jadi mean
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def forward(self, input):
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input for AvgPooling2D, got shape {input.shape}")
        
        batch, h, w, c = input.shape
        ph, pw = self.pool_size
        sh, sw = self.strides

        if self.padding == 'same':
            out_h = int(np.ceil(h / sh))
            out_w = int(np.ceil(w / sw))
            pad_h = max((out_h - 1) * sh + ph - h, 0)
            pad_w = max((out_w - 1) * sw + pw - w, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            input = np.pad(input, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
        else:
            out_h = (h - ph) // sh + 1
            out_w = (w - pw) // sw + 1

        output = np.zeros((batch, out_h, out_w, c))

        for y in range(out_h):
            for x in range(out_w):
                region = input[:, y*sh:y*sh+ph, x*sw:x*sw+pw, :]
                output[:, y, x, :] = np.mean(region, axis=(1, 2))

        return output


class FlattenScratch(LayerScratch):
    def forward(self, input):
        if input.ndim < 2:
            raise ValueError(f"Expected at least 2D input for Flatten, got shape {input.shape}")
        
        return input.reshape((input.shape[0], -1))

class DenseScratch(LayerScratch):
    def __init__(self, neurons, activation=None):
        super().__init__()
        self.neurons = neurons
        self.activation = activation
        self.weights = None
        self.bias = None

    def set_weights(self, weights):
        w, b = weights

        if len(w.shape) != 2:
            raise ValueError(f"Expected weights to be 2D, got shape {w.shape}")
        if w.shape[1] != self.neurons:
            raise ValueError(f"Weights don't match neurons: expected {self.neurons}, got {w.shape[1]}")
        if b.shape != (self.neurons,):
            raise ValueError(f"Bias shape mismatch: expected ({self.neurons},), got {b.shape}")
        
        self.weights = w
        self.bias = b

    def forward(self, input):
        if self.weights.shape[0] != input.shape[1]:
            raise ValueError(f"Weights shape {self.weights.shape[0]} doesn't match input shape {input.shape[1]}")
        
        output = input @ self.weights + self.bias # udh berbentuk vektor, jadi tinggal dot product sama weights + bias
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
        elif self.activation == 'softmax':
            exp_output = np.exp(output - np.max(output, axis=-1, keepdims=True))
            output = exp_output / np.sum(exp_output, axis=-1, keepdims=True)
            
        return output