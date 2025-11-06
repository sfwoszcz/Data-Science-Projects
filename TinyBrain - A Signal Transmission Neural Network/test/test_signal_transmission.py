# tests/test_signal_transmission.py

import numpy as np
from src.signal_transmission import forward_pass

def test_forward_pass_shape():
    I = np.array([0.9, 0.1, 0.8])
    W_input_hidden = np.array([
        [0.9, 0.3, 0.4],
        [0.2, 0.8, 0.2],
        [0.1, 0.5, 0.6]
    ])
    W_hidden_output = np.array([
        [0.3, 0.7, 0.5],
        [0.6, 0.5, 0.2],
        [0.8, 0.1, 0.9]
    ])

    output = forward_pass(I, W_input_hidden, W_hidden_output)
    assert output.shape == (3,), "Output should have shape (3,)"
    assert np.all(output >= 0) and np.all(output <= 1), "Output must be in sigmoid range [0, 1]"
