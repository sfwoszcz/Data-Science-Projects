# src/signal_transmission.py

"""
==============================================================
Module: signal_transmission.py
Topic: Feedforward signal transmission with NumPy + sigmoid
==============================================================

This module:
- defines a forward pass through a tiny neural network
- lets you get ONLY the final output (forward_pass)
- or get ALL intermediate values (forward_pass_with_intermediates)
so that other scripts (like the plotting script) can visualize
the activations.
==============================================================
"""

import numpy as np
from scipy.special import expit as sigmoid  # sigmoid activation


def forward_pass(
    I: np.ndarray,
    W_input_hidden: np.ndarray,
    W_hidden_output: np.ndarray
) -> np.ndarray:
    """
    Perform a simple feedforward pass and return ONLY the final output.

    Parameters
    ----------
    I : np.ndarray
        Input vector, shape (3,) in this exercise.
    W_input_hidden : np.ndarray
        Weight matrix from input to hidden layer, shape (3,3).
    W_hidden_output : np.ndarray
        Weight matrix from hidden to output layer, shape (3,3).

    Returns
    -------
    np.ndarray
        Output vector after sigmoid activation, shape (3,).
    """
    # input -> hidden
    hidden_input = np.dot(W_input_hidden, I)          # shape (3,)
    hidden_output = sigmoid(hidden_input)             # shape (3,)

    # hidden -> output
    output_input = np.dot(W_hidden_output, hidden_output)  # shape (3,)
    output_output = sigmoid(output_input)                  # shape (3,)

    return output_output


def forward_pass_with_intermediates(
    I: np.ndarray,
    W_input_hidden: np.ndarray,
    W_hidden_output: np.ndarray
):
    """
    Same as forward_pass, but return ALL intermediate values
    so that we can plot them.

    Returns
    -------
    I : np.ndarray
        Original input vector.
    hidden_input : np.ndarray
        Net input to hidden layer (before activation).
    hidden_output : np.ndarray
        Activated hidden layer output.
    output_input : np.ndarray
        Net input to output layer (before activation).
    output_output : np.ndarray
        Final activated network output.
    """
    # input -> hidden
    hidden_input = np.dot(W_input_hidden, I)
    hidden_output = sigmoid(hidden_input)

    # hidden -> output
    output_input = np.dot(W_hidden_output, hidden_output)
    output_output = sigmoid(output_input)

    return I, hidden_input, hidden_output, output_input, output_output


def example_run() -> np.ndarray:
    """
    Run the example from the exercise with the given matrices.
    This is useful for scripts/run_signal_transmission.py.
    """
    # Given in the exercise
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

    output_vector = forward_pass(I, W_input_hidden, W_hidden_output)
    return output_vector


if __name__ == "__main__":
    # manual test
    out = example_run()
    print("Output vector:")
    print(out)
