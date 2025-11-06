# scripts/plot_activations.py

import numpy as np
import matplotlib.pyplot as plt

from src.signal_transmission import forward_pass_with_intermediates


def main():
    # same example data as before
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

    (inp,
     hidden_z,
     hidden_a,
     output_z,
     output_a) = forward_pass_with_intermediates(I,
                                                 W_input_hidden,
                                                 W_hidden_output)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1) input vector
    axes[0].bar([1, 2, 3], inp)
    axes[0].set_title("Input layer")
    axes[0].set_xticks([1, 2, 3])
    axes[0].set_ylim(0, 1.2)

    # 2) hidden activations (after sigmoid)
    axes[1].bar([1, 2, 3], hidden_a)
    axes[1].set_title("Hidden layer (activated)")
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_ylim(0, 1.2)

    # 3) output activations
    axes[2].bar([1, 2, 3], output_a)
    axes[2].set_title("Output layer (activated)")
    axes[2].set_xticks([1, 2, 3])
    axes[2].set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig("activations.png")
    plt.show()


if __name__ == "__main__":
    main()