# TinyBrain - A Signal Transmission Neural Network

## Overview
This project implements a simple **feedforward neural network** that demonstrates the **signal transmission** process from an **input vector** to an **output vector** using **NumPy** and **SciPy**.  

The project now includes full support for:
- feedforward computation,
- intermediate activation visualization,
- unit testing,
- CI integration,
- Docker containerization.

---

## Project Structure

signal_transmission_nn/
├── src/
│ └── signal_transmission.py # Core math and feedforward functions
│
├── scripts/
│ ├── run_signal_transmission.py # Runs a demo of the network
│ └── plot_activations.py # Visualizes layer activations
│
├── tests/
│ └── test_signal_transmission.py # Unit tests for correctness
│
├── data/ # (optional, not used in this exercise)
│
├── .github/workflows/ci.yml # Continuous Integration configuration
├── Dockerfile # Container build instructions
├── Makefile # Simple command automation
├── requirements.txt # Python dependencies
├── .gitignore # Ignore cache & local files
├── .dockerignore # Reduce Docker image size
└── README.md # This file

---

## Installation and Setup

### Local Installation

```bash
make install

---

## This installs the dependencies listed in requirements.txt:

numpy
scipy
matplotlib
pytest

---

# Running the Simulation

## To execute the signal transmission and see the resulting output vector:
make run

## Expected Output (approximate):
Running neural signal transmission example...
Network output vector:
[0.721 0.737 0.840]

This shows the final activations of the 3 output neurons after the sigmoid transformation.

---

# Visualize the Layer Activations

## To generate a bar chart showing the signal flow across layers (input → hidden → output):
make plot

This runs scripts/plot_activations.py and produces an image file:
activations.png

---

## The plot contains 3 bars per layer:

 * Input Layer: raw input signals
 
 * Hidden Layer: post-sigmoid activations
 
 * Output Layer: final network output activations
 
---

# How the Code Works

## Feedforward Calculation
The core of the network is implemented in src/signal_transmission.py:
hidden_input = np.dot(W_input_hidden, I)
hidden_output = sigmoid(hidden_input)
output_input = np.dot(W_hidden_output, hidden_output)
output_output = sigmoid(output_input)

---

# ASCII Data Flow Diagram
Input Vector (I)
      │
      ▼
W_input_hidden  →  Hidden Layer (sigmoid)
      │
      ▼
W_hidden_output →  Output Layer (sigmoid)

 * forward_pass() returns only the final output.
 
 * forward_pass_with_intermediates() returns all intermediate vectors for visualization.
 
---

# Testing

## Run all tests:
make test

## You’ll see output like:
============================== test session starts ==============================
collected 1 item

tests/test_signal_transmission.py::test_forward_pass_shape PASSED

---

# Docker Support
## You can build and run the project in a fully containerized environment.

make docker-build

## Run the container:
make docker-run

---

# Continuous Integration (CI)

## This project includes GitHub Actions for automatic testing and validation on every commit or pull request.

### The CI workflow (.github/workflows/ci.yml) performs:

 1. Python setup
 2. Dependency installation
 3. Script execution (run_signal_transmission.py)
 4. Plot generation (plot_activations.py)
 5. Unit testing (pytest)
 
All plots are generated headless using MPLBACKEND=Agg.

---

# Summary
| Feature              | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| **Language**         | Python 3.11                                                        |
| **Libraries**        | NumPy, SciPy, Matplotlib                                           |
| **Concepts**         | Feedforward propagation, sigmoid activation, matrix multiplication |
| **Visualization**    | Bar plots of input, hidden, and output activations                 |
| **Testing**          | Pytest unit tests                                                  |
| **Automation**       | Makefile and GitHub Actions CI                                     |
| **Containerization** | Dockerfile for portable execution                                  |

---

# Learning Objective
## You learn to:

 * represent and multiply matrices in NumPy,
 * use SciPy’s sigmoid function for activation,
 * visualize network activations,
 * and automate reproducible analysis workflows.
 
---

# Author: submarine
TinyBrain - A Signal Transmission Neural Network

---
