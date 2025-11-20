# src/plot_performance.py

import matplotlib.pyplot as plt

def plot_performance(performances):
    """
    Plots performance vs activation functions.
    """
    activation_names = list(performances.keys())
    performance_values = list(performances.values())

    plt.figure(figsize=(10, 6))
    plt.bar(activation_names, performance_values, color='skyblue')
    plt.xlabel('Activation Functions')
    plt.ylabel('Performance (Accuracy)')
    plt.title('Performance Comparison of Different Activation Functions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("performance_comparison.png")  # Save the plot
    plt.show()

if __name__ == "__main__":
    performances = {
        "Sigmoid": 0.80,      # Example values, replace with actual
        "LeakyReLU_0.01": 0.82,
        "LeakyReLU_0.05": 0.85,
        "LeakyReLU_0.1": 0.86,
        "LeakyReLU_0.5": 0.83,
        "PReLU": 0.87,
        "ELU_0.1": 0.89,
        "ELU_0.2": 0.88,
        "ELU_0.3": 0.90,
    }

    plot_performance(performances)
