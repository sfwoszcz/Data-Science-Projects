# scripts/run_signal_transmission.py

from src.signal_transmission import example_run

if __name__ == "__main__":
    print("Running neural signal transmission example...")
    output = example_run()
    print("Network output vector:")
    print(output)
