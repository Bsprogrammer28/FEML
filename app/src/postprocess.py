import numpy as np
import matplotlib.pyplot as plt

def save_history(history, path):
    np.save(path, history)

def plot_loss(history):
    plt.figure()
    plt.semilogy(history["iter"], history["loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
