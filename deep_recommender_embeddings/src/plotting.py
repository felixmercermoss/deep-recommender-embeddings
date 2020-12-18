import numpy as np
from matplotlib import pyplot as plt


def plot_metric(history, metric, validation_freq=1):
    """
    PLot metrics of tfrs model
    Args:
        history: returned by Model.tfrs().fit()
        metric:
        validation_freq:

    Returns:

    """
    num_validation_runs = len(history.history[f"val_{metric}"])
    num_train_runs = len(history.history[metric])
    epochs = [(x + 1)* validation_freq for x in range(num_validation_runs)]
    train_epochs = np.arange(1, num_train_runs+1)

    plt.plot(epochs, history.history[f"val_{metric}"], label="validation")
    plt.plot(train_epochs, history.history[metric], label="train")
    plt.title(f"{metric} vs epoch")
    plt.xlabel("epoch")
    plt.ylabel(metric);
    plt.legend()