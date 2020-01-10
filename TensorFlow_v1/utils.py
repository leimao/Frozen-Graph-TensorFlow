import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model_accuracy(label, prediction):

    # Evaluate the trained model
    return np.sum(label == prediction) / len(prediction)


def plot_curve(train_losses,
               train_accuracies,
               valid_accuracies,
               savefig=True,
               showfig=False,
               filename='training_curve.png'):

    x = np.arange(len(train_losses))
    y1 = train_accuracies
    y2 = valid_accuracies
    y3 = train_losses

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(x, y1, color='b', marker='o', label='Training Accuracy')
    ax1.plot(x, y2, color='g', marker='o', label='Validation Accuracy')
    ax2.plot(x, y3, color='r', marker='o', label='Training Loss')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')

    ax1.legend()
    ax2.legend()

    if savefig:
        fig.savefig(filename, format='png', dpi=600, bbox_inches='tight')
    if showfig:
        plt.show()
    plt.close()

    return
