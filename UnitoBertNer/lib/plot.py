import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes = ['Class 0', 'Class 1'], figsize=(8, 6), saveName=None):
    """ """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if saveName: plt.savefig("./Plot/" + saveName)
    else: 
        path = "./Plot/CM_matrix_" + str(datetime.now()).split(".")[0].replace(" ", "_") + ".png"
        path = path.replace(":", "_")
        plt.savefig(path)
    plt.show()


def plot_metrics(train_loss, val_loss, val_accuracy, val_f1, figsize=(10, 5), saveName=None):
    """ """

    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=figsize)

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, 'g', label='Validation Accuracy')
    plt.plot(epochs, val_f1, 'b', label='Validation F1Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout(w_pad=4)
    if saveName: plt.savefig("./Plot/" + saveName)
    else: plt.savefig("./Plot/plot_" + str(datetime.now()).split(".")[0].replace(" ", "_") + ".png")
    plt.show()
