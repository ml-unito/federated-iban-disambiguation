import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score


class EarlyStopping:
  """ Implement a simple early stopping criterion """
  

  def __init__(self, patience=5, delta=0.001):
      self.patience = patience
      self.delta = delta
      self.best_metric = np.inf
      self.counter = 0
      self.early_stop = False

  def __call__(self, current_metric):
      if current_metric < self.best_metric - self.delta:
          self.best_metric = current_metric
          self.counter = 0
      else:
          self.counter += 1

      if self.counter >= self.patience:
          self.early_stop = True

      return self.early_stop


  def reset(self):
      self.counter = 0
      self.best_metric = np.Inf
      self.early_stop = False



class SaveBestModel():
  """ Save best model """


  def __init__(self, savePath, delta):

    self.delta = delta
    self.savePath = savePath


  def __call__(self, model, epoch, train_loss, valid_loss):

    if (valid_loss < (train_loss + self.delta)):
      torch.save(model.state_dict(), self.savePath)
      print("Best model found, saved on:", self.savePath)
      print("Best epoch: ", epoch)

    elif (epoch == 1):
      torch.save(model.state_dict(), self.savePath)
      print("Model epoch1 saved on:", self.savePath)



def compute_metrics(predictions, labels):
    """ """
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return {"accuracy": round(accuracy,3), "precision": round(precision,3), "recall": round(recall,3), "f1": round(f1,3)}
