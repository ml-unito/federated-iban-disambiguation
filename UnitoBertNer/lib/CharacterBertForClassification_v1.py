import os
import sys
import tqdm
import torch
import numpy as np
import torch.nn as nn
sys.path.insert(0, os.path.abspath('./character_bert_model'))
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from utils.character_cnn import CharacterIndexer
from modeling.character_bert import CharacterBertModel
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CharacterBertForClassification(nn.Module):
    def __init__(self, num_labels=1):
        """ Add classification layer with a sigmoid activation function on the last level"""
        super(CharacterBertForClassification, self).__init__()
        self.character_bert = CharacterBertModel.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, num_labels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids):
        """Forward pass"""
        outputs = self.character_bert(input_ids)[0]       # Use the last hidden states
        pooled_output = self.dropout(outputs[:, 0, :])    # Take the first token's embedding ([CLS])
        logits = self.classifier(pooled_output)
        x = self.sigmoid(logits)
        return x




def compute_metrics(predictions, labels):
    """ """
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    return {"accuracy": round(accuracy,3), "precision": round(precision,3), "recall": round(recall,3), "f1": round(f1,3)}



def train(model, optimizer, train_loader, criterion, scheduler):
    model.train()
    total_loss = 0
    total_acc = 0
    input_ids = None
    labels = None
    outputs = None
    loss = None

    for batch in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()                               # Clear gradients
        outputs = model(input_ids)                          # Forward pass
        loss = criterion(outputs, labels.float())           # Compute loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss = total_loss/len(train_loader)
    return loss


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0
    input_ids = None
    labels = None
    outputs = None
    loss = None
    predictions = None
    val_correct = 0
    elem_list = None
    predictions = []
    total_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)

            # Forward pass
            outputs = model(input_ids)

            # Compute loss
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            # get eval metrics
            elem_list = outputs.round().tolist()
            predictions += [int(el[0]) for el in elem_list]
            elem_list = labels.tolist()
            total_labels += [el[0] for el in elem_list]

    loss = total_loss/len(test_loader)
    metrics = compute_metrics(predictions, total_labels)
    return loss, metrics, predictions, total_labels





class EarlyStopping:
  """ Implement a simple early stopping criterion """

  def __init__(self, patience=5, delta=0.001, prevent_overfit = True, mode='valid'):
      self.patience = patience
      self.delta = delta
      self.mode = mode
      self.best_metric = np.Inf if mode == 'valid' else -np.Inf
      self.counter = 0
      self.prevent_overfit = prevent_overfit
      self.early_stop = False

  def __call__(self, current_metric):
      if self.prevent_overfit:
        if self.mode == 'valid':
            # if loss tends to decrease then good
            if current_metric < self.best_metric - self.delta:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'train':
            # if loss tends to increase then bad
            if current_metric < self.best_metric - self.delta:
                self.counter += 1
            else:
                self.best_metric = current_metric
                self.counter = 0

      if self.counter >= self.patience:
          self.early_stop = True

      return self.early_stop




class SaveBestModel():
  """ Save best model """

  def __init__(self, savePath, delta):
    """ """
    self.delta = delta
    self.savePath = savePath


  def __call__(self, model, epoch, train_loss, valid_loss):
    """ """
    if (valid_loss < (train_loss + self.delta)):
      torch.save(model.state_dict(), self.savePath)
      print("Best model found, saved on:", self.savePath)
      print("Best epoch: ", epoch)

    elif (epoch == 1):
      torch.save(model.state_dict(), self.savePath)
      print("Model epoch1 saved on:", self.savePath)
