import os
import sys
import tqdm
import torch
import numpy as np
import torch.nn as nn
# sys.path.insert(0, os.path.abspath('./character_bert_model'))
from character_bert_model.utils.character_cnn import CharacterIndexer
from character_bert_model.modeling.character_bert import CharacterBertModel
from torchmetrics import Accuracy, F1Score, Precision, Recall


class CharacterBertForClassification(nn.Module):
    def __init__(self, num_labels=2):
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
        # if not self.training:
        #     x = x.round()
        return x


def train(model, optimizer, train_loader, criterion, scheduler):
    """ Train the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    total_loss = 0
    predictions = []
    total_labels = []

    accs, precs, recs, f1s = [], [], [], []

    accuracy = Accuracy(task="multiclass", num_classes=2, top_k=1, average="micro")
    precision = Precision(task="multiclass", num_classes=2, top_k=1, average="micro")
    recall = Recall(task="multiclass", num_classes=2, top_k=1, average="micro")
    f1 = F1Score(task="multiclass", num_classes=2, top_k=1, average="micro")

    for batch in tqdm.tqdm(train_loader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()                               # Clear gradients
        outputs = model(input_ids)                          # Forward pass
        loss = criterion(outputs, labels)           # Compute loss
        total_loss += loss.item()
        
        # Metrics
        accuracy.update(outputs.cpu(), labels.cpu())
        precision.update(outputs.cpu(), labels.cpu())
        recall.update(outputs.cpu(), labels.cpu())
        f1.update(outputs.cpu(), labels.cpu())

        accs.append(accuracy.compute().item())
        precs.append(precision.compute().item())
        recs.append(recall.compute().item())
        f1s.append(f1.compute().item())

        # Backpropagation
        loss.backward()
        optimizer.step()

    scheduler.step()                                        # Step the scheduler after every epoch
    loss = total_loss / len(train_loader)
    
    acc_value = np.round(sum(accs) / len(accs), 5).item()
    prec_value = np.round(sum(precs) / len(precs), 5).item()
    rec_value =  np.round(sum(recs) / len(recs), 5).item()
    f1_value = np.round(sum(f1s) / len(f1s), 5).item()
    metrics = {"accuracy":acc_value, "precision":prec_value, "recall":rec_value, "f1":f1_value}

    return loss, metrics


def test(model, test_loader, criterion):
    """ Evaluate the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
    predictions = []
    total_labels = []

    accs, precs, recs, f1s = [], [], [], []
    accuracy = Accuracy(task="multiclass", num_classes=2, top_k=1, average="micro")
    precision = Precision(task="multiclass", num_classes=2, top_k=1, average="micro")
    recall = Recall(task="multiclass", num_classes=2, top_k=1, average="micro")
    f1 = F1Score(task="multiclass", num_classes=2, top_k=1, average="micro")

    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Metrics
            accuracy.update(outputs.cpu(), labels.cpu())
            precision.update(outputs.cpu(), labels.cpu())
            recall.update(outputs.cpu(), labels.cpu())
            f1.update(outputs.cpu(), labels.cpu())

            accs.append(accuracy.compute().item())
            precs.append(precision.compute().item())
            recs.append(recall.compute().item())
            f1s.append(f1.compute().item())

            predictions += outputs.cpu() 
            total_labels += labels.cpu()

    loss = total_loss/len(test_loader)

    acc_value = np.round(sum(accs) / len(accs), 5).item()
    prec_value = np.round(sum(precs) / len(precs), 5).item()
    rec_value =  np.round(sum(recs) / len(recs), 5).item()
    f1_value = np.round(sum(f1s) / len(f1s), 5).item()
    metrics = {"accuracy":acc_value, "precision":prec_value, "recall":rec_value, "f1":f1_value}

    return loss, metrics, predictions, total_labels
