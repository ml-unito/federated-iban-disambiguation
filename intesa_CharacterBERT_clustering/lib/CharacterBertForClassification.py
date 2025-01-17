import os
import sys
import tqdm
import torch
import numpy as np
import torch.nn as nn
# sys.path.insert(0, os.path.abspath('./character_bert_model'))
from character_bert_model.utils.character_cnn import CharacterIndexer
from character_bert_model.modeling.character_bert import CharacterBertModel
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score
from manageTraining import compute_metrics



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
        if not self.training:
            x = x.round()
        return x


def train(model, optimizer, train_loader, criterion, scheduler):
    """ Train the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    total_loss = 0
    predictions = []
    total_labels = []

    for batch in tqdm.tqdm(train_loader):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()                               # Clear gradients
        outputs = model(input_ids)                          # Forward pass
        loss = criterion(outputs, labels.float())           # Compute loss
        total_loss += loss.item()
        
        # get eval metrics
        elem_list = outputs.round().tolist()
        predictions += [int(el[0]) for el in elem_list]
        elem_list = labels.tolist()
        total_labels += [el[0] for el in elem_list]

        # Backpropagation
        loss.backward()
        optimizer.step()

    scheduler.step()                                        # Step the scheduler after every epoch
    loss = total_loss / len(train_loader)
    metrics = compute_metrics(predictions, total_labels)
    return loss, metrics['accuracy']


def evaluate(model, test_loader, criterion):
    """ Evaluate the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
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
