from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from character_bert_model.utils.character_cnn import CharacterIndexer
from character_bert_model.modeling.character_bert import CharacterBertModel
from lib.trainingUtilities import compute_metrics
from transformers import BertTokenizer

indexer = CharacterIndexer()
# tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')


def lookup_table(tokenized_texts, dataframe):
    """ define the input tensors for the CharacterBert model """
    
    input_tensors = indexer.as_padded_tensor(tokenized_texts)     # Create input tensor
    labels = torch.tensor(dataframe['label'].values)  
    return input_tensors, labels


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


def train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler):
    """ Train the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    total_loss = 0
    predictions = []
    total_labels = []


    for i in tqdm(range(0, len(X_train), batch_size), desc="Training"):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Convert batch to tensors  
        input_ids = indexer.as_padded_tensor(batch_X)
        labels = torch.tensor(batch_y)
         
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
    loss = total_loss / (len(X_train) // batch_size)
    metrics = compute_metrics(predictions, total_labels)
    return loss, metrics['accuracy']


def test(model, X_test, y_test, batch_size, criterion):
    """ Evaluate the model """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    total_loss = 0
    predictions = []
    total_labels = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X_test), batch_size), desc="Testing"):
        # for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            
            # Convert batch to tensors  
            input_ids = indexer.as_padded_tensor(batch_X)
            labels = torch.tensor(batch_y)
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
            

    loss = total_loss / (len(X_test) // batch_size)
    metrics = compute_metrics(predictions, total_labels)
    return loss, metrics, predictions, total_labels
