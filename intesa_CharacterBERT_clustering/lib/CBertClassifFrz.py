from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
from character_bert_model.utils.character_cnn import CharacterIndexer
from character_bert_model.modeling.character_bert import CharacterBertModel
from lib.trainingUtilities import compute_metrics
from torchmetrics import Accuracy, F1Score, Precision, Recall

indexer = CharacterIndexer()
# tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')



def lookup_table(tokenized_texts, dataframe):
    """ define the input tensors for the CharacterBert model """
    
    input_tensors = indexer.as_padded_tensor(tokenized_texts)     # Create input tensor
    labels = torch.tensor(dataframe['label'].values)  
    return input_tensors, labels


class CBertClassifFrz(nn.Module):
    def __init__(self, num_labels=2):
        super(CBertClassifFrz, self).__init__()
        self.character_bert = CharacterBertModel.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
        
        # Freeze the CharacterBert model
        for param in self.character_bert.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.2)
        self.hidden = nn.Linear(768, 100)
        self.classifier = nn.Linear(100, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_ids):
        """Forward pass"""
        outputs = self.character_bert(input_ids)[0]       # Use the last hidden states
        pooled_output = self.dropout(outputs[:, 0, :])    # Take the first token's embedding ([CLS])
        
        hidden_logits = self.relu(self.hidden(pooled_output))

        logits = self.classifier(hidden_logits)
        
        x = self.sigmoid(logits)
        
        return x


def train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler):
    """ Train the model """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.train()
    total_loss = 0
    # predictions = []
    # total_labels = []
    accs, precs, recs, f1s = [], [], [], []

    accuracy = Accuracy(task="multiclass", num_classes=2, top_k=1, average="micro")
    precision = Precision(task="multiclass", num_classes=2, top_k=1, average="micro")
    recall = Recall(task="multiclass", num_classes=2, top_k=1, average="micro")
    f1 = F1Score(task="multiclass", num_classes=2, top_k=1, average="micro")
   
    for i in tqdm(range(0, len(X_train), batch_size), desc="Training"):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Convert batch to tensors  
        input_ids = indexer.as_padded_tensor(batch_X)
        labels = torch.tensor(batch_y)
         
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()                               # Clear gradients
        outputs = model(input_ids)                          # Forward pass
        loss = criterion(outputs, labels)           # Compute loss
        total_loss += loss.item()
        
        accuracy.update(outputs.cpu(), labels.cpu())
        precision.update(outputs.cpu(), labels.cpu())
        recall.update(outputs.cpu(), labels.cpu())
        f1.update(outputs.cpu(), labels.cpu())

        accs.append(accuracy.compute().item())
        precs.append(precision.compute().item())
        recs.append(recall.compute().item())
        f1s.append(f1.compute().item())
        # get eval metrics
        # elem_list = outputs.tolist()
        # print(elem_list)
        # predictions += [int(el[0]) for el in elem_list]
        # predictions += outputs
        # print("pred",predictions)
        # elem_list = labels.tolist()
        # total_labels += [el for el in elem_list]
        # total_labels += labels

        # Backpropagation
        loss.backward()
        optimizer.step()

    scheduler.step()
    
    num_batch = (len(X_train) // batch_size) if (len(X_train) // batch_size) != 0 else 1
    loss = total_loss / num_batch

    acc_value = np.round(sum(accs) / len(accs), 5).item()
    prec_value = np.round(sum(precs) / len(precs), 5).item()
    rec_value =  np.round(sum(recs) / len(recs), 5).item()
    f1_value = np.round(sum(f1s) / len(f1s), 5).item()
    metrics = {"accuracy":acc_value, "precision":prec_value, "recall":rec_value, "f1":f1_value}

    return loss, metrics


def test(model, X_test, y_test, batch_size, criterion):
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
        for i in tqdm(range(0, len(X_test), batch_size), desc="Testing"):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            
            # Convert batch to tensors  
            input_ids = indexer.as_padded_tensor(batch_X)
            labels = torch.tensor(batch_y)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(1)

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

            # elem_list = outputs.tolist()
            predictions += outputs.cpu() #[int(el[0]) for el in elem_list]
            # elem_list = labels.tolist()
            total_labels += labels.cpu() #[el for el in elem_list]

    num_batch = (len(X_test) // batch_size) if (len(X_test) // batch_size) != 0 else 1
    loss = total_loss / num_batch

    acc_value = np.round(sum(accs) / len(accs), 5).item()
    prec_value = np.round(sum(precs) / len(precs), 5).item()
    rec_value =  np.round(sum(recs) / len(recs), 5).item()
    f1_value = np.round(sum(f1s) / len(f1s), 5).item()
    metrics = {"accuracy":acc_value, "precision":prec_value, "recall":rec_value, "f1":f1_value}

    return loss, metrics, predictions, total_labels