from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
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


class CharacterBertForClassificationOptimizedFreezedSeparated(nn.Module):
    def __init__(self, num_labels=2):
        """ Add classification layer with a sigmoid activation function on the last level"""
        super(CharacterBertForClassificationOptimizedFreezedSeparated, self).__init__()
        self.character_bert = CharacterBertModel.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
        
        # Freeze the CharacterBert model
        self.character_bert.eval()
        for param in self.character_bert.parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(0.2)
        self.hidden = nn.Linear(1536, 200)
        self.classifier = nn.Linear(200, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self, input_ids_1, input_ids_2):
        """Forward pass"""
        outputs1 = self.character_bert(input_ids_1)[0]
        pooled_output1 = outputs1[:, 0, :]

        outputs2 = self.character_bert(input_ids_2)[0]
        pooled_output2 = outputs2[:, 0, :]
        
        concatenated_output = torch.cat((pooled_output1, pooled_output2), dim=1)
        
        hidden_logits = self.relu(self.hidden(concatenated_output))
        logits = self.classifier(hidden_logits)
        x = self.sigmoid(logits)
        
        return x


def train(model, X_train, y_train, batch_size, optimizer, criterion, scheduler):
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
   
    for i in tqdm(range(0, len(X_train), batch_size), desc="Training"):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        batch_X_first_names, batch_X_second_names = zip(*batch_X)

        # Convert batch to tensors  
        input_ids_1 = indexer.as_padded_tensor(batch_X_first_names)
        input_ids_1 = input_ids_1.to(device)

        input_ids_2 = indexer.as_padded_tensor(batch_X_second_names)
        input_ids_2 = input_ids_2.to(device)

        labels = torch.tensor(batch_y) 
        labels = labels.to(device)
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()                               # Clear gradients
        outputs = model(input_ids_1, input_ids_2)           # Forward pass
        loss = criterion(outputs, labels.float())           # Compute loss
        total_loss += loss.item()
        
        # get eval metrics
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
        # for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]

            batch_X_first_names, batch_X_second_names = zip(*batch_X)
            
            # Convert batch to tensors  
            input_ids_1 = indexer.as_padded_tensor(batch_X_first_names)
            input_ids_1 = input_ids_1.to(device)

            input_ids_2 = indexer.as_padded_tensor(batch_X_second_names)
            input_ids_2 = input_ids_2.to(device)

            labels = torch.tensor(batch_y)
            labels = labels.to(device)
            labels = labels.unsqueeze(1)

            # Forward pass
            outputs = model(input_ids_1, input_ids_2)

            # Compute loss
            loss = criterion(outputs, labels.float())
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
