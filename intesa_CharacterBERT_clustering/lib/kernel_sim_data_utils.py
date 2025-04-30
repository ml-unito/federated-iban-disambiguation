import os
import sys
import pandas as pd
import torch
from character_bert_model.modeling.character_bert import CharacterBertModel
from character_bert_model.utils.character_cnn import CharacterIndexer

from lib.datasetManipulation import labeled_pairs
from rich.console import Console
from lib import string_kernels as sk
from transformers import BertTokenizer

from rich.progress import track

console = Console()


def load_df(train_path: str, test_path: str):
    cp_train_df = pd.read_csv(train_path)
    cp_test_df = pd.read_csv(test_path)

    train_pairs, train_labels = labeled_pairs(cp_train_df)
    test_pairs, test_labels = labeled_pairs(cp_test_df)

    train_xy = list(zip(train_pairs, train_labels))
    test_xy = list(zip(test_pairs, test_labels))

    # Create a dataframe from the train and test data, split the pairs into two columns
    train_xy = [[pair[0], pair[1], label] for pair, label in train_xy]
    train_df = pd.DataFrame(train_xy, columns=["str1", "str2", "label"])

    test_xy = [[pair[0], pair[1], label] for pair, label in test_xy]
    test_df = pd.DataFrame(test_xy, columns=["str1", "str2", "label"])

    return train_df, test_df

def kernel_features_from_pairs(s1, s2, n_features=4):
    return [
        sk.spectrum_kernel([s1], [s2], p=i)[0].item() for i in range(1, n_features + 1)
    ]

def tokenize(s: str):
    if not hasattr(tokenize, "tokenizer"):
        tokenize.tokenizer = BertTokenizer.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')

        
    x = tokenize.tokenizer.basic_tokenizer.tokenize(s)

    # Add [CLS] and [SEP]
    x = ['[CLS]', *x, '[SEP]']

    # Convert token sequence into character indices
    indexer = CharacterIndexer()
    batch = [x]  # This is a batch with a single token sequence x
    batch_ids = indexer.as_padded_tensor(batch).to("cuda:0")  # FIXME

    return batch_ids

def bert_similarity(s1, s2, use_bert=False) -> list[float]:
    if not use_bert:
        return []

    if not hasattr(bert_similarity, "bert"):
        bert_similarity.bert = CharacterBertModel.from_pretrained('./character_bert_model/pretrained-models/general_character_bert/')
        bert_similarity.bert.eval()
        bert_similarity.bert.to("cuda:0") # FIXME

    indx1 = tokenize(s1)
    indx2 = tokenize(s2)

    emb1 = bert_similarity.bert(indx1)[0][:, 0, :].flatten()
    emb2 = bert_similarity.bert(indx2)[0][:, 0, :].flatten()
    sim = emb1.dot(emb2)
    sim /= (torch.norm(emb1) * torch.norm(emb2))

    return [sim.item()]

def save_sim_data(fname:str, data:pd.DataFrame, n_features, oversample:bool=False, use_bert:bool=False):

    sims = []
    for i, (s1, s2, label) in enumerate(track(data.itertuples(index=False), total=len(data))):
        kernel_features = kernel_features_from_pairs(s1, s2, n_features=n_features)
        bert_features = bert_similarity(s1, s2, use_bert=use_bert)
        sims.append(kernel_features + bert_features + [label])
    
    n_features = len(sims[0]) - 1

    if oversample:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        X = [s[:n_features] for s in sims]
        y = [s[n_features] for s in sims]

        console.log(f"Oversampling data")
        X_res, y_res = sm.fit_resample(X, y)
        sims = [[*s[:n_features], label] for s, label in zip(X_res, y_res)]

    with open(fname, "w") as f:
        f.write(f"{','.join([f'p{i}' for i in range(1,n_features+1)])},label\n")
        for sim in sims:
            f.write(f",".join([str(s) for s in sim]) + "\n")

def load_sim_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df