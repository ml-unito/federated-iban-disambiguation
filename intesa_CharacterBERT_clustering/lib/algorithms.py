from fluke.client import Client
from fluke.server import Server
from fluke.algorithms import PersonalizedFL
from fluke.algorithms.lg_fedavg import LGFedAVGClient
from torch.nn import Module
from fluke.data import FastDataLoader
from typing import Iterable
from rich.console import Console

console = Console()

# FrozenBert
class FrozenBertClient(LGFedAVGClient):
    def fit(self, override_local_epochs = 0):
        if self._last_round == 1:
            console.log(f"Freezing bert modelf for client {self.index}")
            for param in self.model.get_local().parameters():
                param.requires_grad = False 

        return super().fit(override_local_epochs)


class FrozenBert(PersonalizedFL):
    def get_client_class(self):
        return FrozenBertClient

# FrozenBert

class PretrainedBertClient(LGFedAVGClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyper_params.update(bert_pretrain_epochs=kwargs["bert_pretrain_epochs"])
        self.hyper_params.update(bert_freeze_params=kwargs["bert_freeze_params"])

    def fit(self, override_local_epochs = 0):
        # Pretrain BERT model on local data
        if self._last_round == 1:
            self.pretrain_bert()

            # Freeze BERT model parameters if specified
            if self.hyper_params.bert_freeze_params:
                for param in self.model.get_local().parameters():
                    param.requires_grad = False

        return super().fit(override_local_epochs)
    
    def pretrain_bert(self):
        # Pretrain the BERT model on local data
        # This is a placeholder for the actual pretraining logic
        # Implement your pretraining logic here
        
        bert = self.model.get_local()
        bert.train()

        optimizer = self.optimizer_cfg.get_optimizer(bert.parameters())
        for epoch in range(self.hyper_params.bert_pretrain_epochs):
            console.log(f"Pretraining BERT model for client {self.index} - Epoch {epoch + 1}/{self.local_epochs}")
            # Iterate over the training set
            for batch in self.train_set:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = bert(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

class PretrainedBert(PersonalizedFL):
    def get_client_class(self):
        return PretrainedBertClient
    

# LocalBert

class LocalBertClient(Client):
    pass

class LocalBertServer(Server):
    pass

class LocalBert(PersonalizedFL):
    def get_client_class(self):
        return LocalBertClient
    
    def get_server_class(self):
        return LocalBertServer
