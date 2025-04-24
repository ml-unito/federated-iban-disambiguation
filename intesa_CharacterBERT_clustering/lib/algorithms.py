from fluke.client import Client
from fluke.server import Server
from fluke.algorithms import PersonalizedFL
from fluke.algorithms.lg_fedavg import LGFedAVGClient
from torch.nn import Module
from fluke.data import FastDataLoader
from typing import Iterable
from rich.console import Console

console = Console()

# PretrainedBert
class PretrainedBertClient(LGFedAVGClient):
    def fit(self, override_local_epochs = 0):
        if self._last_round == 1:
            console.log(f"Freezing bert modelf for client {self.index}")
            for param in self.model.get_local().parameters():
                param.requires_grad = False 

        return super().fit(override_local_epochs)


class PretrainedBert(PersonalizedFL):
    def get_client_class(self):
        return PretrainedBertClient

# FrozenBert

class FrozenBertClient(Client):
    pass

class FrozenBertServer(Server):
    pass

class FrozenBert(PersonalizedFL):
    def get_client_class(self):
        return FrozenBertClient
    
    def get_server_class(self):
        return FrozenBertServer

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
