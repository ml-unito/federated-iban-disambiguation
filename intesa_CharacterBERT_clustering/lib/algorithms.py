from fluke.algorithms import PersonalizedFL
from fluke.algorithms.lg_fedavg import LGFedAVGClient
from fluke.utils import FlukeENV

# FrozenBert
class FrozenBertClient(LGFedAVGClient):
    def fit(self, override_local_epochs = 0):
        console = FlukeENV().get_progress_bar('clients').console

        if self._last_round == 1:
            console.log(f"Freezing bert modelf for client {self.index}")
            for param in self.model.get_local().parameters():
                param.requires_grad = False 

        return super().fit(override_local_epochs)


class FrozenBert(PersonalizedFL):
    def get_client_class(self):
        return FrozenBertClient

# PretrainedBert
class PretrainedBertClient(LGFedAVGClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not "bert_pretrain_epochs" in kwargs:
            raise ValueError("bert_pretrain_epochs is required")
        
        if not "bert_freeze_params" in kwargs:
            raise ValueError("bert_freeze_params is required")

        self.hyper_params.update(bert_pretrain_epochs=kwargs["bert_pretrain_epochs"])
        self.hyper_params.update(bert_freeze_params=kwargs["bert_freeze_params"])

    def fit(self, override_local_epochs = 0):
        # Pretrain BERT model on local data
        if self._last_round == 1:
            self.pretrain_bert()

        return super().fit(override_local_epochs)
    
    def pretrain_bert(self):
        # Pretrain the BERT model on local data
        console = FlukeENV().get_progress_bar('clients').console
        console.log(f"Pretraining BERT model for client {self.index}")
        super().fit(override_local_epochs=self.hyper_params.bert_pretrain_epochs)
        console.log(f"Finished pretraining BERT model for client {self.index}")

        # Freeze BERT model parameters if specified
        if self.hyper_params.bert_freeze_params:
            console.log(f"Freezing BERT model parameters for client {self.index}")
            for param in self.model.get_local().parameters():
                param.requires_grad = False

class PretrainedBert(PersonalizedFL):
    def get_client_class(self):
        return PretrainedBertClient
    

# LocalBert
# ------------------------------------------------------
# Not necessary? It should be exactly as PretrainedBert, but without the
# pretraining step and the freeze of the bert model parameters.
# In the end one can use the PretrainedBertClient with 0 epochs and 
# False for the freeze parameter.

# class LocalBertClient(Client):
#     pass

# class LocalBertServer(Server):
#     pass

# class LocalBert(PersonalizedFL):
#     def get_client_class(self):
#         return LocalBertClient
    
#     def get_server_class(self):
#         return LocalBertServer
