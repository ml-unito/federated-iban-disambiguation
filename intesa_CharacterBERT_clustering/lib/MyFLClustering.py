from fluke.algorithms import CentralizedFL
from fluke.client import Client
from lib.MyFLClient import MyFLClient

class MyFLClustering(CentralizedFL):
    
    # def __init__(self,n_clients,data_splitter,hyper_params,testset_path):
    #     self.testset_path = testset_path    
    #     super().__init__(n_clients,data_splitter,hyper_params)
    
    def get_client_class(self) -> Client:
        return MyFLClient