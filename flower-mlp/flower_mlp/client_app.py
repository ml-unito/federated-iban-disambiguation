"""flower-mlp: A Flower / PyTorch app."""

import torch
import yaml
import flwr as fl
from flwr.common import Context

from flower_mlp.task import MLP, set_parameters, get_parameters, train, test, log_detailed_metrics, load_data, DEVICE, set_system_seed


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Define Flower Client and client_fn
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, eval_config, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.eval_config = eval_config
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        # Ottieni round corrente per logging
        round_num = config.get("round_num", 0)
        
        # Imposta i parametri e esegui il training
        self.set_parameters(parameters)
        
        # TRAINING
        train_loss = train(self.model, self.trainloader, epochs=self.local_epochs, device=DEVICE)
        
        # Return updated parameters and training stats
        return get_parameters(self.model), len(self.trainloader.dataset), {"loss": train_loss}

    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            round_num = config.get("round_num", 0)
            print(f"[DEBUG] Client: valutazione round {round_num}")
            
            # Assicurati di restituire sempre un risultato, anche se eval è disabilitata
            if not self.eval_config["server"]:
                print("[DEBUG] eval['server']=False, restituisco valori di default")
                return 0.0, 1, {"accuracy": 0.0, "loss": 0.0}
            
            # Prepara i dati di test
            all_test_features = []
            all_test_labels = []
            
            for features, labels in self.valloader:
                all_test_features.append(features)
                all_test_labels.append(labels)
                
            test_x = torch.cat(all_test_features, dim=0)
            test_y = torch.cat(all_test_labels, dim=0)
            
            # Valuta il modello (versione base)
            loss, accuracy = test(self.model, self.valloader, device=DEVICE)
            
            # Crea sempre un dizionario con almeno le metriche base
            metrics = {"accuracy": float(accuracy)}
            
            # Se il server eval è attivato, calcola metriche dettagliate
            if self.eval_config["server"]:
                all_train_features = []
                all_train_labels = []
                
                for features, labels in self.trainloader:
                    all_train_features.append(features)
                    all_train_labels.append(labels)
                    
                train_x = torch.cat(all_train_features, dim=0)
                train_y = torch.cat(all_train_labels, dim=0)
                
                # Log delle metriche dettagliate
                detailed_metrics = log_detailed_metrics(
                    self.model, 
                    train_x, train_y, 
                    test_x, test_y, 
                    loss,
                    round_num,
                    phase="global"
                )
                
                # Aggiungi le metriche dettagliate al risultato
                metrics.update(detailed_metrics)
            
            return loss, len(test_y), metrics
        except Exception as e:
            print(f"[ERROR] Errore durante la valutazione: {e}")
            # Restituisci valori di default in caso di errore
            return 0.0, 1, {"error": str(e), "accuracy": 0.0, "loss": 0.0}


def client_fn(context: Context):
    # Carica la configurazione
    config = load_config("config/flower_exp_kernel_nn.yaml")

    # Ottieni l'ID del client dalla configurazione del nodo
    client_id = int(context.node_config["partition-id"]) + 1
    
    # Fissa il seed di sistema per replicabilità (prima di creare il modello)
    set_system_seed()
    print(f"Client {client_id}: system_seed=42 applicato")

    # Carica i dati
    trainloader, valloader = load_data(client_id, config["data"])

    # Determina la dimensione di input dal primo batch
    sample_features, _ = next(iter(trainloader))
    input_size = sample_features.shape[1]

    # Crea il modello
    model = MLP(input_dim=input_size, hidden_dim=128, output_dim=2).to(DEVICE)

    # Numero di epoche locali (prendi dalla configurazione se disponibile)
    local_epochs = config.get("protocol", {}).get("local_epochs", 4)

    # Crea e restituisci il client Flower con le impostazioni eval
    return FlowerClient(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        eval_config=config["eval"],
        local_epochs=local_epochs
    ).to_client()


# Flower ClientApp
app = fl.client.ClientApp(client_fn=client_fn)
