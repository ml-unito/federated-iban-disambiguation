import copy
import sys
import pandas as pd
import json
import yaml
import time
from fluke import DDict
import fluke.utils.log as log
from fluke import FlukeENV
from fluke.data import DataSplitter
from fluke.algorithms.fedavg import FedAVG
from fluke.data.datasets import DataContainer, DummyDataContainer, FastDataLoader
from fluke.evaluation import ClassificationEval
from lib.download import download_pre_trained_model
from sklearn.model_selection import train_test_split
from datetime import datetime
from transformers import BertTokenizer
from fluke.utils import Configuration
from fluke.utils import OptimizerConfigurator, get_loss, get_model
from fluke.utils.log import get_logger
from rich.progress import track
from lib.datasetManipulation import *


download_pre_trained_model()
# from lib.CBertClassif import *
from lib.CBertClassifFrz import *
# from lib.CBertClassifFrzSep import *


with open("./config/fl_parameters.json", "r") as data_file:
    fl_parameters = json.load(data_file)


DIR_DATASET_PATH = "./dataset/4Clients/"
EXP_PATH = fl_parameters["config"]["exp_path"]
ALG_PATH = fl_parameters["config"]["alg_path"]
SAVE_MODELS = fl_parameters["save_models"]
PATH_SAVE_MODELS = fl_parameters["path_save_models"]



def extract_x_and_y(dataset: pd.DataFrame, tokenizer) -> list:
    tokenized_texts = tokenize_dataset(dataset, tokenizer)
    x, y = lookup_table(tokenized_texts, dataset)
    return x, y


def create_dummy_data_container(
    num_clients: int, client_test=False
) -> DummyDataContainer:
    # Loads tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        "./character_bert_model/pretrained-models/general_character_bert/"
    )

    if num_clients == 1:
        # Loads datasets
        df = pd.read_csv(DIR_DATASET_PATH + "benchmark_intesa_preprocessed_couple.csv")
        x, y = extract_x_and_y(df, tokenizer)

        X_train, X_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, random_state=42, stratify=y
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, train_size=0.5, random_state=42, stratify=y_test
        )

        # Creates FastDataLoader for client data and server data
        fdl_clt = FastDataLoader(X_train, y_train, num_labels=2, batch_size=512)
        fdl_srv = FastDataLoader(X_test, y_test, num_labels=2, batch_size=512)

        return DummyDataContainer(
            clients_tr=[fdl_clt],
            clients_te=[fdl_srv] if client_test else [None],
            server_data=fdl_srv,
            num_classes=2,
        )
    else:
        # Loads client datasets and server dataset
        df_clients = [
            pd.read_csv(DIR_DATASET_PATH + "client" + str(i) + "_train_couple.csv")
            for i in range(1, num_clients + 1)
        ]
        df_server = pd.read_csv(DIR_DATASET_PATH + "server_test_couple.csv")

        # Creates FastDataLoader for each client data
        fdl_clts = []
        for df_client in df_clients:
            x, y = extract_x_and_y(df_client, tokenizer)
            fdl = FastDataLoader(x, y, num_labels=2, batch_size=512)
            fdl_clts.append(fdl)

        # Creates FastDataLoader for server data
        x, y = extract_x_and_y(df_server, tokenizer)
        fdl_srv = FastDataLoader(x, y, num_labels=2, batch_size=512)

        return DummyDataContainer(
            clients_tr=fdl_clts,
            clients_te=[fdl_srv] * num_clients if client_test else [None] * num_clients,
            server_data=fdl_srv,
            num_classes=2,
        )


def load_parameters() -> list:
    config_file_exp = open(EXP_PATH)
    config_exp = yaml.safe_load(config_file_exp)

    config_file_alg = open(ALG_PATH)
    config_alg = yaml.safe_load(config_file_alg)

    return DDict(config_exp), DDict(config_alg)


def clients_only():
    config_exp, config_alg = load_parameters()

    settings = FlukeENV()
    settings.set_seed(config_exp["exp"]["seed"])
    settings.set_device(config_exp["exp"]["device"]) 

    datasets = create_dummy_data_container(num_clients=config_exp["protocol"]["n_clients"], client_test=True)

    settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=datasets.num_classes))
    settings.set_eval_cfg(config_exp["eval"])

    device = FlukeENV().get_device()

    hp = config_alg.hyperparameters
    data_splitter = DataSplitter(
        dataset=datasets
    )
    (clients_tr_data, clients_te_data), _ = data_splitter.assign(
        config_exp.protocol.n_clients, hp.client.batch_size
    )

    criterion = get_loss(hp.client.loss)
    client_evals = []
    epochs =20

    progress = track(
        enumerate(zip(clients_tr_data, clients_te_data)),
        total=len(clients_tr_data),
        description="Clients training...",
    )

    exp_name = "Clients-only"
    log = get_logger(config_exp.logger.name, name=exp_name, **config_exp.logger.exclude("name"))
    cfg = copy.copy(config_exp)
    cfg.update(config_alg)

    log.init(**cfg, exp_id=exp_name)

    running_evals = {c: [] for c in range(config_exp.protocol.n_clients)}
    for i, (train_loader, test_loader) in progress:
        log.log(f"Client [{i+1}/{config_exp.protocol.n_clients}]")
        model = get_model(mname=hp.model, **hp.net_args if "net_args" in hp else {})
        hp.client.optimizer.name = torch.optim.SGD
        optimizer_cfg = OptimizerConfigurator(
            optimizer_cfg=hp.client.optimizer, scheduler_cfg=hp.client.scheduler
        )
        model.to(device)
        optimizer, scheduler = optimizer_cfg(model)
        evaluator = ClassificationEval(
            eval_every=1, n_classes=datasets.num_classes
        )
        
        for e in range(epochs):
            model.train()
            model.to(device)
            for _, (X, y) in enumerate(train_loader):
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
            scheduler.step()

            client_eval = evaluator.evaluate(
                e + 1, model, test_loader, criterion, device=device
            )
            running_evals[i].append(client_eval)

        log.pretty_log(client_eval, title=f"Client [{i}] Performance")
        client_evals.append(client_eval)
        #model.cpu()

    for e in range(epochs):
        for c in running_evals:
            log.add_scalars(f"Client[{c}]", running_evals[c][e], e + 1)

    client_mean = pd.DataFrame(client_evals).mean(numeric_only=True).to_dict()
    client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
    log.pretty_log(client_mean, title="Overall local performance")


if __name__ == "__main__":
    clients_only()