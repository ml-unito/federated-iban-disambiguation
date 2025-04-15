import copy
import pandas as pd
import json
from fluke import FlukeENV
from fluke.data import DataSplitter
from fluke.evaluation import ClassificationEval
from fluke.utils import OptimizerConfigurator, get_loss, get_model
from fluke.utils.log import get_logger
from rich.progress import track
from lib.datasetManipulation import *
from federated_learning import load_parameters, create_dummy_data_container


# download_pre_trained_model()
from lib.CBertClassif import *
# from lib.CBertClassifFrz import *
# from lib.CBertClassifFrzSep import *


with open("./config/fl_parameters.json", "r") as data_file:
    fl_parameters = json.load(data_file)


DIR_DATASET_PATH = fl_parameters["dir_dataset_path"]
TRAIN_PATH = fl_parameters["train_path"]
TEST_PATH = fl_parameters["test_path"]
EXP_PATH = fl_parameters["config"]["exp_path"]
ALG_PATH = fl_parameters["config"]["alg_path"]
SAVE_MODELS = fl_parameters["save_models"]
PATH_SAVE_MODELS = fl_parameters["path_save_models"]


def clients_only():
    config_exp, config_alg = load_parameters(EXP_PATH, ALG_PATH)

    settings = FlukeENV()
    settings.set_seed(config_exp["exp"]["seed"])
    settings.set_device(config_exp["exp"]["device"]) 

    datasets = create_dummy_data_container(num_clients=config_exp["protocol"]["n_clients"], 
                                           train_path=TRAIN_PATH, test_path=TEST_PATH, dir_dataset_path=DIR_DATASET_PATH, 
                                           client_test=True)

    settings.set_evaluator(ClassificationEval(eval_every=1, n_classes=datasets.num_classes))
    settings.set_eval_cfg(config_exp["eval"])

    device = FlukeENV().get_device()

    hp = config_alg.hyperparameters
    data_splitter = DataSplitter(dataset=datasets)
    (clients_tr_data, clients_te_data), _ = data_splitter.assign(config_exp.protocol.n_clients, hp.client.batch_size)

    criterion = get_loss(hp.client.loss)
    client_evals = []
    epochs = 20

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