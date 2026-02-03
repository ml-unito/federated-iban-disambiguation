# Intra-accounts disambiguation

## Information on runs for dataset generation
To split the dataset into training and test sets, you can use the commands in the `create-dataset.zsh` file.  

Specifically, the split is performed by the `split_dataset.py` script, which allows reproducibility using a specific seed. The training set will be divided into four parts to create training sets for the four clients. After the split, each dataset created is preprocessed and saved in files named *__pp.csv_, using the `preprocessing.py` script.

All generated files will be saved in a folder named `./dataset/split_dataset_S$SEED/`, where `$SEED` indicates the seed used for splitting the dataset.

### Kernel version
In order to use Spectrum Kernel, you need to create datasets containing the similarities of the pairs. To do this, you can use the commands in the `create-sim-datasets-kernel.zsh` file.

In particular, the `kernel-classify.py` script with the `create-dataset` command creates the datasets for the training set and the test set. To obtain them for each client's training sets, use the `create-clients-datasets` command. For both commands, you need to specify the seed used for the subdivision to retrieve the correct datasets, whether you want to overwrite any previously created similarity datasets, and whether you want to use the bert feature. The last one is implemented using the CharacterBert model. The created files are saved in the `./dataset/` folder.

### Community detection
After training, the resulting model can be used to generate pairwise predictions. For each account number, we then reconstruct the connected components graph to identify clusters of names and aliases corresponding to legal entities. Each unique name is represented as a separate node in the graph. Edges are established between pairs of names when the predicted label is 0, indicating that they refer to the same entity. This process results in a collection of graphs for each account number, with each graph representing a distinct cluster of aliases for a given entity. Additionally, within each cluster, we select a representative name by identifying the longest name among the nodes.

To do this, you must first set the configuration parameters in the `cluster_params.yaml` file. Finally, to run the script, use the following command:
```bash
uv run clustering.py <command> SEED WEIGHTS_PATH DATASET_PATH [OPTIONS]
```
where:
- `<command>` is the name of the command to perform various functions. The command can be:
    - "`cbert-accounts-disambiguation`": account disambiguation using CharacterBert model;
    - "`kernel-accounts-disambiguation`": account disambiguation using Spectrum Kernel model;
- `SEED`: seed for reproducibility;
- `WEIGHTS_PATH`: the path of weights;
- `DATASET_PATH`: the path of the dataset;
- `[OPTIONS]`:
    - `--name-wandb TEXT`: name for execution on Weights & Biases  
