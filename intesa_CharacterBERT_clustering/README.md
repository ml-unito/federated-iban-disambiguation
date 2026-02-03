# Intra-accounts disambiguation

## Runs information
To split the dataset into training and test sets, you can use the commands in the `create-dataset.zsh` file.  

Specifically, the split is performed by the `split_dataset.py` script, which allows reproducibility using a specific seed. The training set will be divided into four parts to create training sets for the four clients. After the split, each dataset created is preprocessed and saved in files named *__pp.csv_, using the `preprocessing.py` script.

All generated files will be saved in a folder named `./dataset/split_dataset_S$SEED/`, where `$SEED` indicates the seed used for splitting the dataset.

### Kernel version
In order to use Spectrum Kernel, you need to create datasets containing the similarities of the pairs. To do this, you can use the commands in the `create-sim-datasets-kernel.zsh` file.

In particular, the `kernel-classify.py` script with the `create-dataset` command creates the datasets for the training set and the test set. To obtain them for each client's training sets, use the `create-clients-datasets` command. For both commands, you need to specify the seed used for the subdivision to retrieve the correct datasets, whether you want to overwrite any previously created similarity datasets, and whether you want to use the bert feature. The last one is implemented using the CharacterBert model. The created files are saved in the `./dataset/` folder.
