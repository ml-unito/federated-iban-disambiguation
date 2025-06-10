seed=$1
weigth=$2
dataset=$3
namewandb=$4

echo -e "\033[0;32mSeed $seed\033[0m"

uv run split_dataset.py --seed $seed
uv run preprocessing.py

uv run clustering.py kernel-accounts-disambiguation $seed $weigth $dataset --name-wandb $namewandb
