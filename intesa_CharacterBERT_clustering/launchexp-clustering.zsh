mode=$1
seed=$2
weigth=$3
dataset=$4
namewandb=$5

echo -e "\033[0;32mSeed $seed\033[0m"

uv run split_dataset.py --seed $seed
uv run preprocessing.py split-dataset

uv run clustering.py $mode $seed $weigth $dataset --name-wandb $namewandb
