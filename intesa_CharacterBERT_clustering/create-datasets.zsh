#!/bin/zsh

# 30921
seeds=(9046 23517 67895 47874 81789)

echo -e "\033[0;32mSplitting and preprocessing the datasets...\033[0m"

for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  uv run split_dataset.py --seed $seed
  uv run preprocessing.py split-dataset

  uv run kernel-classify.py create-dataset $seed --overwrite --no-use-bert
  uv run kernel-classify.py create-clients-datasets $seed 4 --overwrite --no-use-bert
done