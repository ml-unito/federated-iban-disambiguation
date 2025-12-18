#!/bin/zsh

seeds=(9046 23517 67895 47874 81789)

echo -e "\033[0;32mCreating similarity datasets...\033[0m"

for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  uv run kernel-classify.py create-dataset $seed --overwrite --no-use-bert
  uv run kernel-classify.py create-dataset $seed --overwrite --use-bert
  uv run kernel-classify.py create-clients-datasets $seed 4 --overwrite --no-use-bert
  uv run kernel-classify.py create-clients-datasets $seed 4 --overwrite --use-bert
done