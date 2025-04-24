#!/usr/bin/env zsh

seed=$1

if [ -z "$seed" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

echo -e "\033[0;Splitting and preprocessing the datasets...\033[0m"
uv run split_dataset.py --seed $seed
uv run preprocessing.py

echo -e "\033[0;32mUpdating config files...\033[0m"
sed -i "s/seed: .*/seed: $seed/" config/exp_federated.yaml

if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError updating config/exp_federated.yaml\033[0m"
    exit 1
fi

echo -e "\033[0;32mStarting experiment...\033[0m"
uv run fluke federation config/exp_federation.yaml config/alg_frozen.yaml
