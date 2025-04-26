#!/usr/bin/env zsh

seed=$1
exp=config/$2.yaml
alg=config/$3.yaml

if [ -z "$seed" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

echo -e "\033[0;32mSplitting and preprocessing the datasets...\033[0m"
uv run split_dataset.py --seed $seed
uv run preprocessing.py

echo -e "\033[0;32mUpdating config files...\033[0m"
sed -i "s/seed: .*/seed: $seed/" $exp

if [ $? -ne 0 ]; then
    echo -e "\033[0;31mError updating $exp\033[0m"
    exit 1
fi

echo -e "\033[0;32mStarting experiment...\033[0m"
uv run fluke federation $exp $alg
