#!/usr/bin/env zsh

seed=$1
exp=config/$2.yaml
alg=config/$3.yaml
bert=$4
client=$5

if [ -z "$seed" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

if [[ "$bert" == "use-bert" ]]
then
    echo -e "\033[0;32mStarting experiment with seed $seed using bert feature...\033[0m"

    echo -e "\033[0;32mUpdating config files...\033[0m"
    sed -i "s/seed: .*/seed: $seed/" $exp
    sed -i "s/SEED/$seed/" $exp
    sed -i "s/bert: .*/bert: true/" $exp
    sed -i "s/input_dim: .*/input_dim: 8/" $alg

    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mError updating $exp\033[0m"
        exit 1
    fi
else
    echo -e "\033[0;32mStarting experiment with seed $seed...\033[0m"

    echo -e "\033[0;32mUpdating config files...\033[0m"
    sed -i "s/seed: .*/seed: $seed/" $exp
    sed -i "s/SEED/$seed/" $exp
    sed -i "s/bert: .*/bert: false/" $exp
    sed -i "s/input_dim: .*/input_dim: 7/" $alg

    if [ $? -ne 0 ]; then
        echo -e "\033[0;31mError updating $exp\033[0m"
        exit 1
    fi
fi

sed -i "s/client: .*/client: $client/" $exp
sed -i "s/sim_train_path: .*/sim_train_path: dataset\/similarity_client%d_train_seed_%d%s.csv/" $exp

echo -e "\033[0;32mStarting federation...\033[0m"

uv run fluke centralized $exp $alg

sed -i "s/'$seed'/'SEED'/" $exp 
sed -i "s/fl_models_S$seed/fl_models_SSEED/" $exp 
sed -i "s/client: .*/client: None/" $exp
sed -i "s/sim_train_path: .*/sim_train_path: dataset\/similarity_train_seed_%d%s.csv/" $exp
