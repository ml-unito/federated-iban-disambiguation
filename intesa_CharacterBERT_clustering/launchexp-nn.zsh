#!/usr/bin/env zsh

seed=$1
exp=config/$2.yaml
alg=config/$3.yaml
bert=$4
mode=$5

if [ -z "$seed" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

if [[ "$bert" == "use-bert" ]]
then
    echo -e "\033[0;32mStarting experiment with seed $seed using bert feature...\033[0m"

    echo -e "\033[0;32mUpdating config files...\033[0m"
    sed -i "s/seed: .*/seed: $seed/" $exp
    sed -i "s/'SEED'/'$seed'/" $exp
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

echo -e "\033[0;32mStarting federation...\033[0m"

if [[ "$mode" == "federation" ]]
then
    uv run fluke federation $exp $alg
elif [[ "$mode" == "centralized" ]]
then
    uv run fluke centralized $exp $alg
else
    uv run fluke clients-only $exp $alg
fi

sed -i "s/'$seed'/'SEED'/" $exp 
sed -i "s/fl_models_$seed/fl_models_SEED/" $exp 
