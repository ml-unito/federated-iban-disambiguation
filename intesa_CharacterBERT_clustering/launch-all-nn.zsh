#!/bin/zsh
seeds=(9046 23517 67895 47874 81789)
features=(use-bert no-use-bert)
clients=(1 2 3 4)


### MLP MODEL ###
# Runs on fluke federation mode
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    ./launchexp-nn.zsh $seed exp_kernel_nn alg_kernel_nn $feature federation
  done
done

# Runs on fluke centralize mode
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    ./launchexp-nn.zsh $seed exp_kernel_nn_centralized alg_kernel_lr $feature federation
  done
done

# Centralized (client only) kernel mlp 
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    for client in $clients; do
      echo -e "\033[0;32mClient $client\033[0m"
      ./launchexp-nn-only-clients.zsh $seed exp_kernel_nn_centralized alg_kernel_nn $feature $client
    done
  done
done



### LR MODEL ###
# Runs on fluke federation mode
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    ./launchexp-nn.zsh $seed exp_kernel_lr alg_kernel_lr $feature federation
  done
done

# Runs on fluke centralize mode
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    ./launchexp-nn.zsh $seed exp_kernel_lr_centralized alg_kernel_lr $feature federation
  done
done

# Centralized (client only) kernel lr
for seed in $seeds; do
  echo -e "\033[0;32mSeed $seed\033[0m"
  for feature in $features; do
    for client in $clients; do
      echo -e "\033[0;32mClient $client\033[0m"
      ./launchexp-nn-only-clients.zsh $seed exp_kernel_lr_centralized alg_kernel_lr $feature $client
    done
  done
done




### Runs without Fluke ###

# uv run kernel-classify.py nn-classify 23517 --no-bert
# uv run kernel-classify.py nn-classify 47874 --no-bert
# uv run kernel-classify.py nn-classify 9046 --no-bert
# uv run kernel-classify.py nn-classify 67895 --no-bert
# uv run kernel-classify.py nn-classify 81789 --no-bert

# uv run kernel-classify.py nn-classify 23517 --bert
# uv run kernel-classify.py nn-classify 47874 --bert
# uv run kernel-classify.py nn-classify 9046 --bert
# uv run kernel-classify.py nn-classify 67895 --bert
# uv run kernel-classify.py nn-classify 81789 --bert

# uv run kernel-classify.py classify 23517 --no-bert
# uv run kernel-classify.py classify 47874 --no-bert
# uv run kernel-classify.py classify 9046 --no-bert
# uv run kernel-classify.py classify 67895 --no-bert
# uv run kernel-classify.py classify 81789 --no-bert

# uv run kernel-classify.py classify 23517 --bert
# uv run kernel-classify.py classify 47874 --bert
# uv run kernel-classify.py classify 9046 --bert
# uv run kernel-classify.py classify 67895 --bert
# uv run kernel-classify.py classify 81789 --bert