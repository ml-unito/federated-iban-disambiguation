#!/bin/zsh
seeds=(9046 23517 67895 47874 81789)

### Full CharacterBert ###
for seed in $seeds; do
    ./launchexp.zsh $seed exp alg
done

### Frozen Bert ###
for seed in $seeds; do
    ./launchexp.zsh $seed exp_frozen alg_frozen
done

### Pretrained Bert ###
for seed in $seeds; do
    ./launchexp.zsh $seed exp_pretrained alg_pretrained
done
