
# uv run kernel-classify.py nn-classify 9046
# uv run kernel-classify.py nn-classify 23517
# uv run kernel-classify.py nn-classify 30921
# uv run kernel-classify.py nn-classify 47874
# uv run kernel-classify.py nn-classify 81789

# uv run kernel-classify.py nn-classify 9046 --bert
# uv run kernel-classify.py nn-classify 23517 --bert
# uv run kernel-classify.py nn-classify 30921 --bert
# uv run kernel-classify.py nn-classify 47874 --bert
# uv run kernel-classify.py nn-classify 81789 --bert


### MLP MODEL ###

# Runs mlp model on fluke federation mode with bert feature
# ./launchexp-nn.zsh 23517 exp_kernel_nn alg_kernel_nn use-bert federation
# ./launchexp-nn.zsh 47874 exp_kernel_nn alg_kernel_nn use-bert federation
# ./launchexp-nn.zsh 9046 exp_kernel_nn alg_kernel_nn use-bert federation
# ./launchexp-nn.zsh 30921 exp_kernel_nn alg_kernel_nn use-bert federation
# ./launchexp-nn.zsh 81789 exp_kernel_nn alg_kernel_nn use-bert federation

# Runs mlp model on fluke federation mode
./launchexp-nn.zsh 23517 exp_kernel_nn alg_kernel_nn no-use-bert federation
./launchexp-nn.zsh 47874 exp_kernel_nn alg_kernel_nn no-use-bert federation
./launchexp-nn.zsh 9046 exp_kernel_nn alg_kernel_nn no-use-bert federation
# ./launchexp-nn.zsh 30921 exp_kernel_nn alg_kernel_nn no-use-bert federation
./launchexp-nn.zsh 81789 exp_kernel_nn alg_kernel_nn no-use-bert federation
./launchexp-nn.zsh 67895 exp_kernel_nn alg_kernel_nn no-use-bert federation

### LR MODEL ###

# Runs lr model on fluke federation mode with bert feature
# ./launchexp-nn.zsh 23517 exp_kernel_lr alg_kernel_lr use-bert federation
# ./launchexp-nn.zsh 47874 exp_kernel_lr alg_kernel_lr use-bert federation
# ./launchexp-nn.zsh 9046 exp_kernel_lr alg_kernel_lr use-bert federation
# ./launchexp-nn.zsh 30921 exp_kernel_lr alg_kernel_lr use-bert federation
# ./launchexp-nn.zsh 81789 exp_kernel_lr alg_kernel_lr use-bert federation

# Runs lr model on fluke federation mode
# ./launchexp-nn.zsh 23517 exp_kernel_lr alg_kernel_lr no-use-bert federation
# ./launchexp-nn.zsh 47874 exp_kernel_lr alg_kernel_lr no-use-bert federation
# ./launchexp-nn.zsh 9046 exp_kernel_lr alg_kernel_lr no-use-bert federation
# ./launchexp-nn.zsh 30921 exp_kernel_lr alg_kernel_lr no-use-bert federation
# ./launchexp-nn.zsh 81789 exp_kernel_lr alg_kernel_lr no-use-bert federation
