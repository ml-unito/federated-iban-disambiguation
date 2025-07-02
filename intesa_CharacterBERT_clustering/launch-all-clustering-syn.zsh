
# Test on synthetic dataset

# uv run preprocessing.py dataset ./dataset/syn_dataset_25-06-2025_10-24-09.csv

## FLUKE MODELS
# echo -e "\033[0;32mSeed 9046\033[0m"
# uv run clustering.py kernel-accounts-disambiguation 9046 ./out/fl_models_unique_9046_16658f8372f14bb58cb5b39b10387990/r0030_server.pth ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-kernel-mlp-9046-syn-data

# echo -e "\033[0;32mSeed 23517\033[0m"
# uv run clustering.py kernel-accounts-disambiguation 23517 ./out/fl_models_unique_23517_aa38960e87884a2694cbc1ea4880cf75/r0030_server.pth ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-kernel-mlp-23517-syn-data

# echo -e "\033[0;32mSeed 47874\033[0m"
# uv run clustering.py kernel-accounts-disambiguation 47874 ./out/fl_models_unique_47874_fb3600173d3c4af88279d76ec578473a/r0030_server.pth ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-kernel-mlp-47874-syn-data

# echo -e "\033[0;32mSeed 81789\033[0m"
# uv run clustering.py kernel-accounts-disambiguation 81789 ./out/fl_models_unique_81789_b71a5f5202fb4e0c9b17a99ad4d96c11/r0030_server.pth ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-kernel-mlp-81789-syn-data

# echo -e "\033[0;32mSeed 67895\033[0m"
# uv run clustering.py kernel-accounts-disambiguation 67895 ./out/fl_models_unique_67895_eef73bbaba1745c38ac2627d86580d84/r0030_server.pth ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-kernel-mlp-67895-syn-data


## FLOWER MODELS

echo -e "\033[0;32mSeed 9046\033[0m"
uv run clustering.py kernel-accounts-disambiguation 9046 ../flower-mlp/out/flwr_S9046_2025-07-01_19-03-50/global_model_R30.pt ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-flwr-kernel-mlp-9046-syn-data

echo -e "\033[0;32mSeed 23517\033[0m"
uv run clustering.py kernel-accounts-disambiguation 23517 ../flower-mlp/out/flwr_S23517_2025-07-01_18-34-33/global_model_R30.pt ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-flwr-kernel-mlp-23517-syn-data

echo -e "\033[0;32mSeed 47874\033[0m"
uv run clustering.py kernel-accounts-disambiguation 47874 ../flower-mlp/out/flwr_S47874_2025-07-01_18-52-00/global_model_R30.pt ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-flwr-kernel-mlp-47874-syn-data

echo -e "\033[0;32mSeed 81789\033[0m"
uv run clustering.py kernel-accounts-disambiguation 81789 ../flower-mlp/out/flwr_S81789_2025-07-01_19-27-19/global_model_R30.pt ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-flwr-kernel-mlp-81789-syn-data

echo -e "\033[0;32mSeed 67895\033[0m"
uv run clustering.py kernel-accounts-disambiguation 67895 ../flower-mlp/out/flwr_S67895_2025-07-01_19-15-22/global_model_R30.pt ./dataset/syn_dataset_25-06-2025_10-24-09.csv --name-wandb clustering-flwr-kernel-mlp-67895-syn-data

