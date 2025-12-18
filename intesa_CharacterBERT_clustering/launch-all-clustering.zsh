
# ./launchexp-clustering.zsh 9046 ./out/fl_models_9046_afb1bdd45bfa4c62a58aa14122450a9a/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-9046
# ./launchexp-clustering.zsh 23517 ./out/fl_models_23517_5ebf65a2dbc14769aa0250bfe9ed488a/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-23517
# ./launchexp-clustering.zsh 47874 ./out/fl_models_47874_c194c115e1034b208a3122f156e66090/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-47874
# ./launchexp-clustering.zsh 81789 ./out/fl_models_81789_beae0be00c444c30bcfa62caccd2806a/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-81789
# ./launchexp-clustering.zsh 67895 ./out/fl_models_67895_6b6da2a609cc42b4bfcdba5c26ec3d51/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-67895

# unique version of server
# ./launchexp-clustering.zsh 9046 ./out/fl_models_unique_9046_16658f8372f14bb58cb5b39b10387990/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-9046-unique--no-complex-iban
# ./launchexp-clustering.zsh 23517 ./out/fl_models_unique_23517_aa38960e87884a2694cbc1ea4880cf75/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-23517-unique--no-complex-iban
# ./launchexp-clustering.zsh 47874 ./out/fl_models_unique_47874_fb3600173d3c4af88279d76ec578473a/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-47874-unique--no-complex-iban
# ./launchexp-clustering.zsh 81789 ./out/fl_models_unique_81789_b71a5f5202fb4e0c9b17a99ad4d96c11/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-81789-unique--no-complex-iban
# ./launchexp-clustering.zsh 67895 ./out/fl_models_unique_67895_eef73bbaba1745c38ac2627d86580d84/r0030_server.pth ./dataset/split_dataset/df_test_pp.csv clustering-kernel-mlp-67895-unique--no-complex-iban


# Flower model with Kernel model with MLP
# ./launchexp-clustering.zsh kernel-accounts-disambiguation 9046 ../flower-mlp/out/flwr_S9046_2025-07-01_19-03-50/global_model_R30.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-kernel-mlp-9046-unique--no-complex-iban
# ./launchexp-clustering.zsh kernel-accounts-disambiguation 23517 ../flower-mlp/out/flwr_S23517_2025-07-01_18-34-33/global_model_R30.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-kernel-mlp-23517-unique--no-complex-iban
# ./launchexp-clustering.zsh kernel-accounts-disambiguation 47874 ../flower-mlp/out/flwr_S47874_2025-07-01_18-52-00/global_model_R30.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-kernel-mlp-47874-unique--no-complex-iban
# ./launchexp-clustering.zsh kernel-accounts-disambiguation 81789 ../flower-mlp/out/flwr_S81789_2025-07-01_19-27-19/global_model_R30.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-kernel-mlp-81789-unique--no-complex-iban
# ./launchexp-clustering.zsh kernel-accounts-disambiguation 67895 ../flower-mlp/out/flwr_S67895_2025-07-01_19-15-22/global_model_R30.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-kernel-mlp-67895-unique--no-complex-iban


# Flower model Full CBert
./launchexp-clustering.zsh cbert-accounts-disambiguation 9046 ../flower-mlp/out/cbert/flwr_S9046_20251204_230420/global_model_R10.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-cbert-9046--no-complex-iban
# ./launchexp-clustering.zsh cbert-accounts-disambiguation 23517 ../flower-mlp/out/cbert/flwr_S23517_20251204_200840/global_model_R10.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-cbert-23517--no-complex-iban
# ./launchexp-clustering.zsh cbert-accounts-disambiguation 47874 ../flower-mlp/out/cbert/flwr_S47874_20251204_162434/global_model_R10.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-cbert-47874--no-complex-iban
# ./launchexp-clustering.zsh cbert-accounts-disambiguation 67895 ../flower-mlp/out/cbert/flwr_S67895_20251205_082434/global_model_R10.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-cbert-67895--no-complex-iban
# ./launchexp-clustering.zsh cbert-accounts-disambiguation 81789 ../flower-mlp/out/cbert/flwr_S81789_20251205_111057/global_model_R10.pt ./dataset/split_dataset/df_test_pp.csv clustering-flwr-cbert-81895--no-complex-iban
