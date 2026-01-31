#!/bin/zsh

seeds=(9046 23517 67895 47874 81789)

## Fluke models of Full CBert (batch 512) ##
# fl_framework="fluke"
# model="cbert"
# weights_paths=(
#   "./out/fl_bert_models_S9046_9eb503ed7c2b4556abb4df916163581e/r0010_server.pth"
#   "./out/fl_bert_models_S23517_89c01d149c104f83918dc0144051ccd5/r0010_server.pth"
#   "./out/fl_bert_models_S67895_bf5d821bee144924a67c1e6c545d2e78/r0010_server.pth"
#   "./out/fl_bert_models_S47874_61bb95aee8b44302a6422405e5555ac9/r0010_server.pth"
#   "./out/fl_bert_models_S81789_e1f33b9b63d64fdbb0d14f5d355ed89c/r0010_server.pth"
# )

## Fluke models of Full CBert (batch 256) ##
# fl_framework="fluke"
# model="cbert"
# weights_paths=(
#   "./out/fluke-cbert-B256/fl_bert_models_S9046_2939e09d850844ecb0f2dc30c80dcbed/r0010_server.pth"
#   "./out/fluke-cbert-B256/fl_bert_models_S23517_737119ef348140fba66958d42015055d/r0010_server.pth"
#   "./out/fluke-cbert-B256/fl_bert_models_S67895_f5cebb21d6894805966f007b2a7e642c/r0010_server.pth"
#   "./out/fluke-cbert-B256/fl_bert_models_S47874_1c78416ba587403489a6fc597f94da6e/r0010_server.pth"
#   "./out/fluke-cbert-B256/fl_bert_models_S81789_1340cecc5f634bbb875edac873c88315/r0010_server.pth"
# )

## Flower models of Full CBert (batch 128) ##
# fl_framework="flwr"
# model="cbert"
# weights_paths=(
#   "../flower-bertmlp/out/flwr_S9046_20260123_185447/global_model_R10.pt"
#   "../flower-bertmlp/out/flwr_S23517_20260124_115448/global_model_R10.pt"
#   "../flower-bertmlp/out/flwr_S67895_20260124_233918/global_model_R10.pt"
#   "../flower-bertmlp/out/flwr_S47874_20260124_152245/global_model_R10.pt"
#   "../flower-bertmlp/out/flwr_S81789_20260125_163619/global_model_R10.pt"
# )

## Flower models of Full CBert (batch 256) ##
fl_framework="flwr"
model="cbert"
weights_paths=(
  "../flower-bertmlp/out/flwr-bert-B256/flwr_S9046_20260127_142403/global_model_R10.pt"
  "../flower-bertmlp/out/flwr-bert-B256/flwr_S23517_20260128_002018/global_model_R10.pt"
  "../flower-bertmlp/out/flwr-bert-B256/flwr_S67895_20260128_141402/global_model_R10.pt"
  "../flower-bertmlp/out/flwr-bert-B256/flwr_S47874_20260128_075941/global_model_R10.pt"
  "../flower-bertmlp/out/flwr-bert-B256/flwr_S81789_20260129_000535/global_model_R10.pt"
)


## Fluke models of Kernel MLP no bert ##
# fl_framework="fluke"
# model="kernel"
# weights_paths=(
#   "./out/fl_models_S9046_6df8dd20ef664ab8b1fa08527f563e81/r0030_server.pth"
#   "./out/fl_models_S23517_20141fdf304642eebcdbb68f26a7b444/r0030_server.pth"
#   "./out/fl_models_S67895_3fea22fd59dd497791636098ff566320/r0030_server.pth"
#   "./out/fl_models_S47874_58b4fd8ff53d46639ef1c088771f078f/r0030_server.pth"
#   "./out/fl_models_S81789_684278c8abec4ab9baf043dea2abe4a9/r0030_server.pth"
# )

## Flower models of Kernel MLP no bert ##
# fl_framework="flwr"
# model="kernel"
# weights_paths=(
#   "../flower-mlp/out/flwr_S9046_2026-01-27_11-50-50/global_model_R30.pt"
#   "../flower-mlp/out/flwr_S23517_2026-01-27_12-01-53/global_model_R30.pt"
#   "../flower-mlp/out/flwr_S67895_2026-01-27_11-39-38/global_model_R30.pt"
#   "../flower-mlp/out/flwr_S47874_2026-01-27_13-10-42/global_model_R30.pt"
#   "../flower-mlp/out/flwr_S81789_2026-01-27_13-27-21/global_model_R30.pt"
# )

for i in {1..5}; do
  seed=${seeds[i]}
  echo -e "\033[0;32mSeed $seed\033[0m"
  weights_path=${weights_paths[i]}
  dataset="./dataset/split_dataset_S$seed/df_test_pp.csv"
  name_wandb="clustering-$fl_framework-$model-S$seed--no-complex-iban"

  uv run clustering.py $model-accounts-disambiguation $seed $weights_path $dataset --name-wandb $name_wandb
done
