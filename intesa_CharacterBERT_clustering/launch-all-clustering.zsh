#!/bin/zsh

seeds=(9046 23517 67895 47874 81789)
fl_framework="fluke"
# fl_framework="flwr"
mode="cbert-accounts-disambiguation"
# mode="kernel-accounts-disambiguation"

## Fluke models of Full CBert ##
models=(
  "./out/fl_bert_models_S9046_9eb503ed7c2b4556abb4df916163581e/r0010_server.pth"
  "./out/fl_bert_models_S23517_89c01d149c104f83918dc0144051ccd5/r0010_server.pth"
  "./out/fl_bert_models_S67895_bf5d821bee144924a67c1e6c545d2e78/r0010_server.pth"
  "./out/fl_bert_models_S47874_61bb95aee8b44302a6422405e5555ac9/r0010_server.pth"
  "./out/fl_bert_models_S81789_e1f33b9b63d64fdbb0d14f5d355ed89c/r0010_server.pth"
)

## Flower models of Full CBert ##
# models=(
#   ""
#   ""
#   ""
#   ""
#   ""
# )

## Fluke models of Kernel MLP no bert ##
# models=(
#   "./out/fl_models_S9046_6df8dd20ef664ab8b1fa08527f563e81/r0030_server.pth"
#   "./out/fl_models_S23517_20141fdf304642eebcdbb68f26a7b444/r0030_server.pth"
#   "./out/fl_models_S67895_3fea22fd59dd497791636098ff566320/r0030_server.pth"
#   "./out/fl_models_S47874_58b4fd8ff53d46639ef1c088771f078f/r0030_server.pth"
#   "./out/fl_models_S81789_684278c8abec4ab9baf043dea2abe4a9/r0030_server.pth"
# )

for i in {1..5}; do
  seed=${seeds[i]}
  echo -e "\033[0;32mSeed $seed\033[0m"
  model=${models[i]}
  dataset="./dataset/split_dataset_S$seed/df_test_pp.csv"
  name_wandb="clustering-$fl_framework-cbert-$seed--no-complex-iban"

  uv run clustering.py cbert-accounts-disambiguation $seed $model $dataset --name-wandb $name_wandb
done
