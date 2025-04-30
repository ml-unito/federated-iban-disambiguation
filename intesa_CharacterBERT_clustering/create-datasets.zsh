#!/bin/zsh

uv run kernel-classify.py create-dataset 9046 --no-use-bert
uv run kernel-classify.py create-dataset 23517 --no-use-bert
uv run kernel-classify.py create-dataset 30921 --no-use-bert
uv run kernel-classify.py create-dataset 47874 --no-use-bert
uv run kernel-classify.py create-dataset 81789 --no-use-bert

uv run kernel-classify.py create-dataset 9046 --use-bert
uv run kernel-classify.py create-dataset 23517 --use-bert
uv run kernel-classify.py create-dataset 30921 --use-bert
uv run kernel-classify.py create-dataset 47874 --use-bert
uv run kernel-classify.py create-dataset 81789 --use-bert