
# ./launchexp.zsh 23517 exp_pretrained alg_pretrained
# ./launchexp.zsh 47874 exp_pretrained alg_pretrained
# ./launchexp.zsh 9046 exp_pretrained alg_pretrained
# ./launchexp.zsh 30921 exp_pretrained alg_pretrained
# ./launchexp.zsh 81789 exp_pretrained alg_pretrained

uv run kernel-classify.py nn-classify 9046
uv run kernel-classify.py nn-classify 23517
uv run kernel-classify.py nn-classify 30921
uv run kernel-classify.py nn-classify 47874
uv run kernel-classify.py nn-classify 81789

uv run kernel-classify.py nn-classify 9046 --bert
uv run kernel-classify.py nn-classify 23517 --bert
uv run kernel-classify.py nn-classify 30921 --bert
uv run kernel-classify.py nn-classify 47874 --bert
uv run kernel-classify.py nn-classify 81789 --bert
