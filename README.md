- src/0_grad_similarity.py - for Paris experiments
- src/0_transferability_per_token.py - token position x layer
- src/1_isu2_pre_mlp_activation.py - statically look at transfer, without training, example x layer
- src/2_proj_training.py - main training loop
- src/generation.py - generate pair corpus from ABCD examples

The `.requirements_exact.txt` is tested with python3.13.

# Installation
```
python -m pip install -r requirements.txt
wandb login
huggingface-cli login
```