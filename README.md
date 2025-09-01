# Installation

Clone the repo and make a venv, then:
```
pip install -e .
wandb login
huggingface-cli login
```

(In the unlikely case of dependency issues, try this in a fresh venv, to use the exact package versions:)
```bash
pip install -r requirements.txt
pip install -e . --no-deps
```