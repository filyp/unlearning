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

# Running

The main runner for running various methods is [`src/main_runner.py`](src/main_runner.py).
Run it as:

```bash
python src/main_runner.py --config-name=CONFIG_NAME --exp-num=NUM
```

[`example_job.sh`](example_job.sh) provides an example compute configuration if using `sbatch`.

## WMDP-Bio

To reproduce results from the paper for WMDP-Bio, run:

```bash
python src/main_runner.py --config-name=main_comparison_llama_bio --exp-num=NUM
```
With `NUM` as 0-4 (5 CIR runs), 20-24 (5 circuit breakers runs), 25-30 (5 gradient difference runs + one repeated with more epochs).

## WMDP-Cyber

```bash
python src/main_runner.py --config-name=main_comparison_llama_cyber --exp-num=NUM
```

With `NUM` as 0-4 (5 CIR runs), 35-39 (5 circuit breakers runs), 40-44 (5 gradient difference runs)

## Other

In the [`configs`](configs) directory you can find configs for other plots from the appendix.

# CIR code

[`src/main_runner.py`](src/main_runner.py) is the entry point for running everything, so it's quite complex. If you just want to use CIR, best to look in [`src/utils/common_cir.py`](src/utils/common_cir.py) for the implementation of CIR's crucial operations, and then in the `main_runner.py` to see how they are used.