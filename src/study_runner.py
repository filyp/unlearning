# %%

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # necessary for determinism:

import gc
import logging
from copy import deepcopy

import hydra
import optuna
import torch as pt
from omegaconf import ListConfig, OmegaConf

import wandb
from datasets import load_dataset
from unlearning.unlearning import unlearn
from utils.data_loading import (
    data_paths,
    filter_by_question,
    load_batches,
    load_low_mi_set,
    load_retain_corpus,
)
from utils.evals import eval_on
from utils.git_and_reproducibility import commit_hash, get_storage, is_repo_clean
from utils.training import relearn, set_seeds


@hydra.main(version_base="1.2", config_path="../configs", config_name="wmdp_main")
def run_study(cfg):
    s = cfg.study_config

    storage = get_storage()
    set_seeds(42)
    pt.set_default_device("cuda")

    # split into hyperparams and other keys
    variant_name, custom = list(cfg.variants.items())[cfg.variant_num]
    hyperparam_ranges = OmegaConf.merge(cfg.hyperparams, custom)
    study_name = f"{cfg.name}|{variant_name}"
    print(f"{study_name=}")

    # todo also try pairwise forget retain batches on same question

    # load forget set
    _f_corpus = load_low_mi_set(data_paths[s.forget_set_name])
    _f_corpus = filter_by_question(_f_corpus, category=s.category, portion=s.portion)
    forget_batches = load_batches(_f_corpus, s.model_id, s.batch_size, s.max_length)

    # load retain set
    retain_corpus = load_retain_corpus(s.retain_set_name)
    retain_batches = load_batches(retain_corpus, s.model_id, s.batch_size, s.max_length)

    # load relearn set
    rel_corpus = load_low_mi_set(data_paths[cfg.relearn_config.set_name])
    rel_corpus = filter_by_question(rel_corpus, category=s.category, portion=s.portion)
    relearn_batches = load_batches(rel_corpus, s.model_id, s.batch_size, s.max_length)

    # load unlearning eval set
    wmdp_set = load_low_mi_set(data_paths["wmdp_deduped_mcq_eval"])
    wmdp_set = filter_by_question(wmdp_set, category=s.category, portion=s.portion)
    mmlu_set = load_dataset("cais/mmlu", "all", split="validation")
    mmlu_set = mmlu_set.shuffle(seed=42).select(range(300))  # todo revert back later

    def _eval_callback(model):
        model.eval()
        # eval mmlu and wmdp
        with pt.no_grad():
            mmlu_acc = eval_on(mmlu_set, model, temperature=s.eval_temperature)
            wmdp_acc = eval_on(wmdp_set, model, temperature=s.eval_temperature)
        logging.info(f"{mmlu_acc=:.4f}  {wmdp_acc=:.4f}")
        wandb.log({"mmlu_acc": mmlu_acc, "wmdp_acc": wmdp_acc})
        return mmlu_acc, wmdp_acc

    def objective(trial):
        # construct hyperparams
        hyperparams = deepcopy(hyperparam_ranges)
        for hp_name, distribution in hyperparams.items():
            if isinstance(distribution, ListConfig):
                low, high, log = distribution
                hyperparams[hp_name] = trial.suggest_float(hp_name, low, high, log=log)
        logging.info(f"trial {trial.number} - {trial.params}")

        wandb.init(project="wmdp2", name=f"{trial.number}", group=study_name)
        set_seeds(42)

        print("unlearning...")
        model = unlearn(
            hyperparams,
            s,
            retain_batches,
            forget_batches,
            _eval_callback,
        )
        pt.cuda.empty_cache()
        gc.collect()

        mmlu_pre = eval_on(mmlu_set, model, temperature=s.eval_temperature)

        print("relearning...")
        model = relearn(model, relearn_batches, cfg.relearn_config, _eval_callback)

        mmlu_post, wmdp_post = _eval_callback(model)
        wandb.finish()
        return mmlu_pre, wmdp_post

    if cfg.extend_existing_study:
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            directions=["maximize", "minimize"],
        )
        study.set_metric_names(["mmlu_accuracy", "wmdp_accuracy"])
    study.set_user_attr("commit_hash", commit_hash())
    study.set_user_attr("is_repo_clean", is_repo_clean())

    # run remaining trials
    n_trials = s.n_trials - len(study.trials)
    logging.info(f"Trials existing: {len(study.trials)}, remaining: {n_trials}")
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    run_study()
