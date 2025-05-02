# %%
# %load_ext autoreload
# %autoreload 2
import json
import logging
import os
from copy import deepcopy

import hydra
import torch as pt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import loss_fns, masking
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")


@hydra.main(config_path="../configs", config_name="per_module_exp.yaml")
def main(conf: dict):
    # load corpora
    f_all = load_local("wmdp_deduped_correct_answers_corpus.jsonl")
    r_all = load_local("wmdp_deduped_wrong_answers_corpus.jsonl")

    # load questions
    wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")

    # load disrution eval set
    disruption_batches = load_batches(
        load_fineweb_edu_corpus(), conf.model_id, batch_size=16
    )[: conf.num_disruption_batches]

    tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # we load the model to save some time on loading, copying orig_model instead
    orig_model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
    )

    for q in wmdp_mcq:
        if q["Llama-3.2-1B"] < 0.25:
            # do not unlearn questions where the model already does not know
            # use accuracies of llama 1B even for other models, to have the same questions
            continue
        logging.info("\n\n" + q["question"])

        # ! load texts
        f_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
        if conf.use_related_retain:
            r_corpus = r_all.filter(lambda ex: ex["original_question"] == q["question"])
        else:
            r_corpus = r_all.shuffle(seed=42).select(range(3))

        def _eval(model):
            model.eval()
            with pt.no_grad():
                wmdp_acc = eval_on(Dataset.from_list([q]), model, temperature=1)
                disr_loss = pt.mean(
                    pt.Tensor([
                        loss_fns.cross_entropy(model(**d_batch), d_batch)
                        for d_batch in disruption_batches
                    ])
                )
            return float(wmdp_acc), float(disr_loss)

        orig_wmdp_acc, orig_disr_loss = _eval(orig_model)
        logging.info(
            f"wmdp={orig_wmdp_acc:8.4f}    disr={orig_disr_loss:8.4f}    orig model"
        )

        wmdp_accs = {}
        disr_losses = {}
        for param_name, _ in orig_model.named_parameters():
            if ("layers." not in param_name) or ("layernorm" in param_name):
                continue

            # ! setup
            set_seeds(42)
            model = deepcopy(orig_model)
            model.config.use_cache = False

            # ! limit which parameters are trained
            for n, p in model.named_parameters():
                p.requires_grad = any(pattern in n for pattern in [param_name])

            # ! one step of unlearning
            optimizer = pt.optim.SGD(model.parameters(), lr=conf.unlearning_rate)
            model.train()

            unlearning_method = getattr(masking, conf.unlearning_method)
            unlearning_method(model, tokenizer, conf, f_corpus)

            masking_method = getattr(masking, conf.masking_method)
            if masking_method is not None:
                masking_method(model, tokenizer, conf, r_corpus)

            optimizer.step()

            # ! evaluate
            wmdp_accs[param_name], disr_losses[param_name] = _eval(model)
            logging.info(
                f"wmdp={wmdp_accs[param_name]:8.4f}    disr={disr_losses[param_name]:8.4f}    {param_name}"
            )

            # ! clean up
            del model
            pt.cuda.empty_cache()

            # ! save
            dir_ = (
                repo_root()
                / "data"
                / "per_module_exp"
                / f"{conf.masking}_{conf.use_related_retain}"
            )
            os.makedirs(dir_, exist_ok=True)
            path = dir_ / f"{q['question'][:100]}.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "wmdp_accs": wmdp_accs,
                        "disr_losses": disr_losses,
                        "orig_wmdp_acc": orig_wmdp_acc,
                        "orig_disr_loss": orig_disr_loss,
                        "mcq": q,
                    },
                    f,
                )


if __name__ == "__main__":
    main()
