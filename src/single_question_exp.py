# %%
import json
import logging
import os
from copy import deepcopy

import hydra
import torch as pt
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import loss_fns, masking
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root
from utils.loss_fns import print_per_token_colored_loss
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")


# @hydra.main(config_path="../configs", config_name="single_question_exp.yaml")
# def main(conf: dict):
conf = OmegaConf.load("../configs/single_question_exp.yaml")

# load corpora
f_all = load_local("wmdp_deduped_correct_answers_corpus.jsonl")
r_all = load_local("wmdp_deduped_wrong_answers_corpus.jsonl")

# load questions
wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")
# do not unlearn questions where the model already does not know
# use accuracies of llama 1B even for other models, to have the same questions
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

# load disrution eval set
disruption_batches = load_batches(
    load_fineweb_edu_corpus(), conf.model_id, batch_size=16
)[: conf.num_disruption_batches]

tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token

# %% choose question
q = wmdp_mcq[4]

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


# initial eval
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
orig_wmdp_acc, orig_disr_loss = _eval(model)
logging.info(f"wmdp={orig_wmdp_acc:8.4f}    disr={orig_disr_loss:8.4f}    orig model")
del model

q

# %%
conf.unlearning_steps = 200
conf.unlearning_rate = 1e-3
conf.unlearning_loss_fn = "neg_cross_entropy"
# conf.unlearning_loss_fn = "neg_entropy"
# conf.unlearning_loss_fn = "correct_logit_minus_avg"
# conf.unlearning_loss_fn = "correct_logit"
# conf.adv_lr = 1e-3
# conf.retain_lr = 1e-3
for variant in [
    # fmt: off
    # dict(unlearning_method="normal", masking_method=None),
    # dict(unlearning_method="normal", masking_method="disruption_mask_avg"),
    dict(unlearning_method="normal", masking_method="disruption_mask_each"),
    # dict(unlearning_method="common_core", masking_method=None),
    # dict(unlearning_method="common_core", masking_method="disruption_mask_avg"),
    # dict(unlearning_method="common_core", masking_method="disruption_mask_each"),
    # fmt: on
]:
    conf.unlearning_method = variant["unlearning_method"]
    conf.masking_method = variant["masking_method"]

    # ! setup
    set_seeds(42)
    model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    wandb.init(
        project="unlearning-wmdp2",
        name=f"{conf.unlearning_method}-{conf.masking_method}-{conf.unlearning_loss_fn}",
        group="training_algos",
        config=OmegaConf.to_container(conf),
    )
    wandb.log({"wmdp_acc": orig_wmdp_acc, "disr_loss": orig_disr_loss})

    # init_log = print_per_token_colored_loss(
    #     model, tokenizer, conf, f_corpus[0], max_log=20
    # )

    # ! limit which parameters are trained
    for n, p in model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in conf.target_modules)

    optimizer = pt.optim.SGD(model.parameters(), lr=conf.unlearning_rate)



    # ! one step of unlearning
    for i in range(conf.unlearning_steps):
        model.train()
        
        # if i % 10 == 0:
        #     adversary = deepcopy(model)
        # # ! train adversary
        # f_batch = tokenizer(f_corpus["text"], **conf.tokenizer)
        # adversary.zero_grad(set_to_none=True)
        # output = adversary(**f_batch)
        # forget_loss = loss_fns.cross_entropy(output, f_batch)
        # forget_loss.backward()
        # for ap in adversary.parameters():
        #     if ap.grad is not None:
        #         ap.data -= ap.grad * conf.adv_lr
        # adversary.zero_grad(set_to_none=True)


        unlearning_method = getattr(masking, conf.unlearning_method)
        unlearning_method(model, tokenizer, conf, f_corpus)
        
        if conf.masking_method is not None:
            masking_method = getattr(masking, conf.masking_method)
            masking_method(model, tokenizer, conf, r_corpus)

        # for p, ap in zip(model.parameters(), adversary.parameters()):
        #     p.grad = ap.grad
        #     del ap.grad

        optimizer.step()


        # r_batch = tokenizer(r_corpus["text"], **conf.tokenizer)
        # model.zero_grad(set_to_none=True)
        # output = model(**r_batch)
        # retain_loss = loss_fns.cross_entropy(output, r_batch)
        # retain_loss.backward()
        # for p in model.parameters():
        #     if p.grad is not None:
        #         p.data -= p.grad * conf.retain_lr
        #     del p.grad






        if i % 10 == 0:
            # ! evaluate
            wmdp_acc, disr_loss = _eval(model)
            logging.info(f"wmdp={wmdp_acc:8.4f}    disr={disr_loss:8.4f}    {variant}")
            if disr_loss - orig_disr_loss > 0.01:
                break
            wandb.log({"wmdp_acc": wmdp_acc, "disr_loss": disr_loss})

        # print_per_token_colored_loss(
        #     model, tokenizer, conf, f_corpus[0], max_log=20, reference=init_log
        # )

    # ! clean up
    del model
    pt.cuda.empty_cache()
    wandb.finish()
