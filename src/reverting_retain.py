# %%
import json
import logging
import os
from copy import deepcopy
import random

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


# @hydra.main(config_path="../configs", config_name="reverting_retain.yaml")
# def main(conf: dict):
conf = OmegaConf.load("../configs/reverting_retain.yaml")

# load corpora
# f_all = load_local("wmdp_deduped_correct_answers_corpus.jsonl")
# r_all = load_local("wmdp_deduped_wrong_answers_corpus.jsonl")
f_all = load_local("my_generation/wmdp_bio.jsonl")

# load questions
wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")
# do not unlearn questions where the model already does not know
# use accuracies of llama 1B even for other models, to have the same questions
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

# load disrution eval set
_fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=8)
disruption_batches = _fineweb_batches[: conf.num_disruption_batches]
additional_retain_batches = _fineweb_batches[conf.num_disruption_batches :]

tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def model_dist(model, orig_model):
    dist = 0
    for p, op in zip(trainable_params(model), trainable_params(orig_model)):
        dist += pt.sum((p.data - op.data) ** 2)
    return float(pt.sqrt(dist))


# %% choose question
num_ex = 15

# ! load texts
f_corpora_per_question = []
for q in wmdp_mcq:
    f_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
    f_corpus = f_corpus.map(lambda ex: dict(text=f"{ex['beginning']} {ex['ending']}"))
    f_corpus = f_corpus.select(range(num_ex))
    f_corpora_per_question.append(f_corpus)


# %%
def _eval(model):
    model.eval()
    with pt.no_grad():
        # wmdp_acc = eval_on(Dataset.from_list([q]), model, temperature=1)
        # wmdp_acc = eval_on(wmdp_mcq.select([19]), model, temperature=1)
        wmdp_acc = eval_on(wmdp_mcq, model, temperature=1)
        disr_loss = pt.mean(
            pt.Tensor([
                loss_fns.cross_entropy(model(**d_batch), d_batch)
                for d_batch in disruption_batches
            ])
        )
    return float(wmdp_acc), float(disr_loss)


# initial eval
orig_model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
orig_wmdp_acc, orig_disr_loss = _eval(orig_model)
logging.info(f"wmdp={orig_wmdp_acc:8.4f}    disr={orig_disr_loss:8.4f}    orig model")

# %%

# ! compute original outuputs, for reference
last_states_per_question = []
for f_corpus in f_corpora_per_question:
    orig_model.zero_grad(set_to_none=True)
    batch = tokenizer(f_corpus["text"], **conf.tokenizer)
    # storing last_state is 1MB per batch, while storing logits is 50MB (for Llama-3.2-1B)
    # but if we calculate it only once per question dive, maybe just don't store anything
    with pt.no_grad():
        output = orig_model(**batch, output_hidden_states=True)
    last_state = output.hidden_states[-1].detach().clone()
    last_states_per_question.append(last_state)

# %%
def retain_pass(model, orig_model, batch, target_logits):
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    loss = loss_fns.non_target_disruption(output, batch, target_logits)
    loss.backward()
    for p, op in zip(trainable_params(model), trainable_params(orig_model)):
        if conf.retain_only_in_reverting_direction:
            mask = (op.data - p.data).sign() != p.grad.sign()
            p.grad *= mask
        p.data -= p.grad * conf.retaining_rate
        del p.grad
# %%

for variant in [
    # fmt: off
    # dict(name="normal", unlearning_method="normal"),
    # dict(name="only-answer", unlearning_method="only_answer_tokens"),
    # dict(name="only-answer-ans-masked", unlearning_method="only_answer_tokens", masking_method="mask_out_answer_without_context"),
    # dict(name="dynamic", precompute_unlearning_grads=False),
    # dict(name="no-retain", retaining_rate=0),
    # dict(name="common-core", unlearning_method="common_core"),
    # dict(name="non-reverting-retain", retain_only_in_reverting_direction=False, unlearning_rate=3e-2, retaining_rate=1e-2),
    dict(name="non-reverting-retain-decay99", retain_only_in_reverting_direction=False, unlearning_rate=3e-2, retaining_rate=1e-2, weight_decay=0.99),
    # fmt: on
]:
    conf = OmegaConf.load("../configs/reverting_retain.yaml")
    conf.update(variant)

    # ! setup
    set_seeds(42)
    model = AutoModelForCausalLM.from_pretrained(
        conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
    )
    model.config.use_cache = False
    wandb.init(
        project="unlearning-wmdp2",
        name=f"{conf.unlearning_rate}-{conf.retaining_rate}-{num_ex}ex-{conf.name}",
        group=f"reverting-retain-all-questions",
        config=OmegaConf.to_container(conf),
    )
    wandb.log({"wmdp_acc": orig_wmdp_acc, "disr_loss": orig_disr_loss})

    # ! limit which parameters are trained
    for n, p in model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in conf.target_modules)
    for n, p in orig_model.named_parameters():
        p.requires_grad = any(pattern in n for pattern in conf.target_modules)

    unlearning_method = getattr(masking, conf.unlearning_method)
    if conf.masking_method is not None:
        masking_method = getattr(masking, conf.masking_method)

    # ! unlearning
    for epoch in range(conf.num_epochs):
        for q_index, q in enumerate(wmdp_mcq):
            model.train()
            f_corpus = f_corpora_per_question[q_index]

            # ! unlearn pass
            if conf.precompute_unlearning_grads:
                unlearning_method(orig_model, tokenizer, conf, f_corpus)
                if conf.masking_method is not None:
                    masking_method(orig_model, tokenizer, conf, f_corpus)
                for p, op in zip(trainable_params(model), trainable_params(orig_model)):
                    p.grad = op.grad.clone()
                    del op.grad
            else:
                unlearning_method(model, tokenizer, conf, f_corpus)
                if conf.masking_method is not None:
                    masking_method(model, tokenizer, conf, f_corpus)

            # ! normalize unlearning grads and apply them
            grad_norm = sum(p.grad.norm() ** 2 for p in trainable_params(model)) ** 0.5
            for p in trainable_params(model):
                p.grad /= grad_norm
                p.data -= p.grad * conf.unlearning_rate
                del p.grad

            # decay into orig model
            for p, op in zip(trainable_params(model), trainable_params(orig_model)):
                p.data = p.data * conf.weight_decay + op.data * (1 - conf.weight_decay)

            # ! retain pass
            last_state = last_states_per_question[q_index]
            batch = tokenizer(f_corpus["text"], **conf.tokenizer)
            target_logits = pt.einsum(
                "bph,lh->bpl", last_state, orig_model.model.embed_tokens.weight
            )
            retain_pass(model, orig_model, batch, target_logits)
            pt.cuda.empty_cache()

            # # ! additional retain pass
            # a_batch = random.choice(additional_retain_batches)
            # with pt.no_grad():
            #     output = orig_model(**a_batch)
            # retain_pass(model, orig_model, a_batch, output.logits)
            # del output, a_batch
            # pt.cuda.empty_cache()

            logging.info(
                f"epoch={epoch:4d}   q_index={q_index:4d}   dist={model_dist(model, orig_model):8.4f}   grad_norm={grad_norm:8.4f}"
            )

        # ! evaluate
        wmdp_acc, disr_loss = _eval(model)
        logging.info(f"wmdp={wmdp_acc:8.4f}    disr={disr_loss:8.4f}")
        if disr_loss - orig_disr_loss > 0.1:
            break
        wandb.log({"wmdp_acc": wmdp_acc, "disr_loss": disr_loss})

    wandb.finish()

    # ! clean up
    del model
    pt.cuda.empty_cache()
