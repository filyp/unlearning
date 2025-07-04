# %%
import json
import logging
import os
import random
from copy import deepcopy

import hydra
import torch as pt
import torch.nn.functional as F
from datasets import Dataset, load_dataset
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


conf = OmegaConf.load("../configs/input_specific_unlearning.yaml")

# load corpora
f_all = load_local("my_generation/wmdp_bio.jsonl")

# load questions
wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")
# do not unlearn questions where the model already does not know
# use accuracies of llama 1B even for other models, to have the same questions
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

wmdp_mcq_disr = load_local(f"wmdp_deduped_{conf.category}/{conf.split_disr}.jsonl")
wmdp_mcq_disr = wmdp_mcq_disr.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

# load disrution eval set
_fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=8)
disruption_batches = _fineweb_batches[: conf.num_disruption_batches]
additional_retain_batches = _fineweb_batches[conf.num_disruption_batches :]

tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


# %% choose question
num_ex = 1

# ! load texts
f_corpora_per_question = []
for q in wmdp_mcq:
    f_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
    f_corpus = f_corpus.map(lambda ex: dict(text=f"{ex['beginning']} {ex['ending']}"))
    f_corpus = f_corpus.select(range(num_ex))
    f_corpora_per_question.append(f_corpus)
r_corpora_per_question = []
for q in wmdp_mcq_disr:
    r_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
    r_corpus = r_corpus.map(lambda ex: dict(text=f"{ex['beginning']} {ex['ending']}"))
    r_corpus = r_corpus.select(range(num_ex))
    r_corpora_per_question.append(r_corpus)


# %%
# alternative_questions = """\
# The protein responsible for mediating attachment of Species B rotavirus to host cell receptors is the
# The protein responsible for mediating attachment of Species A rotavirus to bacterial cell receptors is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell cytoplasm is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell nucleus is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell mitochondria is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell ribosomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell lysosomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell Golgi is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell endoplasmic reticulum is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell peroxisomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell vacuoles is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell vesicles is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell microtubules is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell microfilaments is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell intermediate filaments is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell centrosomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell centrioles is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell cilia is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell flagella is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell tight junctions is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell gap junctions is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell desmosomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell hemidesmosomes is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell focal adhesions is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell adherens junctions is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell plasmodesmata is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell synapses is the
# The protein responsible for mediating attachment of Species A rotavirus to host cell extracellular matrix is the
# """
alternative_questions = """\
The viral protein that mediates rotavirus replication in the host cell nucleus is the
The structural component that enables rotavirus to penetrate the host cell's nuclear membrane is the
The viral factor that facilitates rotavirus integration into the host cell's genome is the
The molecular structure that allows rotavirus to bind to host cell DNA is the
The viral component that mediates rotavirus entry into the host cell's mitochondria is the
The protein domain that enables rotavirus to interact with host cell ribosomes is the
The viral structure that facilitates rotavirus binding to host cell lysosomes is the
The molecular component that mediates rotavirus attachment to host cell Golgi apparatus is the
The viral protein that enables rotavirus to bind to host cell endoplasmic reticulum is the
The structural element that facilitates rotavirus interaction with host cell peroxisomes is the
The viral protein that mediates rotavirus binding to host cell tight junctions is the
The structural component that enables rotavirus to interact with host cell gap junctions is the
The viral factor that facilitates rotavirus attachment to host cell desmosomes is the
The molecular structure that allows rotavirus to bind to host cell hemidesmosomes is the
The viral component that mediates rotavirus interaction with host cell focal adhesions is the
The protein domain that enables rotavirus to bind to host cell adherens junctions is the
The viral structure that facilitates rotavirus attachment to host cell plasmodesmata is the
The molecular component that mediates rotavirus binding to host cell synapses is the
The viral protein that enables rotavirus to interact with host cell extracellular matrix is the
The structural element that facilitates rotavirus binding to host cell centrosomes is the
The viral component that mediates rotavirus attachment to host cell centrioles is the
The protein domain that enables rotavirus to bind to host cell cilia is the
The viral structure that facilitates rotavirus interaction with host cell flagella is the
The molecular component that mediates rotavirus binding to host cell microtubules is the
The viral protein that enables rotavirus to attach to host cell microfilaments is the
The structural element that facilitates rotavirus binding to host cell intermediate filaments is the
The viral component that mediates rotavirus interaction with host cell nuclear pores is the
The protein domain that enables rotavirus to bind to host cell nuclear lamina is the
The viral structure that facilitates rotavirus attachment to host cell chromatin is the
The molecular component that mediates rotavirus binding to host cell nucleolus is the
"""

alternative_questions = alternative_questions.strip().split("\n")
# %%

conf = OmegaConf.load("../configs/input_specific_unlearning.yaml")
# conf.update(variant)
conf.name = "onlyans-isu-with-context"
conf.unlearning_rate = 5e-2
dist_thresh = 0.3

# ! setup
set_seeds(42)
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False
wandb.init(
    project="unlearning-wmdp3",
    name=f"{conf.unlearning_rate}-{dist_thresh}-{conf.name}",
    group=f"ISU-test",
    config=OmegaConf.to_container(conf),
)

# ! limit which parameters are trained
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

# ! calc unlearning
model.train()
q_index = 1
q = wmdp_mcq[q_index]
f_corpus = f_corpora_per_question[q_index]

# %% calculate momentums over alternative questions
for p in trainable_params(model):
    p.m1 = pt.zeros_like(p).to(pt.float32)
    p.m2 = pt.zeros_like(p).to(pt.float32)

for alt_beg in alternative_questions:
    ex = deepcopy(f_corpus[0])
    ex["beginning"] = alt_beg
    ex["text"] = ex["beginning"] + " " + ex["ending"]
    f_corpus_mod = Dataset.from_list([ex])

    masking.only_answer_tokens(model, tokenizer, conf, f_corpus_mod)
    for p in trainable_params(model):
        p.m1 += p.grad
        p.m2 += p.grad ** 2

# %% ! calculate mean and std
n = len(alternative_questions)
for p in trainable_params(model):
    p.std = (p.m2 / n - (p.m1 / n) ** 2) ** 0.5
    p.mean = p.m1 / n
    del p.m1
    del p.m2


# %% ! unlearning grads
masking.only_answer_tokens(model, tokenizer, conf, f_corpus)
for p in trainable_params(model):
    p.unlearning_grad_acc = p.grad
    del p.grad

# %% ! calculate distance from mean and std
for p in trainable_params(model):
    p.dist = (p.unlearning_grad_acc - p.mean).abs() / p.std
    del p.mean
    del p.std
    # fill nan with +inf
    p.dist = pt.nan_to_num(p.dist, nan=float("inf"))

# %% ! only take outliers
for p in trainable_params(model):
    mask_out = p.dist < dist_thresh
    p.unlearning_grad_acc[mask_out] = 0

# %% ! normalize grads and record
grad_norm = (
    sum(p.unlearning_grad_acc.norm() ** 2 for p in trainable_params(model)) ** 0.5
)
logging.info(f"q_index={q_index:4d}   grad_norm={grad_norm:8.4f}")
for p in trainable_params(model):
    p.unlearning_grad_acc += p.unlearning_grad_acc / grad_norm

# %%
# ! unlearning
for epoch in range(4):
    # ! evaluate
    model.eval()
    with pt.no_grad():
        # wmdp_acc = eval_on(wmdp_mcq, model, temperature=1)
        wmdp_acc = eval_on(wmdp_mcq.select(range(1)), model, temperature=1)
        wmdp_acc_disr = eval_on(wmdp_mcq_disr, model, temperature=1)

    logging.info(
        f"epoch={epoch:4d}   wmdp={wmdp_acc:8.4f}    wmdp_disr={wmdp_acc_disr:8.4f}"
    )
    wandb.log({"wmdp_acc": wmdp_acc, "wmdp_disr_acc": wmdp_acc_disr})

    # ! apply unlearning
    for p in trainable_params(model):
        p.data -= p.unlearning_grad_acc * conf.unlearning_rate

wandb.finish()

# ! clean up
del model
pt.cuda.empty_cache()

# %%
