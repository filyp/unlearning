# %%
"""These functions are supposed to set the gradients, based on some texts."""
import os

from datasets import Dataset

os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import loss_fns

# %%
# after disruption_mask_avg: 39%
# after disruption_mask_each: 10%
# after common_core: 45%
# after common_core + disruption_mask_avg: 15%
# after common_core + disruption_mask_each: 3.5%

# after common_core with 2 batches: 60%
# after common_core with 2 batches + disruption_mask_avg: 22%


# %%
def normal(model, tokenizer, conf, forget_texts: Dataset):
    f_batch = tokenizer(forget_texts["text"], **conf.tokenizer)
    model.zero_grad(set_to_none=True)
    output = model(**f_batch)
    loss_fn = getattr(loss_fns, conf.unlearning_loss_fn)
    forget_loss = loss_fn(output, f_batch)
    forget_loss.backward()


def common_core(model, tokenizer, conf, forget_texts: Dataset):
    loss_fn = getattr(loss_fns, conf.unlearning_loss_fn)

    for i, forget_text in enumerate(forget_texts):
        model.zero_grad(set_to_none=True)
        f_batch = tokenizer(forget_text["text"], **conf.tokenizer)
        output = model(**f_batch)
        forget_loss = loss_fn(output, f_batch)
        forget_loss.backward()

        if i == 0:
            # ! first batch, so initialize acc
            for p in model.parameters():
                p.acc = p.grad
            continue

        # ! update acc
        for p in model.parameters():
            if not p.requires_grad:
                continue
            mask = p.acc.sign() == p.grad.sign()
            p.acc *= mask
            p.grad *= mask

            p.acc += p.grad
    
    # ! set grads
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p.grad = p.acc / len(forget_texts)
        del p.acc


def disruption_mask_avg(model, tokenizer, conf, retain_texts: Dataset):
    r_batch = tokenizer(retain_texts["text"], **conf.tokenizer)
    # ! backup grads
    for p in model.parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None
        p.unlearning_grad = p.grad
        del p.grad

    # ! retain pass
    model.zero_grad(set_to_none=True)
    output = model(**r_batch)
    retain_loss = loss_fns.cross_entropy(output, r_batch)
    retain_loss.backward()

    # ! apply mask
    for p in model.parameters():
        if not p.requires_grad:
            continue
        mask = p.unlearning_grad.sign() == p.grad.sign()
        p.grad = p.unlearning_grad * mask
        del p.unlearning_grad


def disruption_mask_each(model, tokenizer, conf, retain_texts: Dataset):
    # ! backup grads
    for p in model.parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None
        p.unlearning_grad = p.grad
        del p.grad

    for retain_text in retain_texts:
        # ! retain pass
        model.zero_grad(set_to_none=True)
        r_batch = tokenizer(retain_text["text"], **conf.tokenizer)
        output = model(**r_batch)
        retain_loss = loss_fns.cross_entropy(output, r_batch)
        retain_loss.backward()

        for p in model.parameters():
            if not p.requires_grad:
                continue
            mask = p.unlearning_grad.sign() == p.grad.sign()
            p.unlearning_grad *= mask

    # ! revert backup
    for p in model.parameters():
        if not p.requires_grad:
            continue
        assert p.unlearning_grad is not None
        p.grad = p.unlearning_grad
        del p.unlearning_grad


def normalize_grads(model, target_grad_norm=1):
    for p in model.parameters():
        if not p.requires_grad:
            continue
        p.grad *= target_grad_norm / p.grad.norm()


# # ! limit percentiles
# if h.percentile is not None:
#     abs_vals = p.grad.flatten().abs()
#     k = int(len(abs_vals) * h.percentile)
#     cutoff = abs_vals.kthvalue(k).values.item()
#     mask = p.grad.abs() > cutoff
#     p.grad *= mask


# # %%

# import json
# import logging
# import os
# from copy import deepcopy

# import hydra
# import torch as pt
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from utils import loss_fns
# from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
# from utils.evals import eval_on
# from utils.git_and_reproducibility import repo_root
# from utils.training import set_seeds

# pt.set_default_device("cuda")


# from omegaconf import OmegaConf
# conf = OmegaConf.load("../../configs/per_module_exp.yaml")
# # conf.model_id = "HuggingFaceTB/SmolLM-135M"

# model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda")
# # model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.float32, device_map="cuda")
# model.config.use_cache = False

# # ! limit which parameters are trained
# for n, p in model.named_parameters():
#     # p.requires_grad = any(pattern in n for pattern in ["layers.13.mlp.gate_proj"])
#     p.requires_grad = any(pattern in n for pattern in ["gate_proj"])

# # ! one step of unlearning
# optimizer = pt.optim.SGD(model.parameters(), lr=1e-2)
# model.train()

# # load corpora
# f_all = load_local("wmdp_deduped_correct_answers_corpus.jsonl")
# r_all = load_local("wmdp_deduped_wrong_answers_corpus.jsonl")

# # load questions
# wmdp_mcq = load_local(f"wmdp_deduped_bio/dev_T.jsonl")

# # load disrution eval set
# disruption_batches = load_batches(
#     load_fineweb_edu_corpus(), conf.model_id, batch_size=16
# )[: 16]

# tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
# tokenizer.pad_token = tokenizer.eos_token
# # we load the model to save some time on loading, copying orig_model instead


# q = wmdp_mcq[0]
# f_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
# r_corpus = r_all.filter(lambda ex: ex["original_question"] == q["question"])

# # %%
# model.zero_grad(set_to_none=True)
# # normal(model, tokenizer, conf, f_corpus)
# common_core(model, tokenizer, conf, f_corpus)

# # %%
# # disruption_mask_avg(model, tokenizer, conf, r_corpus)
# disruption_mask_each(model, tokenizer, conf, r_corpus)
# # %%
# acc_non_zero = 0
# acc_numel = 0
# for n, p in model.named_parameters():
#     if not p.requires_grad:
#         continue
#     acc_non_zero += pt.sum(p.grad != 0)
#     acc_numel += p.grad.numel()
#     print(p.grad.norm())
# # print(acc_non_zero / acc_numel)


# %%

# def unlearn(
#     h,
#     conf,
#     retain_batches,
#     forget_batches,
#     eval_callback,
# ):
#     h.fork_every_n_loops = int(h.fork_every_n_loops)
#     loss_fn = loss_fns[h.unlearning_loss_fn]
#     clip_at = h.clip_at if "clip_at" in h else 0

#     set_seeds(42)
#     model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.bfloat16)
#     model.config.use_cache = False

#     # todo maybe i should skip the embedding layer when unlearning?

#     for p in model.parameters():
#         p.retain_acc = pt.zeros_like(p.data)

#     # ! unlearning loop
#     if "rep_eng_retain_lr" in h:
#         frozen_model = deepcopy(model)
#         frozen_model.eval()

#     if h.train_adversary:
#         adversary = deepcopy(model)
#     else:
#         adversary = model

#     # calibrate normalization
#     f_batch = forget_batches[0]
#     model.zero_grad(set_to_none=True)
#     output = model(**f_batch)
#     loss = loss_fn(output, f_batch, clip_at)
#     loss.backward()
#     target_grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5

#     # ! unlearning loop
#     num_of_loops = int(len(forget_batches) * conf.unlearning_epochs)
#     for loop_num in range(num_of_loops):
#         batch_index = loop_num % len(forget_batches)
#         f_batch = forget_batches[batch_index]
#         r_batch = retain_batches[batch_index]
#         pt.cuda.empty_cache()
#         gc.collect()

#         if batch_index == 0:
#             eval_callback(model)

#         if loop_num % h.fork_every_n_loops == 0 and h.train_adversary:
#             # fork the adversary
#             adversary.load_state_dict(model.state_dict())
#             pt.cuda.empty_cache()
#             gc.collect()

#         model.train()
#         # ! retain pass
#         model.zero_grad(set_to_none=True)
#         adversary.zero_grad(set_to_none=True)
#         output = model(**r_batch)
#         retain_loss = cross_entropy_loss(output, r_batch)
#         if "rep_eng_retain_lr" in h:
#             # ! representation engineering retain loss
#             rep_eng_loss = circuit_breaker_retain_loss(
#                 model, r_batch, frozen_model, square_norm=h.square_norm
#             )
#             # note this loss is scaled both by this LR and retaining_rate
#             rep_eng_loss *= h.rep_eng_retain_lr
#             retain_loss += rep_eng_loss
#         retain_loss.backward()
#         for p in model.parameters():
#             # ! update disruption scores
#             p.retain_acc *= h.retain_momentum
#             p.retain_acc += p.grad * (1 - h.retain_momentum)
#             # ! retain update
#             p.data -= h.retaining_rate * p.retain_acc

#         # ! relearn the adversary
#         model.zero_grad(set_to_none=True)
#         adversary.zero_grad(set_to_none=True)
#         output = adversary(**f_batch)
#         if h.train_adversary:
#             adversary_loss = cross_entropy_loss(output, f_batch)
#             adversary_loss.backward(retain_graph=True)
#             for p in adversary.parameters():
#                 # apply adversary update
#                 p.data -= h.adv_lr * p.grad

#         # ! unlearning step with masking
#         # get unlearning grads loss from adversary
#         # reuse the computation graph from previous block
#         model.zero_grad(set_to_none=True)
#         adversary.zero_grad(set_to_none=True)
#         forget_loss = loss_fn(output, f_batch, clip_at)
#         forget_loss.backward()
#         grad_norm = sum(p.grad.norm() ** 2 for p in adversary.parameters()) ** 0.5
#         for p, ap in zip(model.parameters(), adversary.parameters()):
#             if h.use_masking:
#                 mask = p.retain_acc.sign() == ap.grad.sign()
#                 ap.grad *= mask

#             if h.normalize_grads:
#                 ap.grad *= target_grad_norm / grad_norm

#             p.data -= h.unlearning_rate * ap.grad

#     return model
