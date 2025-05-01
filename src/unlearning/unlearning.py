import gc
import logging
from copy import deepcopy

import torch as pt
from transformers import AutoModelForCausalLM

from utils.loss_fns import circuit_breaker_retain_loss, cross_entropy_loss, loss_fns
from utils.training import set_seeds


def unlearn(
    h,
    conf,
    retain_batches,
    forget_batches,
    eval_callback,
):
    h.fork_every_n_loops = int(h.fork_every_n_loops)
    loss_fn = loss_fns[h.unlearning_loss_fn]
    clip_at = h.clip_at if "clip_at" in h else 0

    set_seeds(42)
    model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.bfloat16)
    model.config.use_cache = False

    # todo maybe i should skip the embedding layer when unlearning?

    for p in model.parameters():
        p.retain_acc = pt.zeros_like(p.data)

    # ! unlearning loop
    if "rep_eng_retain_lr" in h:
        frozen_model = deepcopy(model)
        frozen_model.eval()
    
    if h.train_adversary:
        adversary = deepcopy(model)
    else:
        adversary = model
    
    # calibrate normalization
    f_batch = forget_batches[0]
    model.zero_grad(set_to_none=True)
    output = model(**f_batch)
    loss = loss_fn(output, f_batch, clip_at)
    loss.backward()
    target_grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters()) ** 0.5

    # ! unlearning loop
    num_of_loops = int(len(forget_batches) * conf.unlearning_epochs)
    for loop_num in range(num_of_loops):
        batch_index = loop_num % len(forget_batches)
        f_batch = forget_batches[batch_index]
        r_batch = retain_batches[batch_index]
        pt.cuda.empty_cache()
        gc.collect()
        
        if batch_index == 0:
            eval_callback(model)

        if loop_num % h.fork_every_n_loops == 0 and h.train_adversary:
            # fork the adversary
            adversary.load_state_dict(model.state_dict())
            pt.cuda.empty_cache()
            gc.collect()

        model.train()
        # ! retain pass
        model.zero_grad(set_to_none=True)
        adversary.zero_grad(set_to_none=True)
        output = model(**r_batch)
        retain_loss = cross_entropy_loss(output, r_batch)
        if "rep_eng_retain_lr" in h:
            # ! representation engineering retain loss
            rep_eng_loss = circuit_breaker_retain_loss(
                model, r_batch, frozen_model, square_norm=h.square_norm
            )
            # note this loss is scaled both by this LR and retaining_rate
            rep_eng_loss *= h.rep_eng_retain_lr
            retain_loss += rep_eng_loss
        retain_loss.backward()
        for p in model.parameters():
            # ! update disruption scores
            p.retain_acc *= h.retain_momentum
            p.retain_acc += p.grad * (1 - h.retain_momentum)
            # ! retain update
            p.data -= h.retaining_rate * p.retain_acc

        # ! relearn the adversary
        model.zero_grad(set_to_none=True)
        adversary.zero_grad(set_to_none=True)
        output = adversary(**f_batch)
        if h.train_adversary:
            adversary_loss = cross_entropy_loss(output, f_batch)
            adversary_loss.backward(retain_graph=True)
            for p in adversary.parameters():
                # apply adversary update
                p.data -= h.adv_lr * p.grad

        # ! unlearning step with masking
        # get unlearning grads loss from adversary
        # reuse the computation graph from previous block
        model.zero_grad(set_to_none=True)
        adversary.zero_grad(set_to_none=True)
        forget_loss = loss_fn(output, f_batch, clip_at)
        forget_loss.backward()
        grad_norm = sum(p.grad.norm() ** 2 for p in adversary.parameters()) ** 0.5
        for p, ap in zip(model.parameters(), adversary.parameters()):
            if h.use_masking:
                mask = p.retain_acc.sign() == ap.grad.sign()
                ap.grad *= mask

            if h.normalize_grads:
                ap.grad *= target_grad_norm / grad_norm

            p.data -= h.unlearning_rate * ap.grad

    return model
