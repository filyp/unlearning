# %%
import logging
import time
from contextlib import contextmanager

import hydra
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.data_loading import *
from utils.evals import eval_on
from utils.loss_fns import cross_entropy
from utils.training import get_update_norm, scale_grads_, set_seeds, trainable_modules

logging.basicConfig(level=logging.INFO)

with hydra.initialize(config_path="../configs", version_base="1.2"):
    cfg = hydra.compose(config_name="CIR_simple")


@contextmanager
def trim_layers(model, max_layer):
    """Temporarily tell the model to use only the first max_layer layers."""
    all_layers = model.model.layers
    model.model.layers = model.model.layers[:max_layer]
    try:
        yield
    finally:
        model.model.layers = all_layers


# ! setup
set_seeds(42)

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load data
wikitext = load_local("wikitext_16k.jsonl")
wikitext_batches = [
    tokenizer(x["text"], **cfg.data.tokenizer)
    for x in wikitext.shuffle(seed=42).batch(cfg.data.wikitext_batch_size)
]
train_dataset, retraining_batches, recall_batches, eval_qs = load_wmdp_simple_set(
    cfg.data, tokenizer
)


def _get_loss(model, batches, use_answer_mask=False):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**batch)
            if use_answer_mask:
                answer_mask = batch["answer_mask"]
                loss_acc += cross_entropy(output, batch, answer_mask).item()
            else:
                loss_acc += cross_entropy(output, batch).item()
    return loss_acc / len(batches)


def get_metrics(model):
    res = {}
    model.eval()

    # * eval forget acc
    res["forget_acc_t0"], res["forget_acc_t1"] = eval_on(eval_qs, model)

    nb = cfg.num_eval_batches
    res["wikitext_loss"] = _get_loss(model, wikitext_batches[:nb])
    res["retain_loss"] = _get_loss(model, [x["retain"] for x in train_dataset[:nb]])
    res["recall_loss"] = _get_loss(model, recall_batches, use_answer_mask=True)

    logging.info(res)
    return res


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)

max_layer = max(max(cfg.layer_range), max(cfg.cb_retaining_layers)) + 1

# * set trainable params
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)

install_hooks(model)

unit_optimizer = pt.optim.SGD(model.parameters(), lr=1.0)


# %%
# * install hooks for MLPs
def save_output_hook(module, args, output):
    module.cached_out = output


for layer_id in range(*cfg.layer_range):
    model.model.layers[layer_id].mlp.register_forward_hook(save_output_hook)

# * cache the activations for circuit breaker retaining
if cfg.get("retaining_rate", 0) > 0:
    for batch_pair in train_dataset:
        batch = batch_pair["retain"]
        with pt.no_grad():
            with trim_layers(model, max_layer):
                output = model(**batch, output_hidden_states=True)
        batch["retain_acts"] = {
            l_num: output.hidden_states[l_num].detach().to("cpu")
            for l_num in cfg.cb_retaining_layers
        }

# * cache the activations for MLP breaking
for batch_pair in train_dataset:
    batch = batch_pair["forget"]
    with pt.no_grad():
        output = model(**batch)
    _mask = batch["attention_mask"].bool().clone()
    _mask[:, : cfg.cut_off_tokens] = False
    batch["org_mlp_out"] = {}
    batch["org_mlp_out_norm"] = {}
    for layer_id in range(*cfg.layer_range):
        mlp = model.model.layers[layer_id].mlp
        out = mlp.cached_out.detach()[_mask]
        batch["org_mlp_out"][layer_id] = out.cpu()
        batch["org_mlp_out_norm"][layer_id] = out.float().norm(dim=-1).mean().cpu()

# %%

wandb.init(
    project="unlearning|src|CIR.py",
    name=f"no_trainer",
    config=OmegaConf.to_container(cfg),
)

init_res = get_metrics(model)
wandb.log(init_res)

# % full training loop
start_time = time.time()
for epoch in range(cfg.max_num_epochs):
    pt.cuda.empty_cache()

    acts_list = {n: [] for n, _ in trainable_modules(model)}
    grads_list = {n: [] for n, _ in trainable_modules(model)}

    # ! one epoch
    model.train()
    for batch_pair in train_dataset:
        batch = batch_pair["forget"]
        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        with trim_layers(model, max_layer):
            output = model(**batch, output_hidden_states=True)
        loss = loss_fns.mlp_confuse(model, batch, cfg)
        loss.backward()

        # ! here we modify the grad
        for name, module in trainable_modules(model):
            if module.weight.grad is None:
                continue
            acts = get_last_act(module, batch["attention_mask"], cfg.cut_off_tokens)
            grads = get_last_grad(module, batch["attention_mask"], cfg.cut_off_tokens)
            acts_list[name].append(acts.clone().to("cpu"))
            grads_list[name].append(grads.clone().to("cpu"))
            assert len(acts.shape) == len(grads.shape) == 2

            if epoch != 0:
                # ! proj out the means and PCA components
                for comp in act_to_collapse[name]:
                    acts -= project_out(acts, comp)
                for comp in grad_to_collapse[name]:
                    grads -= project_out(grads, comp)

                # without the projections, this is the equivalent of normal backprop
                module.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
                assert module.weight.grad.shape == module.weight.shape

        if epoch == 0:
            continue

        # * normalize grads
        norm = get_update_norm(model)
        scale_grads_(model, cfg.unlearning_rate / norm)
        unit_optimizer.step()  # unit_optimizer has lr=1.0

        if cfg.get("retaining_rate", 0) > 0:
            model.zero_grad(set_to_none=True)
            batch = batch_pair["retain"]
            with trim_layers(model, max_layer):
                output = model(**batch, output_hidden_states=True)
            loss = loss_fns.cb_retain(output, batch, cfg)
            loss.backward()

            scale_grads_(model, cfg.retaining_rate)  # apply intended lr
            unit_optimizer.step()  # unit_optimizer has lr=1.0

    if epoch % cfg.get("pca_every_n", 1) == 0:
        # ! calculate means and PCA components
        model.zero_grad(set_to_none=True)
        act_to_collapse = get_projections(acts_list, cfg.act_proj_num, cfg.cir_niter)
        grad_to_collapse = get_projections(grads_list, cfg.grad_proj_num, cfg.cir_niter)

    # ! get metrics
    res = get_metrics(model)
    wandb.log(res)
    if res["wikitext_loss"] > init_res["wikitext_loss"] * cfg.get("loss_budget", 1.01):
        break

wandb.finish()
logging.info(f"time taken: {time.time() - start_time:.2f}s")


# stats = dict(
#     update_norm=get_update_norm(model),
#     act_norm=output.hidden_states[4].norm(dim=-1).mean(),
# )
