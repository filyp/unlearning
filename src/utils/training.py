import random

import numpy as np
import torch as pt
import torch.nn.functional as F
from tensordict import TensorDict
from transformers import set_seed as set_transformers_seed

from utils import loss_fns


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)
    # pt.use_deterministic_algorithms(True)


def relearn(model, relearn_batches, conf, eval_callback):
    # relearning
    set_seeds(42)
    optimizer = pt.optim.SGD(model.parameters(), lr=conf.lr)
    for p in model.parameters():
        p.requires_grad = True
    num_of_loops = int(len(relearn_batches) * conf.epochs)
    for loop_num in range(num_of_loops):
        pt.cuda.empty_cache()
        batch_index = loop_num % len(relearn_batches)
        batch = relearn_batches[batch_index]

        if batch_index == 0:
            eval_callback(model)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = loss_fns.cross_entropy(output, batch)
        loss.backward()
        optimizer.step()

    return model


def trainable_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def trainable_modules(model):
    return [
        (n, m)
        for n, m in model.named_modules()
        if "_proj" in n and m.weight.requires_grad
    ]


def get_grad(model, batch, loss_mask=None, loss_fn_name="cross_entropy"):
    # deprecated
    model.zero_grad(set_to_none=True)
    output = model(**batch)
    if loss_mask is not None:
        # moving this before model inference may break the inference, so keep it here
        batch["attention_mask"] *= loss_mask
    loss_fn = getattr(loss_fns, loss_fn_name)
    loss = loss_fn(output, batch)
    loss.backward()
    grad = TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
    )
    model.zero_grad(set_to_none=True)
    return grad


def prepare_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    answer_mask = (long_attn != short_attn_padded).to(pt.int64)
    return answer_mask


# def get_grad_from_pair(model, tokenizer, conf, beginning, ending):
#     beginning_batch = tokenizer(beginning, **conf.tokenizer)
#     full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
#     loss_mask = prepare_answer_mask(beginning_batch, full_batch)
#     return get_grad(model, full_batch, loss_mask)


def PCA_gpu(v, n_components=10, center=True):
    # Center the data
    if center:
        v = v - v.mean(axis=0)
    # Compute covariance matrix
    cov = (v.T @ v) / (v.shape[0] - 1)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = pt.linalg.eigh(cov)
    # Sort in descending order
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Get the top n_components
    return eigenvectors.T[:n_components]


def get_grads_dict(model):
    grads_dict = TensorDict(
        {n: p.grad for n, p in model.named_parameters() if p.requires_grad},
    )
    model.zero_grad(set_to_none=True)
    return grads_dict


# def make_sure_optimal_values_are_not_near_range_edges(study):
#     best_trial = study.best_trial  # ask only once because it's slow
#     """Make sure the value is not in the top or bottom 10% of the range."""
#     for param_name, param_dist in best_trial.distributions.items():
#         min_ = param_dist.low
#         max_ = param_dist.high
#         value = best_trial.params[param_name]
#         if param_dist.log:
#             min_ = np.log10(min_)
#             max_ = np.log10(max_)
#             value = np.log10(value)

#         method_name = study.study_name.split("|")[-1]
#         if value < min_ + 0.1 * (max_ - min_):
#             print(f"\t{param_name}\t in bottom 10% with value {value} in {method_name}")
#         if value > max_ - 0.1 * (max_ - min_):
#             print(f"\t{param_name}\t in top 10% with value {value} in {method_name}")
#             # print(f"WARNING: {param_name} in the top 10% of the range in best trial")
#             # print(f"range: {min_} - {max_}, value: {value}, log={param_dist.log}")


# # stats for the last n non-pruned trials
# def get_stats_from_last_n_trials(study, trials, n=10):
#     ok_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
#     print(f"all_trials={len(trials)}, ok_trials={len(ok_trials)}, {study.study_name}")
#     values = [t.values[0] for t in ok_trials]

#     # max_val = study.best_trial.values[0]
#     last_n_mean = np.mean(values[-n:])
#     last_n_sem = np.std(values[-n:]) / np.sqrt(n)
#     pure_name = study.study_name.split("|")[-1]
#     # result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {max_val:.4f} | {pure_name} |  |"
#     result = f"| {last_n_mean:.4f}±{last_n_sem:.4f} | {pure_name} |  |"
#     return result, last_n_mean, last_n_sem


# def delete_study_if_exists(study_name, storage):
#     try:
#         _ = optuna.load_study(study_name=study_name, storage=storage)
#         optuna.delete_study(study_name=study_name, storage=storage)
#     except KeyError:
#         pass
#         pass
