import torch as pt
import torch.nn.functional as F

# new_logits = []
# new_target_probs = []
# for id_, target_prob, logit in zip(ids, target_probs, logits):
#     mask = pt.ones_like(logit, dtype=pt.bool)
#     mask[id_] = False
#     new_logits.append(logit[mask])
#     new_target_probs.append(target_prob[mask])
# new_logits = pt.stack(new_logits)
# new_target_probs = pt.stack(new_target_probs)


def _normalize_logits(logits):
    """Shift the raw logits to make them logs of probabilities summing to one."""
    return logits - logits.exp().sum(dim=-1, keepdim=True).log()


def kl_loss(output, batch, model, mask):

    logits = output.logits[mask]
    # we store acts and recalculate logits to save memory
    original_last_act = batch["original_last_act"].to("cuda")[mask]
    original_logits = (model.model.embed_tokens.weight @ original_last_act.T).T

    logits = _normalize_logits(logits.float())
    original_logits = _normalize_logits(original_logits.float())

    # calculate KL divergence between original and current logits
    kl_div = F.kl_div(logits, original_logits, reduction="batchmean", log_target=True)
    assert kl_div > -1e-6  # it can be slightly negative due to numerical errors
    # this kl_div calculation is the same as:
    # (original_logits.exp() * (original_logits - logits)).sum(dim=-1).mean()
    return kl_div


def cross_entropy(output, batch, answer_mask=None):
    shifted_logits = output.logits[:, :-1, :].float()
    shifted_ids = batch["input_ids"][:, 1:]

    # mask out the beginning tokens
    if answer_mask is not None:
        # note, that answer_mask is a subset of attention mask
        assert pt.all(batch["attention_mask"] * answer_mask == answer_mask)
        mask = answer_mask
    else:
        mask = batch["attention_mask"]
    mask = mask[:, 1:].bool()

    return pt.nn.functional.cross_entropy(
        shifted_logits[mask],
        shifted_ids[mask],
        # reduction="sum",
    )
    # equivalent to:
    # probs = pt.nn.functional.softmax(logits, dim=-1)
    # true_probs = probs[pt.arange(len(ids)), ids]
    # return -pt.log(true_probs).mean()


def neg_cross_entropy(output, batch, answer_mask=None):
    return -cross_entropy(output, batch, answer_mask)


def correct_logit(output, batch, answer_mask=None, clip_at=0):
    logits = output.logits[:, :-1, :].flatten(end_dim=1).float()
    ids = batch["input_ids"][:, 1:].flatten()
    true_logits = logits[pt.arange(len(ids)), ids]
    # true_logits -= logits.mean(dim=-1)  # this is for the _minus_avg version
    # true_logits -= logits.mean(dim=-1).detach()

    true_logits = true_logits.clip(min=clip_at)

    # mask out the beginning tokens
    if answer_mask is not None:
        # note, that answer_mask is a subset of attention mask
        assert pt.all(batch["attention_mask"] * answer_mask == answer_mask)
        mask = answer_mask
    else:
        mask = batch["attention_mask"]
    mask = mask[:, 1:].bool().flatten()

    return true_logits[mask].mean()


# def proj_out_target(output, batch, answer_mask, model):
#     shifted_mask = answer_mask[:, 1:].bool()

#     # get relevant tensors and shift them
#     shifted_ids = batch["input_ids"][:, 1:]
#     shifted_last_act = output.hidden_states[-1][:, :-1, :].float()
#     shifted_org_last_act = batch["last_act"][:, :-1, :].float().to("cuda")

#     # take only relevant vectors, according to the answer mask
#     ids = shifted_ids[shifted_mask]
#     last_act = shifted_last_act[shifted_mask]
#     org_last_act = shifted_org_last_act[shifted_mask]

#     # get the embeddings and normalize them
#     embs = model.model.embed_tokens.weight[ids].float()
#     embs /= embs.norm(dim=1, keepdim=True)

#     # # these modifications essentially modify it into a "correct logit" loss
#     # proj_magns = pt.einsum("is,is->i", last_act, embs)
#     # proj_magns = proj_magns.clip(min=0)
#     # return (proj_magns ** 1.1).mean()

#     # project out the embeddings
#     proj_magns = pt.einsum("is,is->i", org_last_act, embs)
#     targets = org_last_act - proj_magns.reshape(-1, 1) * embs

#     # loss is how far we are from the target
#     return ((targets - last_act).norm(dim=1) ** 1.1).mean()


# #########################################################


def cross_entropy_per_token(output, batch):
    shifted_logits = output.logits[:, :-1, :].float()
    shifted_ids = batch["input_ids"][:, 1:]

    assert pt.all(batch["attention_mask"] == 1), "only full-length inputs supported"
    assert batch["attention_mask"].shape[0] == 1

    logits = shifted_logits.flatten(end_dim=1)
    ids = shifted_ids.flatten()

    probs = pt.nn.functional.softmax(logits, dim=-1)
    true_probs = probs[pt.arange(len(ids)), ids]
    return -pt.log(true_probs)


# def stream_activation(output, batch):
#     # last activation is huge for some reason, so ignore it
#     acc = 0
#     flat_attn_mask = batch["attention_mask"].reshape(-1, 1)
#     for layer_acts in output.hidden_states[:-1]:
#         flat_acts = layer_acts.flatten(end_dim=1)
#         flat_acts *= flat_attn_mask
#         acc += flat_acts.norm(dim=-1).mean() ** 2
#     return acc / flat_attn_mask.sum()


# adapted from https://github.com/rishub-tamirisa/tamper-resistance/blob/41b749ca4d9bcb7608c7ead2ca48b0508714af99/modules/objectives.py#L114
def neg_entropy(output, batch) -> pt.Tensor:
    # todo implement answer mask
    """
    Compute the negative mean entropy loss for the given logits.

    This function calculates the entropy of the softmax distribution of the input logits
    and returns the negative mean entropy as a loss value. Minimizing this loss
    encourages the model to produce more uniform (higher entropy) probability distributions.

    Returns:
        pt.Tensor: The negative mean entropy loss.
    """
    logits = output.logits
    softmax = pt.nn.functional.softmax(logits, dim=-1)
    log_softmax = pt.nn.functional.log_softmax(logits, dim=-1)
    neg_entropy = -pt.sum(-softmax * log_softmax, dim=-1)
    neg_entropy *= batch["attention_mask"]
    return neg_entropy.sum() / batch["attention_mask"].sum()


# def circuit_breaker_forget(
#     model,
#     forget_inputs,
#     target_layers,
#     frozen_model=None,
#     lora_model=None,
# ):

#     # ===== loss components =====
#     layers_forget_attention_mask = (
#         forget_inputs["attention_mask"].repeat(len(target_layers), 1, 1).unsqueeze(-1)
#     )

#     if lora_model is not None and frozen_model is None:
#         lora_model.disable_adapter_layers()
#         frozen_model = model

#     if lora_model is None and frozen_model is None:
#         raise Exception("Function did not get frozen model and LoRA is disabled.")

#     assert frozen_model is not None

#     frozen_model.eval()
#     with pt.no_grad():
#         forget_outputs = frozen_model(
#             **forget_inputs, output_hidden_states=True
#         ).hidden_states
#         forget_hidden = pt.stack([forget_outputs[l].detach() for l in target_layers])
#     del forget_outputs
#     gc.collect()

#     if lora_model is not None:
#         lora_model.enable_adapter_layers()
#     model.train()

#     lora_forget_outputs = model(
#         **forget_inputs, output_hidden_states=True
#     ).hidden_states
#     lora_forget_hidden = pt.stack([lora_forget_outputs[l] for l in target_layers])

#     normalized_lora_forget_outputs = lora_forget_hidden / (
#         pt.norm(lora_forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
#     )
#     normalized_forget_outputs = forget_hidden / (
#         pt.norm(forget_hidden, dim=-1, keepdim=True, dtype=pt.float)
#     )
#     inner_product = (
#         normalized_lora_forget_outputs * normalized_forget_outputs
#     ) * layers_forget_attention_mask
#     forget_loss = (
#         pt.relu(inner_product.sum(dim=-1)).sum() / layers_forget_attention_mask.sum()
#     )

#     return forget_loss


# def circuit_breaker_retain(
#     model, retain_inputs, frozen_model=None, lora_model=None, square_norm=False
# ):

#     if lora_model is not None:
#         lora_model.disable_adapter_layers()
#         frozen_model = model

#     if lora_model is None and frozen_model is None:
#         raise Exception("Function did not get frozen model and LoRA is disabled.")

#     assert frozen_model is not None

#     frozen_model.eval()
#     with pt.no_grad():
#         orig_retain_outputs = frozen_model(
#             **retain_inputs, output_hidden_states=True
#         ).hidden_states
#         orig_retain_hidden = pt.stack(orig_retain_outputs).detach()
#         layers_retain_attention_mask = (
#             retain_inputs["attention_mask"]
#             .repeat(len(orig_retain_outputs), 1, 1)
#             .unsqueeze(-1)
#         )
#         orig_retain_hidden *= layers_retain_attention_mask

#     del orig_retain_outputs
#     gc.collect()

#     if lora_model is not None:
#         lora_model.enable_adapter_layers()
#     model.train()

#     lora_retain_outputs = model(
#         **retain_inputs, output_hidden_states=True
#     ).hidden_states
#     lora_retain_hidden = pt.stack(lora_retain_outputs) * layers_retain_attention_mask
#     diffs = lora_retain_hidden - orig_retain_hidden
#     # the last hidden state is anomalously high (at least for pythia)
#     diffs = diffs[:-1]

#     if square_norm:
#         return pt.norm(diffs, dim=-1, p=2, dtype=pt.float).pow(2).nanmean()
#     else:
#         return pt.norm(diffs, dim=-1, p=2, dtype=pt.float).nanmean()


# def correct_logit_loss(output, input_ids):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     return true_logits.mean()


# def clipped_correct_logit_loss(output, input_ids):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     # note: clipping at 0 is actually pretty useless, because logits don't come that low!
#     return true_logits.clip(min=0).mean()


# def soft_clipped_correct_logit_loss(output, input_ids, atan_scale):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     true_logits = logits[pt.arange(len(ids)), ids]
#     soft_clipped = (true_logits / atan_scale).atan() * atan_scale
#     return soft_clipped.mean()


# def soft_clipped_cross_entropy_loss(output, input_ids, atan_scale):
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     probs = pt.nn.functional.softmax(logits, dim=-1)
#     true_probs = probs[pt.arange(len(ids)), ids]
#     losses = -pt.log(true_probs)
#     soft_clipped = (losses / atan_scale).atan() * atan_scale
#     return soft_clipped.mean()


# def flipped_prob_loss(output, input_ids, correct_logit_bias=0, only_grad_correct=False):
#     """Compute loss that tries to reduce probability of correct tokens.

#     In contrast to neg_entropy_loss, it's monotonic in respect to the correct logit.
#     It has two asymptotes: on the right has derivative 1, on the left has derivative 0.

#     Args:
#         output: Model output containing logits
#         input_ids: Input token IDs
#         correct_logit_bias: Bias added to correct token logits
#         only_grad_correct:
#             If True, only compute gradients for correct token positions
#             If False, compute gradients for all tokens
#     """
#     orig_logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     target_indices = (pt.arange(len(ids)), ids)

#     if only_grad_correct:
#         # create a detached copy of logits
#         logits = orig_logits.detach().clone()
#         # only keep gradients for the target tokens
#         logits[target_indices] = orig_logits[target_indices]
#     else:
#         logits = orig_logits

#     # shift up the correct logits, so that they'll need to be brought further down
#     logits[target_indices] += correct_logit_bias
#     probs = pt.nn.functional.softmax(logits, dim=-1)

#     true_probs = probs[pt.arange(len(ids)), ids]
#     # ! invert the probability
#     losses = -pt.log(1 - true_probs)
#     return losses.mean()


# # simpler way to get all the loss functions
# loss_fns = {
#     name: obj
#     for name, obj in globals().items()
#     if callable(obj) and not name.startswith("_") and obj.__module__ == __name__
# }

# def biased_neg_entropy_loss(output, input_ids, correct_logit_bias) -> pt.Tensor:
#     logits = output.logits[:, :-1, :].flatten(end_dim=1).to(pt.float32)
#     ids = input_ids[:, 1:].flatten()
#     # shift up the correct logits, so that they'll need to be brought further down
#     logits[pt.arange(len(ids)), ids] += correct_logit_bias
#     # calculate entropy
#     softmax = pt.nn.functional.softmax(logits, dim=-1)
#     log_softmax = pt.nn.functional.log_softmax(logits, dim=-1)
#     entropy = pt.sum(-softmax * log_softmax, dim=-1).mean()
#     return entropy.mean() * -1


# def non_target_disruption(output, batch, target_logits):
#     _vocab_size = output.logits.shape[2]
#     shifted_logits = output.logits[:, :-1, :].to(pt.float32)
#     shifted_target_logits = target_logits[:, :-1, :].detach().clone()
#     shifted_ids = batch["input_ids"][:, 1:]
#     shifted_attn_mask = batch["attention_mask"][:, 1:] == 1

#     logits = shifted_logits[shifted_attn_mask]
#     target_logits_flat = shifted_target_logits[shifted_attn_mask]
#     ids = shifted_ids[shifted_attn_mask]
#     assert ids.shape == (shifted_attn_mask.sum(),)
#     assert logits.shape == target_logits_flat.shape == (ids.shape[0], _vocab_size)

#     target_logits_flat[pt.arange(len(target_logits_flat)), ids] = float("-inf")
#     target_probs = pt.nn.functional.softmax(target_logits_flat, dim=-1)

#     # mask out the correct answer
#     # this is equivalent to the new_logit = []... code below
#     mask = pt.ones_like(logits, dtype=pt.bool)
#     mask[pt.arange(len(logits)), ids] = False
#     selected_logits = logits[mask].reshape(len(logits), _vocab_size - 1)
#     selected_target_probs = target_probs[mask].reshape(len(logits), _vocab_size - 1)

#     return pt.nn.functional.cross_entropy(
#         input=selected_logits, target=selected_target_probs
#     )
