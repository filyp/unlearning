"""These functions are supposed to set the gradients, based on some texts."""

from datasets import Dataset

from utils import loss_fns


def no_mask(model, tokenizer, conf, f_texts: Dataset, r_texts: Dataset):
    f_batch = tokenizer(f_texts["text"], **conf.tokenizer)
    r_batch = tokenizer(r_texts["text"], **conf.tokenizer)
    model.train()

    # ! forget pass
    model.zero_grad(set_to_none=True)
    output = model(**f_batch)
    loss_fn = getattr(loss_fns, conf.unlearning_loss_fn)
    forget_loss = loss_fn(output, f_batch)
    forget_loss.backward()


def disruption_mask(model, tokenizer, conf, f_texts: Dataset, r_texts: Dataset):
    f_batch = tokenizer(f_texts["text"], **conf.tokenizer)
    r_batch = tokenizer(r_texts["text"], **conf.tokenizer)
    model.train()

    # ! retain pass
    model.zero_grad(set_to_none=True)
    output = model(**r_batch)
    retain_loss = loss_fns.cross_entropy(output, r_batch)
    retain_loss.backward()
    for p in model.parameters():
        p.retain_acc = p.grad

    # ! forget pass
    model.zero_grad(set_to_none=True)
    output = model(**f_batch)
    loss_fn = getattr(loss_fns, conf.unlearning_loss_fn)
    forget_loss = loss_fn(output, f_batch)
    forget_loss.backward()

    for p in model.parameters():
        if not p.requires_grad:
            continue
        mask = p.retain_acc.sign() == p.grad.sign()
        del p.retain_acc
        p.grad *= mask


# # ! limit percentiles
# if h.percentile is not None:
#     abs_vals = p.grad.flatten().abs()
#     k = int(len(abs_vals) * h.percentile)
#     cutoff = abs_vals.kthvalue(k).values.item()
#     mask = p.grad.abs() > cutoff
#     p.grad *= mask
