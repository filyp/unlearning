
def unlearn_common_core(
    h,
    conf,
    retain_batches,
    forget_batches,
    eval_callback,
):
    loss_fn = loss_fns[h.unlearning_loss_fn]

    set_seeds(42)
    model = AutoModelForCausalLM.from_pretrained(conf.model_id, torch_dtype=pt.bfloat16)
    model.config.use_cache = False

    # ! unlearning loop
    for loop_num in range(conf.unlearning_epochs):
        eval_callback(model)

        # ! forget pass
        assert len(forget_batches) == 3
        assert len(forget_batches[0]["input_ids"]) == 1
        for batch in forget_batches:
            model.zero_grad(set_to_none=True)
            output = model(**batch)
            forget_loss = loss_fn(output, batch, 0)
            forget_loss.backward()
            for p in model.parameters():
                if not hasattr(p, "acc"):
                    # first batch, so initialize acc
                    p.acc = p.grad.detach().clone()
                    continue

                if h.use_masking:
                    mask = p.acc.sign() == p.grad.sign()
                    p.acc *= mask
                    p.grad *= mask

                p.acc += p.grad

        # ! retain pass
        assert len(retain_batches) == 3
        assert len(retain_batches[0]["input_ids"]) == 1
        for batch in retain_batches:
            model.zero_grad(set_to_none=True)
            output = model(**batch)
            retain_loss = cross_entropy_loss(output, batch)
            retain_loss.backward()
            for p in model.parameters():
                if h.use_masking:
                    mask = p.acc.sign() == p.grad.sign()
                    p.acc *= mask

        for p in model.parameters():
            p.data -= h.unlearning_rate * p.acc
            del p.acc

    return model
