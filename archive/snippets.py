
# %% derive control grad
# control_grad = TensorDict({n: pt.zeros_like(p) for n, p in trainable_params(model)})
# # for q in [q_alt, q_alt2]:
# #     for q_rot in get_rotations(q):
# #         control_grad += get_grad_from_abcd_question(model, q_rot)

# # for control in q["controls_answer_end"]:
# #     control_grad += get_grad_from_example(model, control, q["answer_core"])

# # for beg, end in q["control_pairs"]:
# #     control_grad += get_grad_from_example(model, beg, end)

# for batch in fineweb_batches[:10]:
#     control_grad += get_grad(model, batch)

# norm = pt.Tensor(list(control_grad.norm().values())).norm()
# control_grad /= norm