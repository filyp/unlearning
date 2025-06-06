
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


# %%

means = vs.mean(axis=0)[:lim]
stds = vs.std(axis=0)[:lim]
target = vs[0, :lim]

plt.axhline(y=0, color="white", linestyle="-")

# plt.errorbar(range(len(means)), means, yerr=stds, fmt='none', ecolor="red")
# plt.scatter(range(len(target)), target, color="white", marker="o")

plt.errorbar(range(len(means)), means*0, yerr=stds, fmt='none', ecolor="red")
plt.scatter(range(len(target)), target - means, color="white", marker="o")

plt.show()

# %%

# for word in ["France", "Japan", "Korea", "India", "Pakistan", "Bangladesh", "Afghanistan", "Iran", "Iraq", "Israel", "Palestine", "Jordan", "Qatar", "Bahrain", "Oman", "Kuwait", "Lebanon", "Syria", "Turkey"]:
#     beginning = f"The capital of {word} is"

# for word in ["capital", "population", "area", "language", "currency", "religion", "government", "anthem", "flag", "climate", "food"]:
#     beginning = f"The {word} of France is"