
# %%
# augment by rotations
target_mcq = []
for _ in range(4):
    target_mcq.append(q)
    # rotate the possible answers
    _tmp = q["choices"].pop(0)
    q["choices"].append(_tmp)
    q["answer"] = (q["answer"] - 1) % len(q["choices"])
    q["answer"], q["choices"]
target_mcq = Dataset.from_list(target_mcq)

control_mcq = []
for q_index in [0, 1, 2, 3, 4]:
    q_tmp = wmdp_mcq[q_index]
    for _ in range(4):
        control_mcq.append(q_tmp)
        # rotate the possible answers
        _tmp = q_tmp["choices"].pop(0)
        q_tmp["choices"].append(_tmp)
        q_tmp["answer"] = (q_tmp["answer"] - 1) % len(q_tmp["choices"])
        q_tmp["answer"], q_tmp["choices"]
control_mcq = Dataset.from_list(control_mcq)

# %%
lr = 0.0002
xs = []
ys = []
del model
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

# %%

target_acc = eval_on(target_mcq, model, temperature=1)
control_acc = eval_on(control_mcq, model, temperature=1)
xs.append(target_acc)
ys.append(control_acc)
print(f"target acc: {target_acc}, control acc: {control_acc}")

for n, p in model.named_parameters():
    if not any(pattern in n for pattern in ["o_proj", "gate_proj", "up_proj"]):
        continue
    if not any(pattern in n for pattern in [".0.", ".1.", ".2.", ".3.", ".4."]):
        continue
    p.data += interven_grad[n] * lr
