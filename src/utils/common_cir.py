import torch as pt
from transformers import AutoModelForCausalLM

from utils.training import PCA_gpu, trainable_modules


def project_out(base, unwanted):
    # check dimensions
    _pos, _stream = base.shape
    (_stream2,) = unwanted.shape
    assert _stream == _stream2

    unwanted = unwanted / unwanted.norm()
    magnitudes = (base * unwanted).sum(axis=-1)
    return pt.einsum("t,s->ts", magnitudes, unwanted)


def save_act_hook(module, args):
    module.last_act_full = args[0].detach().clone()


def save_grad_hook(module, args):
    module.last_grad_full = args[0].detach().clone()


def install_hooks(model):
    for n, module in trainable_modules(model):
        module.register_forward_pre_hook(save_act_hook)
        module.register_full_backward_pre_hook(save_grad_hook)


def get_last_act(module, attn_mask):
    # ignore BOS token and the last token
    act = module.last_act_full[:, 1:-1]
    final_mask = attn_mask.bool()[:, 1:-1]
    return act[final_mask]


def get_last_grad(module, attn_mask):
    # ignore BOS token and the last token
    grad = module.last_grad_full[:, 1:-1]
    final_mask = attn_mask.bool()[:, 1:-1]
    return grad[final_mask]


def get_projections(vector_lists: dict[str, list[pt.Tensor]], num_proj=11, niter=16):
    # vectors can be either acts or grads
    num_pc = num_proj - 1
    to_collapse = {}
    for n in list(vector_lists.keys()):
        pt.cuda.empty_cache()
        vectors_flattened = pt.cat(vector_lists.pop(n)).to("cuda").float()
        mean = vectors_flattened.mean(axis=0)
        
        if num_proj == 0:
            to_collapse[n] = pt.tensor([])
            continue
        elif num_proj == 1:
            to_collapse[n] = mean.reshape(1, -1)
            continue

        _, S, V = pt.pca_lowrank(vectors_flattened, num_pc, niter=niter)
        pca_components = V.T

        # to collapse is one tensor of mean and the pca components
        to_collapse[n] = pt.cat([mean.reshape(1, -1), pca_components], dim=0)

    return to_collapse


# but it doesn't have batches, but it needs to be done only once, so maybe not important to optimize it
# def get_act_principal_components(model, batches, num_pc=10, niter=16):
#     # maybe deprecate this in favor of get_projections
#     # ! gather acts
#     acts_list = {n: [] for n, _ in trainable_modules(model)}

#     for batch in batches:
#         with pt.no_grad():
#             model(**batch)
#         for n, module in trainable_modules(model):
#             acts_list[n].append(get_last_act(module, batch["attention_mask"]).to("cpu"))

#     # ! calculate projection basis
#     act_means = {}
#     act_pca_components = {}
#     for n, _ in trainable_modules(model):
#         pt.cuda.empty_cache()
#         acts_flattened = pt.cat(acts_list.pop(n)).to("cuda").float()
#         act_means[n] = acts_flattened.mean(axis=0)
#         # ! calculate act PCA
#         # if exact_pca:
#             # #  note: it seems to leak memory!!
#             # act_pca_components[n] = PCA_gpu(acts_flattened, n_components=num_pc)
#         _, S, V = pt.pca_lowrank(acts_flattened, num_pc, niter=niter)
#         act_pca_components[n] = V.T

#     return act_means, act_pca_components


# def get_projections(acts_list, grads_list, num_pc=10, niter=16):
#     act_means = {}
#     grads_means = {}
#     act_pca_components = {}
#     for n in list(acts_list.keys()):
#         pt.cuda.empty_cache()
#         acts_flattened = pt.cat(acts_list.pop(n)).to("cuda").float()
#         act_means[n] = acts_flattened.mean(axis=0)
#         grads_flattened = pt.cat(grads_list.pop(n)).to("cuda").float()
#         grads_means[n] = grads_flattened.mean(axis=0)
#         _, S, V = pt.pca_lowrank(acts_flattened, num_pc, niter=niter)
#         act_pca_components[n] = V.T

#     return act_means, grads_means, act_pca_components


# # for comparing similarity of full and approximate PCA
# ref = list(act_pca_components.values())[0]
# (ref2 * ref).sum(dim=1).abs()


# note: on CPU, pca_lowrank is 3x faster than PCA_gpu
# but there are some differences (numerical?) on later components
# sklearn agrees with PCA_gpu, so looks that pca_lowrank is inaccurate after 8 PCs
# on CPU:
# PCA_gpu: 2s
# pca_lowrank: 0.7s
# sklearn pca: 26s! (and seems to be higher precision)
# wow, but for down_proj it's 30x faster!
# PCA_gpu: 60s
# pca_lowrank: 2s
# on GPU (and down_proj) it's even more drastic:
# 45s vs 0.1s, 450x
# on gate_proj:
# 1.4s vs 0.04, so 35x
# somehow on gpu it's more drastic
# with niter=8, for 8 PCs accuracy is decent, and speed drop is not that bad (only 2x worse than niter=2)


# ! to also include grads:
# acts_list = {n: [] for n, _ in trainable_modules(model)}
# # grads_list = {n: [] for n, _ in trainable_modules(model)}

# # gather acts and grads
# for ex in deception_set.select(range(20)):
#     print(".")
#     beginning_txt = ex["context"][-400:]
#     full_txt = f"{beginning_txt} {ex['answer']}"

#     # note: when not using grad, we could not use answer_mask
#     beginning_batch = tokenizer(beginning_txt, **conf.tokenizer)
#     batch = tokenizer(full_txt, **conf.tokenizer)
#     answer_mask = prepare_answer_mask(beginning_batch, batch)

#     output = model(**batch, output_hidden_states=True)
#     loss = loss_fn(output, batch, answer_mask)
#     # loss.backward()

#     for n, module in trainable_modules(model):
#         acts_list[n].append(get_last_act(module, batch["attention_mask"]))
#         # grads_list[n].append(get_last_act(module, batch["attention_mask"]))

# # * limit which layers are trained
# if ".layers." in n:
#     layer_num = int(n.split(".")[2])
#     # # * freeze early layers
#     # if layer_num < 16:
#     # p.requires_grad = False
#     # * use only every some layers to save memory
#     if use_every_n_layers is not None:
#         if layer_num % use_every_n_layers != 0:
#             p.requires_grad = False
