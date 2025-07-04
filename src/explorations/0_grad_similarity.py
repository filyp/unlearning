# %%
# %load_ext autoreload
# %autoreload 2
import logging

import matplotlib.pyplot as plt
import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.training import get_grad, prepare_answer_mask, set_seeds

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)
pt.set_default_device("cuda")

conf = OmegaConf.load("../configs/transferability.yaml")
# conf.model_id = "meta-llama/Llama-3.2-3B"
conf.model_id = "meta-llama/Llama-3.2-1B"
# ! setup
set_seeds(42)
tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False


def get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy"):
    beginning_batch = tokenizer(beginning, **conf.tokenizer)
    full_batch = tokenizer(f"{beginning} {ending}", **conf.tokenizer)
    loss_mask = prepare_answer_mask(beginning_batch, full_batch)
    return get_grad(model, full_batch, loss_mask, loss_fn_name)


# %%

# ! limit which parameters are trained
# conf.target_modules = ["gate_proj", "up_proj", "down_proj", "k_proj", "v_proj", "q_proj", "o_proj"]  # fmt: skip
conf.target_modules = ["gate_proj", "up_proj", "down_proj"]
# conf.target_modules = ["gate_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)


# %%
# module_name = "model.layers.14.mlp.gate_proj.weight"

vs = []
# * we look at the transfer from the first pair to the last
# * the other pairs serve as basis for creating the mask
for beginning, ending in [
    ("The capital of France is", "Paris"),
    ("The capital of Spain is", "Madrid"),
    ("The capital of Italy is", "Rome"),
    # ("The capital of England is", "London"),
    # ("The capital of Poland is", "Warsaw"),
    # ("The capital of Argentina is", "Buenos Aires"),
    # ("The capital of Chile is", "Santiago"),
    # ("The capital of Japan is", "Tokio"),
    # ("The capital of Germany is", "Berlin"),
    # ("The capital of China is", "Beijing"),
    # ("The capital of Ukraine is", "Kyiv"),
    # ("The capital of Russia is", "Moscow"),

    # ("The capital of Poland is", "Kraków"),
    # ("The capital of Japan is", "Tokyo"),

    # ("The capital of Italy is", "Paris"),
    # ("The capital of the country below England is", "Paris"),

    # ("Столица Франции", "Париж"),
    # ("Stolica Francji to", "Paryz"),
    # ("La capital de Francia es", "París"),
    # ("Die Hauptstadt von Frankreich ist", "Paris"),
    # ("A capital de França é", "Paris"),

    # ("The anthem of France is", "La Marseillaise"),
    # ("The city of Eiffel Tower is", "Paris"),
    # ("The Statue of Liberty is in", "New York"),
    # ("The hardest metal is", "Tungsten"),

    # ("The term for egg development without fertilization is", "parthenogenesis"),
    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),

    # ("The term for self-fertilization is", "autogamy"),
    # ("The term for the death of cells is", "apoptosis"),
    # (format_prompt(q), ["A", "B", "C", "D"][q["answer"]]),

    # ("The symbol of helium is", "He"),
    # ("The Brandenburg Gate is in", "Berlin"),
    # ("The oldest building in the world is", "The Great Pyramid of Giza"),
]:
    # _g = get_grad_from_example(model, beginning, ending, loss_fn_name="cross_entropy")
    _g = get_grad_from_example(model, beginning, ending, loss_fn_name="correct_logit")
    vs.append(_g)

res = []
# * the first value is raw transfer, the others are transfers after masking
for final in [
    vs[0],
    vs[0] * ((vs[0] * (vs[1])) < 0),
    # vs[0] * ((vs[0] * (vs[2])) < 0),
    # vs[0] * ((vs[0] * (vs[1] + vs[2] + vs[3])) < 0),
    # vs[0] * ((vs[0] * (vs[1] + vs[2] + vs[3] + vs[4])) < 0),
    # vs[0] * ((vs[0] * (vs[1] + vs[2])) < 0),
]:
    bad = (final * vs[-1]).sum()
    good = (final * final).sum()
    _target_norm2 = (vs[-1] * vs[-1]).sum()

    bad = sum(v for v in bad.values())
    good = sum(v for v in good.values())
    _target_norm2 = sum(v for v in _target_norm2.values())

    ratio = bad / good
    cossim = bad / (good * _target_norm2).sqrt()
    print(f"{ratio=:7.4f}   {cossim=:7.4f}   {bad=:9.2f}   {good=:9.2f}")

    # bad = list(bad.values())
    # good = list(good.values())
    # res.append((bad, good))

# res = pt.Tensor(res)
# # res[:, :, 1] *= 0
# res /= res.max()
# r_channel = res[:, 0, :]
# g_channel = res[:, 1, :]
# # imshow
# colors = pt.zeros((r_channel.shape[0], r_channel.shape[1], 3))
# colors[:, :, 0] = r_channel * 10
# colors[:, :, 1] = g_channel
# colors = colors.clip(0, 1)
# # colors = colors / colors.max(dim=-1, keepdim=True).values
# colors[1] *= 10
# colors = colors.cpu().numpy()
# plt.imshow(colors)

# # %% A-GEM
# # project out the bad direction
# final = vs[0].flatten()
# to_avoid = vs[2].flatten()
# to_avoid /= to_avoid.norm()
# final = final - (final * to_avoid).sum() * to_avoid
# final = final.reshape(vs[0].shape)


# %%
