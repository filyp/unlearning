# %%
import logging

import torch as pt
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from utils import masking
from utils.data_loading import load_batches, load_fineweb_edu_corpus, load_local
from utils.evals import eval_on
from utils.training import set_seeds

logging.basicConfig(level=logging.INFO)

pt.set_default_device("cuda")


conf = OmegaConf.load("../configs/precomputed_no_retain.yaml")

# load corpora
# f_all = load_local("wmdp_deduped_correct_answers_corpus.jsonl")
# r_all = load_local("wmdp_deduped_wrong_answers_corpus.jsonl")
f_all = load_local("my_generation/wmdp_bio.jsonl")

# load questions
wmdp_mcq = load_local(f"wmdp_deduped_{conf.category}/{conf.split}.jsonl")
# do not unlearn questions where the model already does not know
# use accuracies of llama 1B even for other models, to have the same questions
wmdp_mcq = wmdp_mcq.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

wmdp_mcq_disr = load_local(f"wmdp_deduped_{conf.category}/{conf.split_disr}.jsonl")
wmdp_mcq_disr = wmdp_mcq_disr.filter(lambda ex: ex["Llama-3.2-1B"] > 0.25)

# wmdp_mcq_disr = load_dataset("cais/mmlu", "all", split="validation")
# wmdp_mcq_disr = wmdp_mcq_disr.shuffle(seed=42).select(range(20))


# load disrution eval set
_fineweb_batches = load_batches(load_fineweb_edu_corpus(), conf.model_id, batch_size=8)
disruption_batches = _fineweb_batches[: conf.num_disruption_batches]
additional_retain_batches = _fineweb_batches[conf.num_disruption_batches :]

tokenizer = AutoTokenizer.from_pretrained(conf.model_id)
tokenizer.pad_token = tokenizer.eos_token


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def model_dist(model, orig_model):
    dist = 0
    for p, op in zip(trainable_params(model), trainable_params(orig_model)):
        dist += pt.sum((p.data - op.data) ** 2)
    return float(pt.sqrt(dist))


# # %%
# q = wmdp_mcq_disr[3]
# for _ in range(4):
#     # cycle
#     ans = q["choices"].pop(0)
#     q["choices"].append(ans)
#     q["answer"] = (q["answer"] - 1) % 4
#     acc = eval_on(Dataset.from_list([q]), model, temperature=1)
#     print(q["answer"], acc)
# # result: sometimes acc varies significantly, especially for 1B model
# # also, from before I know that in Spanish accuracy is much lower

# %% choose question
num_ex = 15

# ! load texts
f_corpora_per_question = []
for q in wmdp_mcq:
    f_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
    f_corpus = f_corpus.map(lambda ex: dict(text=f"{ex['beginning']} {ex['ending']}"))
    f_corpus = f_corpus.select(range(num_ex))
    f_corpora_per_question.append(f_corpus)
r_corpora_per_question = []
for q in wmdp_mcq_disr:
    r_corpus = f_all.filter(lambda ex: ex["original_question"] == q["question"])
    r_corpus = r_corpus.map(lambda ex: dict(text=f"{ex['beginning']} {ex['ending']}"))
    r_corpus = r_corpus.select(range(num_ex))
    r_corpora_per_question.append(r_corpus)


# %%
def _eval(model):
    model.eval()
    with pt.no_grad():
        wmdp_acc = eval_on(wmdp_mcq, model, temperature=1)
        wmdp_acc_disr = eval_on(wmdp_mcq_disr, model, temperature=1)
        # disr_loss = pt.mean(
        #     pt.Tensor([
        #         loss_fns.cross_entropy(model(**d_batch), d_batch)
        #         for d_batch in disruption_batches
        #     ])
        # )
    return float(wmdp_acc), float(wmdp_acc_disr)


# initial eval
orig_model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
orig_wmdp_acc, orig_disr_loss = _eval(orig_model)
logging.info(f"wmdp={orig_wmdp_acc:8.4f}    disr={orig_disr_loss:8.4f}    orig model")
del orig_model
pt.cuda.empty_cache()

# %%

conf = OmegaConf.load("../configs/precomputed_no_retain.yaml")
# conf.update(variant)
conf.name = "neg-CE-only-gate-proj"
# conf.unlearning_rate = 3e-5
conf.unlearning_rate = 6e-3
# conf.unlearning_method = "only_answer_tokens"
# conf.masking_method = "mask_out_answer_without_context"
conf.unlearning_method = "normal"
conf.masking_method = None

# ! setup
set_seeds(42)
model = AutoModelForCausalLM.from_pretrained(
    conf.model_id, torch_dtype=pt.bfloat16, device_map="cuda"
)
model.config.use_cache = False
wandb.init(
    project="unlearning-wmdp3",
    # name=f"{conf.unlearning_rate}-{conf.name}",
    name=f"{conf.unlearning_rate}-{conf.unlearning_method}-{conf.masking_method}-{conf.name}",
    group=f"reverting-retain-all-questions",
    config=OmegaConf.to_container(conf),
)
wandb.log({"wmdp_acc": orig_wmdp_acc, "wmdp_disr_acc": orig_disr_loss})

# ! limit which parameters are trained
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in conf.target_modules)

# ! calc unlearning
model.train()
for p in trainable_params(model):
    p.unlearning_grad_acc = pt.zeros_like(p.data)
for q_index, q in enumerate(wmdp_mcq):
    f_corpus = f_corpora_per_question[q_index]
    # r_corpus = r_corpora_per_question[q_index % len(r_corpora_per_question)]



    unlearning_method = getattr(masking, conf.unlearning_method)
    unlearning_method(model, tokenizer, conf, f_corpus)
    if conf.masking_method is not None:
        masking_method = getattr(masking, conf.masking_method)
        masking_method(model, tokenizer, conf, f_corpus)
        # masking_method(model, tokenizer, conf, r_corpus)


    # for i, forget_text in enumerate(f_corpus):
    #     masking.only_answer_tokens(model, tokenizer, conf, forget_text)
    #     masking.mask_out_answer_without_context(model, tokenizer, conf, forget_text)

    #     if i == 0:
    #         # ! first batch, so initialize acc
    #         for p in trainable_params(model):
    #             p.acc = p.grad
    #         continue

    #     # ! update acc
    #     for p in trainable_params(model):
    #         mask = p.acc.sign() == p.grad.sign()
    #         p.acc *= mask
    #         p.grad *= mask

    #         p.acc += p.grad
    #         del p.grad

    for p in trainable_params(model):
        # p.unlearning_grad_acc += p.acc
        p.unlearning_grad_acc += p.grad


# ! normalize grads and record
grad_norm = sum(p.unlearning_grad_acc.norm() ** 2 for p in trainable_params(model)) ** 0.5
logging.info(f"q_index={q_index:4d}   grad_norm={grad_norm:8.4f}")
for p in trainable_params(model):
    p.unlearning_grad_acc += p.unlearning_grad_acc / grad_norm


# ! unlearning
for epoch in range(conf.num_epochs):
    # ! apply unlearning
    for p in trainable_params(model):
        p.data -= p.unlearning_grad_acc * conf.unlearning_rate

    # ! evaluate
    wmdp_acc, wmdp_acc_disr = _eval(model)
    logging.info(
        f"epoch={epoch:4d}   wmdp={wmdp_acc:8.4f}    wmdp_disr={wmdp_acc_disr:8.4f}"
    )
    wandb.log({"wmdp_acc": wmdp_acc, "wmdp_disr_acc": wmdp_acc_disr})

wandb.finish()

# ! clean up
del model
pt.cuda.empty_cache()
