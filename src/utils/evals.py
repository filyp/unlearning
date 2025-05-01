"""Evaluations for WMDP, MMLU, WMDP-Deduped."""

# %%
import torch as pt
from datasets import load_dataset
from transformers import AutoTokenizer

from utils.data_loading import data_paths, load_low_mi_set

pt.set_default_device("cuda")
answer_tokens = [" A", " B", " C", " D"]


# %%
def format_prompt(ex):
    # taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wmdp/_default_template_yaml
    # "The following are multiple choice questions (with answers) about biology."
    return f"""\
{ex["question"].strip()}
A. {ex["choices"][0]}
B. {ex["choices"][1]}
C. {ex["choices"][2]}
D. {ex["choices"][3]}
Answer:"""


# %%
def eval_on(dataset, model, batch_size=4, subset=None, temperature=0):
    assert model.config.name_or_path in [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
    ]
    pt.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # note that this assumes start-of-sequence token is used (which is true for llama)
    answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)

    # sort wmdp_bio by the prompt length
    dataset = sorted(dataset, key=lambda ex: len(format_prompt(ex)))
    if subset is not None:
        dataset = dataset[:subset]

    acc = 0
    for i in range(0, len(dataset), batch_size):
        # print(i)
        batch = dataset[i : i + batch_size]
        batch_text = [format_prompt(ex) for ex in batch]

        input_dict = tokenizer(batch_text, return_tensors="pt", padding=True)

        with pt.inference_mode():
            output = model(**input_dict)
        last_positions = input_dict["attention_mask"].sum(dim=-1) - 1
        last_token_logits = output.logits[range(len(batch)), last_positions]

        probs = pt.softmax(last_token_logits, dim=-1)
        answer_probs = probs[:, answer_ids]
        # if not all(answer_probs.sum(dim=-1) > 0.2):
        #     raise ValueError("Sum of answer probs is too low")

        answer_probs /= answer_probs.sum(dim=-1, keepdim=True)  # normalize
        # assert pt.allclose(answer_probs.sum(dim=-1), pt.tensor(1.0, dtype=pt.bfloat16))
        _correct_answers = pt.tensor([ex["answer"] for ex in batch])

        if temperature == 1:
            correct_answer_probs = answer_probs[range(len(batch)), _correct_answers]
            acc += correct_answer_probs.sum().item()
        elif temperature == 0:
            # for temperature=0
            hits = answer_probs.argmax(dim=-1) == _correct_answers
            acc += hits.sum().item()
            # print(hits)
        else:
            raise ValueError(f"Not supported temperature: {temperature}")

        del answer_probs, probs, last_token_logits, output
        pt.cuda.empty_cache()

    return acc / len(dataset)
