# %%
import re

import torch as pt
from datasets import load_dataset
from transformers import AutoModelForCausalLM

pt.set_default_device("cuda")
base_url = "https://raw.githubusercontent.com/aghyad-deeb/unlearning_evaluation/refs/heads/main/data"


# bio=1065 cyber=1179 other=111, where other are due to small updates to official WMDP
def load_low_mi_set(paths):
    return load_dataset(
        "json", data_files=[f"{base_url}/{path}" for path in paths], split="train"
    )

# todo: wmdp-deduped has been deduped to ensure low mutual information between splits
# todo: but after we have already split, we could reintroduce the duplicates!

# %%
questions_T = load_low_mi_set([
    "wmdp-deduped/split_0.jsonl",
    "wmdp-deduped/split_1.jsonl",
    "wmdp-deduped/split_2.jsonl",
    "wmdp-deduped/split_3.jsonl",
])
questions_V = load_low_mi_set(["wmdp-deduped/split_4.jsonl"])

questions_V = questions_V.map(lambda x: {"s": "V"})
questions_T = questions_T.map(lambda x: {"s": "T"})
from datasets import concatenate_datasets

questions = concatenate_datasets([questions_T, questions_V])
print(f"{len(questions)=}")


# %%
# filter out all negated questions
# if we don't, some corpus examples are awkard, speaking only of the benign option
# about 6% of the questions are removed
def is_not_a_not_question(q):
    text = q["question"].lower()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    last_sentence = sentences[-1]
    return "not" not in last_sentence


questions = questions.filter(is_not_a_not_question)
print(f"{len(questions)=}")


# %%
# filter out all "none of the above" questions
# they also result in awkward and mostly benign corpus examples
# removes 0.7% of the questions
def is_not_a_none_of_the_above_question(q):
    correct_ans = q["choices"][q["answer"]].lower()
    if "none" in correct_ans and ("above" in correct_ans or "choices" in correct_ans):
        return False
    return True


questions = questions.filter(is_not_a_none_of_the_above_question)
print(f"{len(questions)=}")


# %%
# official wmdp has been updated, so now some questions differ by some punctuation
def sanitize_question(question):
    return question.lower().replace("?", "").replace(".", "").replace(",", "").replace("!", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace(" ", "").replace("-", "").replace("_", "").replace("\n", "").replace("\\", "").replace("`", "").replace("*", "").replace("^", "").replace("~", "").replace("+", "").replace("=", "").replace("|", "").replace(":", "").replace(";", "").replace("'", "").replace('"', "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("'", "").replace("\t", "")  # fmt: skip


_wmdp_bio = load_dataset("cais/wmdp", "wmdp-bio")["test"]
bio_questions = [sanitize_question(ex["question"]) for ex in _wmdp_bio]
_wmdp_cyber = load_dataset("cais/wmdp", "wmdp-cyber")["test"]
cyber_questions = [sanitize_question(ex["question"]) for ex in _wmdp_cyber]


# %%
def ex_to_cat(ex):
    q = sanitize_question(ex["question"])
    if q in bio_questions:
        return {"category": "bio"}
    elif q in cyber_questions:
        return {"category": "cyber"}
    else:
        return {"category": "other"}


questions = questions.map(ex_to_cat)

from collections import Counter

print(Counter(questions["category"]))

# there are just 6 "other" after sanitization
questions = questions.filter(lambda ex: ex["category"] != "other")

# %%
# choose random 20% questions as dev set
dev_size = int(0.2 * len(questions))
questions = questions.shuffle(seed=42)  # For reproducibility
dev_indices = range(dev_size)
train_indices = range(dev_size, len(questions))

# Split into dev and train
dev_questions = questions.select(dev_indices)
train_questions = questions.select(train_indices)

# Modify the 's' field for dev set by prepending 'dev_'
dev_questions = dev_questions.map(lambda x: {"s": f"dev_{x['s']}"})

# Recombine the datasets
questions = concatenate_datasets([train_questions, dev_questions])

# %%
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=pt.bfloat16, device_map="cuda"
)

# %%
import os

# add to pythonpath the parent directory
os.sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset

from utils.evals import eval_on

# %%

questions = questions.map(
    lambda ex: {"Llama-3.2-1B": eval_on(Dataset.from_list([ex]), model, temperature=1)}
)

# %%
from git_and_reproducibility import repo_root

general_data_path = repo_root() / "data"
for category in ["bio", "cyber"]:
    for split in ["T", "V", "dev_T", "dev_V"]:
        path = general_data_path / f"wmdp_deduped_{category}" / f"{split}.jsonl"
        sub_q = questions.filter(
            lambda ex: (ex["s"] == split) and (ex["category"] == category)
        )
        sub_q = sub_q.remove_columns(["s", "category"])
        print(path, len(sub_q))
        sub_q.to_json(path)

# %%

correct_corpus = load_low_mi_set([
    "wmdp-deduped/corpus_split_0.jsonl",
    "wmdp-deduped/corpus_split_1.jsonl",
    "wmdp-deduped/corpus_split_2.jsonl",
    "wmdp-deduped/corpus_split_3.jsonl",
    "wmdp-deduped/corpus_split_4.jsonl",
])
correct_corpus.to_json(general_data_path / "wmdp_deduped_correct_answers_corpus.jsonl")

# %%

wrong_answers_corpus = load_low_mi_set([
    "wmdp-deduped/whp_corpus_split_0.jsonl",
    "wmdp-deduped/whp_corpus_split_1.jsonl",
    "wmdp-deduped/whp_corpus_split_2.jsonl",
    "wmdp-deduped/whp_corpus_split_3.jsonl",
    "wmdp-deduped/whp_corpus_split_4.jsonl",
])
wrong_answers_corpus.to_json(
    general_data_path / "wmdp_deduped_wrong_answers_corpus.jsonl"
)

# %%

# years_unlearning=[
#     "dates-years-trimmed/corpus_split_0.jsonl",
#     "dates-years-trimmed/corpus_split_1.jsonl",
#     "dates-years-trimmed/corpus_split_2.jsonl",
#     "dates-years-trimmed/corpus_split_3.jsonl",
#     "dates-years-trimmed/corpus_split_4.jsonl",
# ],
# years_relearning=[
#     "dates-years-trimmed/corpus_split_0.jsonl",
#     "dates-years-trimmed/corpus_split_1.jsonl",
#     "dates-years-trimmed/corpus_split_2.jsonl",
#     "dates-years-trimmed/corpus_split_3.jsonl",
# ],
# years_mcq_eval=[
#     "dates-years-trimmed/split_4.jsonl",
# ],
# mmlu_forget = [
#     "mmlu_cats_random_trimmed/corpus_mmlu_STEM.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_business.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_chemistry.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_culture.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_geography.jsonl",
# ],
# mmlu_retain=[
#     "mmlu_cats_random_trimmed/corpus_mmlu_health.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_history.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_law.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_philosophy.jsonl",
#     "mmlu_cats_random_trimmed/corpus_mmlu_social sciences.jsonl",
# ],

# %%
