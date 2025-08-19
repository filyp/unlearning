# %%
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from utils.git_and_reproducibility import repo_root
from utils.training import prepare_answer_mask

# # mmlu_set = load_dataset("cais/mmlu", "college_biology", split="test")
# mmlu_set = load_dataset("cais/mmlu", "all", split="validation")
# mmlu_acc = eval_on(mmlu_set, model, temperature=1)


# %%
# load_local("wmdp_deduped_correct_answers_corpus.jsonl")
# load_local("wmdp_deduped_wrong_answers_corpus.jsonl")
# load_local("wmdp_deduped_bio/dev_T.jsonl")
# load_local("wmdp_deduped_bio/dev_V.jsonl")
def load_local(path):
    path = repo_root() / "data" / path
    return load_dataset("json", data_files=[path.as_posix()], split="train")


# %%
def load_fineweb_edu_corpus():
    corpus = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        # just the smallest parquet file will be enough
        data_files=["sample/10BT/013_00000.parquet"],
    )
    return corpus.select(range(2_500))


def load_fineweb_bio_corpus():
    corpus = load_dataset(
        "m-a-p/FineFineWeb",
        split="train",
        # just the smallest parquet file will be enough
        data_files=["biology/biology_000849.jsonl"],
    )
    return corpus.select(range(2_500))


def load_fineweb_tech_corpus():
    corpus = load_dataset(
        "m-a-p/FineFineWeb",
        split="train",
        data_files=[
            "computer_science_and_technology/computer_science_and_technology_000000.jsonl"
        ],
    )
    return corpus.select(range(2_500))


# def load_filtered_mmlu_dataset():
#     # %% prepare filtered mmlu dataset
#     mmlu_dataset = load_dataset("cais/mmlu", "all", split="validation")

#     # filter out all the subcategories of biology and health
#     # keep even the ones like anatomy, clinical_knowledge and professional_medicine,
#     # because they contain some questions about molecular biology
#     categories_to_reject = {
#         "college_biology",
#         "college_medicine",
#         "high_school_biology",
#         "human_aging",
#         "medical_genetics",
#         "nutrition",
#         "professional_medicine",
#         "virology",
#         "anatomy",
#         "clinical_knowledge",
#     }
#     # filter out the ones in the categories_to_reject
#     return [ex for ex in mmlu_dataset if ex["subject"] not in categories_to_reject]


def load_jigsaw_dataset():
    full_path = repo_root() / "data" / "jigsaw" / "train.csv"
    if not full_path.exists():
        txt = """\
You must manually download the Jigsaw dataset from:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
and put the csv files in the data/jigsaw directory.
        """
        raise FileNotFoundError(txt)
    return pd.read_csv(full_path)


# %%
def load_batches(corpus, model_id, batch_size=4, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    corpus = corpus.shuffle(seed=42)
    corpus = corpus.batch(batch_size)
    batches = [
        tokenizer(
            x["text"],
            # todo use tokenizer conf
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        for x in corpus
    ]
    return batches


def load_batches_from_pairs_set(dataset, cfg):
    batch_size = cfg.train_batch_size
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    beginnings = []
    fulls = []
    for idx in range(cfg.num_examples_per_question):
        for q in dataset:
            beginnings.append(q["contexts"][idx])
            fulls.append(f"{q['contexts'][idx]} {q['answer_core']}")

    batches = []
    for i in range(0, len(beginnings), batch_size):
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, **cfg.tokenizer)
        full_batch = tokenizer(f_txt, **cfg.tokenizer)
        full_batch["answer_mask"] = prepare_answer_mask(beginning_batch, full_batch)

        batches.append(full_batch)

    return batches


def load_recall_batches(questions, cfg, batch_size=1):
    # batch_size 1 if slower but recommended, because it means we will first average
    # loss per answer and then across answers, which is more stable given some answers
    # are shorter than others
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    beginnings = []
    fulls = []
    for q in questions:
        beginning = f"""\
    {q["question"].strip()}
    Answer:"""
        ending = q["choices"][q["answer"]]
        full = f"{beginning} {ending}"
        beginnings.append(beginning)
        fulls.append(full)

    batches = []
    for i in range(0, len(beginnings), batch_size):
        b_txt = beginnings[i : i + batch_size]
        f_txt = fulls[i : i + batch_size]
        beginning_batch = tokenizer(b_txt, **cfg.tokenizer)
        full_batch = tokenizer(f_txt, **cfg.tokenizer)
        full_batch["answer_mask"] = prepare_answer_mask(beginning_batch, full_batch)

        batches.append(full_batch)

    return batches


def load_batches_from_simple_set(dataset, cfg):
    batch_size = cfg.train_batch_size
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    texts = []
    for idx in range(cfg.num_examples_per_question):
        for q in dataset:
            texts.append(q["sentences"][idx])

    batches = []
    for i in range(0, len(texts), batch_size):
        txt = texts[i : i + batch_size]
        batch = tokenizer(txt, **cfg.tokenizer)
        # batch["answer_mask"] = ...
        batches.append(batch)

    return batches


