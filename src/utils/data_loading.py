# %%
from datasets import load_dataset
from transformers import AutoTokenizer

from utils.git_and_reproducibility import repo_root

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
def load_batches(corpus, model_id, batch_size=4, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    corpus = corpus.shuffle(seed=42)
    corpus = corpus.batch(batch_size)
    batches = [
        tokenizer(
            x["text"],
            max_length=max_length,
            padding=True,
            # padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        for x in corpus
    ]
    return batches


def load_fineweb_edu_corpus():
    corpus = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        # just the smallest parquet file will be enough
        data_files=["sample/10BT/013_00000.parquet"],
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
