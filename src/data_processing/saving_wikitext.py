# %%
import json

from datasets import load_dataset

from utils.git_and_reproducibility import repo_root

# huggingface stopped hosting wikitext (or has some technical issues)
# so I need to add my cached wikitext to the repo, for reliability

wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
wikitext = wikitext.filter(lambda x: x["text"])  # filter out empty texts

# %%
wikitext = wikitext.shuffle(seed=42).select(range(16_384))
# %%
# Save as JSONL instead of using save_to_disk
output_path = repo_root() / "data" / "wikitext_16k.jsonl"
with open(output_path, "w") as f:
    for item in wikitext:
        f.write(json.dumps(item) + "\n")
