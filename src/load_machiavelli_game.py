# %%
"""
Script to load and explore Machiavelli game data.
"""

import json
from typing import Counter
from utils.git_and_reproducibility import repo_root


def load_game_metadata():
    metadata_file = repo_root() / "game_data" / "game_metadata.json"
    with open(metadata_file, 'r') as f:
        return json.load(f)


def load_game_data(game_name="dragon"):
    """Load scenarios and annotations for a specific game."""
    
    # Get the game data directory
    game_data_dir = repo_root() / "game_data"
    
    # Load scenarios
    scenarios_file = game_data_dir / "scenarios" / f"{game_name}_scenarios_bfs_filled.json"
    with open(scenarios_file, 'r') as f:
        scenarios = json.load(f)
    
    # Load annotations
    annotations_file = game_data_dir / "annotations_clean" / f"{game_name}.json"
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    # Load trees
    trees_file = game_data_dir / "trees" / f"{game_name}_game_tree.json"
    with open(trees_file, 'r') as f:
        trees = json.load(f)
    
    return scenarios, annotations["scenes"], trees


def get_situations_with_trait(trait="deception", game_name="deathless"):
    scenarios, annotations, trees = load_game_data(game_name)
    scene_ids = list(scenarios.keys())

    for scene_id in scene_ids:
        if scene_id not in trees or scene_id not in annotations:
            continue

        # * use only the scenes with given trait
        bad_trait_value = annotations[scene_id]["morality"][trait]
        if bad_trait_value != 1:
            # values go from 0 to 1, but let's use only the maximal ones, with 1
            continue

        # * skip cases where the type is OTHER
        # apart from some initial nodes, they don't contain a choice, it seems
        if trees[scene_id]["type"] == "OTHER":
            continue


        # * get the chain of parent texts
        parent_texts_chain = []
        parent_ids_chain = []
        cursor = scene_id
        while True:
            parent_ids = trees[cursor]["parents"]
            parent_ids = [p for p in parent_ids if "_collapsed" not in p]
            if not parent_ids:
                break

            # we found a valid parent, so move up the tree
            # choose the parent with the lowest line number
            parent_ids.sort(key=lambda x: int(x.split("_")[-1]))
            cursor = parent_ids[0]

            if cursor in scenarios:  # some of the texts are empty, so we just move through them
                parent_texts_chain.append(scenarios[cursor])
            parent_ids_chain.append(cursor)

        parent_texts_chain.reverse()
        parent_ids_chain.reverse()
        # * if no valid parents, skip
        if len(parent_texts_chain) == 0:
            continue
        
        # * get the sibling texts without the bad trait
        parent_id = parent_ids_chain[-1]
        sibling_ids = trees[parent_id]["children"].copy()
        sibling_ids.remove(scene_id)
        # filter out absent siblings
        sibling_ids = [s for s in sibling_ids if s in annotations and s in scenarios]
        # filter out siblings with the bad trait
        sibling_ids = [s for s in sibling_ids if annotations[s]["morality"][trait] == 0]
        sibling_texts = [scenarios[s_id] for s_id in sibling_ids if s_id in scenarios]

        # * get the current scene text
        text = scenarios[scene_id]

        # note, only the first lines in text and sibling_texts are the action
        # also the parent_texts_chain can be really long, so only take the last N items
        yield parent_texts_chain, text, sibling_texts

# %%
trait = "deception"
game_name = "deathless"
use_last_n_parents = 5

corpus = []
for parent_texts_chain, text, sibling_texts in get_situations_with_trait(trait, game_name):
    item = dict(
        context = "\n".join(parent_texts_chain[-use_last_n_parents:]),
        answer = text.splitlines()[0],
        alt_answers = [s.splitlines()[0] for s in sibling_texts],
    )
    print(item)
    print()
    corpus.append(item)

# * save the corpus
path = repo_root() / "data" / "machiavelli" / trait / f"{game_name}.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, 'w') as f:
    for item in corpus:
        f.write(json.dumps(item) + "\n")
# %%
