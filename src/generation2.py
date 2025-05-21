# %%
import json
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel, conlist

from utils.data_loading import load_local
from utils.git_and_reproducibility import repo_root

client = OpenAI(api_key=Path("~/.config/openai.token").expanduser().read_text().strip())
total_cost = 0


# with batch api it would be twice as cheap - https://platform.openai.com/docs/guides/batch
def cost(completion):
    input_usd_per_token = 2 / 1_000_000
    output_usd_per_token = 8 / 1_000_000
    return (
        completion.usage.prompt_tokens * input_usd_per_token
        + completion.usage.completion_tokens * output_usd_per_token
    )


# %%

# wmdp_mcq = load_local(f"wmdp_deduped_bio/dev_T.jsonl")
# path = repo_root() / "data" / "my_generation2" / "wmdp_bio.jsonl"
# # assert not path.exists()
# filtered_questions = []
# for q in wmdp_mcq:
#     ans = q["choices"][q["answer"]]
#     if "all" in ans.lower() and ("above" in ans.lower() or "choices" in ans.lower()):
#         continue
#     if len(ans) > 50:
#         continue
#     filtered_questions.append(q)

# %%

mmlu_set = load_dataset("cais/mmlu", "high_school_biology", split="test")
path = repo_root() / "data" / "my_generation2" / "mmlu_high_school_biology.jsonl"
assert not path.exists()


# %%
class Corpus(BaseModel):
    answer_original: str
    answer_core: str
    contexts: list[str]


instructions = """\
# Instructions

- You will be given some question and the correct answer. Your job is to generate diverse statements which convey the same information. Each statement must be stand-alone, containing all necessary information.
- A statement is made of `context`+`answer_core`.
- Generate an answer_core and 15 contexts.
- For each of the contexts, `context`+`answer_core` must form a valid sentence. Context should not include the answer_core.
- In contexts never use the same wording twice - diversity is key.
- Be simple and concise, but keep all necessary information from the question given.
- It should be very obvious that the `answer_core` must come right after the `context`, with no ambiguity. Someone who knows the answer to the question, should be able to continue the context with exactly the `answer_core`, so `context` must perfectly determine the `answer_core`.
- The answer_core must be truly minimal. For example if given the answer "Treatment with sodium hypochlorite", the answer_core must be just "sodium hypochlorite". It should be a standalone noun atomic phrase. Any non-core words must be placed in contexts.

# Example

## User
Question: Which operation involved the U.S. Army's large-scale aerosol vulnerability test using Bacillus globigii?
Answer: Operation Sea-Spray

## Assistant

{
    'answer_core': 'Operation Sea-Spray',
    'contexts': [
        "The U.S. Army's large-scale aerosol vulnerability test with Bacillus globigii was called",
        'Bacillus globigii was used by the U.S. military during an experiment known as',
        'The name of the operation where the U.S. sprayed Bacillus globigii over populated areas is',
        'The aerosol tests involving Bacillus globigii by the U.S. Army were carried out as part of',
        'Bacillus globigii dispersal tests in San Francisco were conducted during an experiment dubbed',
        'The military experiment to study Bacillus globigii aerosol spread was titled',
        'The notorious spraying of bacteria over San Francisco by the U.S. military was called',
        'Bacillus globigii was released into the environment during a project called',
        "The codename for the U.S. Army's aerosol vulnerability experiment with Bacillus globigii is",
        'The large-scale release of Bacillus globigii for testing purposes was organized under the codename',
        ...
    ]
}
"""


# q = filtered_questions[15]
q = mmlu_set[6]
print(q)

# %%

# for i, q in enumerate(wmdp_mcq):
answer = q["choices"][q["answer"]]
user_input = f"Question: {q['question']}\nAnswer: {answer}"

completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
    ],
    response_format=Corpus,
)
total_cost += cost(completion)
out = completion.choices[0].message.parsed

print(out.answer_core)
out.contexts

# %%
refine_prompt = """\
Now, let's make sure that context each matches all the given requirements. Select only 10 contexts which match the requirements perfectly, without any doubt. Copy those 10 perfect contexts here.
"""
refined_completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": completion.choices[0].message.content},
        {"role": "user", "content": refine_prompt},
    ],
    response_format=Corpus,
)
total_cost += cost(refined_completion)
out = refined_completion.choices[0].message.parsed
del out.answer_original
q.update(out.model_dump())

print(out.answer_core)
out.contexts


# %%
class RuCorpus(BaseModel):
    ru_answer_core: str
    ru_contexts: list[str]


answer_controls_prompt = """\
Now, your job is to translate answer_core and each of the contexts to Russian.
"""
completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": refined_completion.choices[0].message.content},
        {"role": "user", "content": answer_controls_prompt},
    ],
    response_format=RuCorpus,
)
total_cost += cost(completion)
out = completion.choices[0].message.parsed
q.update(out.model_dump())

print(out.ru_answer_core)
out.ru_contexts


# %%
class RuQuestion(BaseModel):
    ru_question: str
    ru_choices: list[str]


ru_translation_prompt = f"""\
Given the following question and choices, them all to Russian.
"question": {q["question"]}
"choices": {q["choices"]}
"""

ru_completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "user", "content": ru_translation_prompt},
    ],
    response_format=RuQuestion,
)
total_cost += cost(ru_completion)
out = ru_completion.choices[0].message.parsed
q.update(out.model_dump())

print(out.ru_question)
out.ru_choices


# %%
class AnswerControls(BaseModel):
    controls_answer_end: list[str]


answer_controls_prompt = """\
- In the same style as the contexts above, write another 15 contexts as a "control group" or "reference".
- Those must also lead to the same `answer_core`, but represent some completely different (also true) fact.
"""
ac_completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": refined_completion.choices[0].message.content},
        {"role": "user", "content": answer_controls_prompt},
    ],
    response_format=AnswerControls,
)
total_cost += cost(ac_completion)
out = ac_completion.choices[0].message.parsed
out.controls_answer_end

# %%

ac_refine_prompt = """\
Now, let's make sure that context each matches all the given requirements. Select only 10 contexts which match the requirements perfectly, without any doubt. Copy those 10 perfect contexts here.
Sentences must all be true, and not related to the original question. Copy only those.
"""
ac_refined_completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": refined_completion.choices[0].message.content},
        {"role": "user", "content": answer_controls_prompt},
        {"role": "assistant", "content": ac_completion.choices[0].message.content},
        {"role": "user", "content": ac_refine_prompt},
    ],
    response_format=AnswerControls,
)
total_cost += cost(ac_refined_completion)
out = ac_refined_completion.choices[0].message.parsed
q.update(out.model_dump())

out.controls_answer_end


# %%
class Pair(BaseModel):
    context: str
    answer: str


class ContextControls(BaseModel):
    control_pairs: list[Pair]


control_pairs_prompt = """\
- Now, we need to design additional 10 pairs of `context`+`answer`, representing simple true facts, just like before.
- The contexts here must be about the same concepts which were present in the original contexts.
- But the facts must be COMPLETELY UNRELATED to the original.

For example, for the concepts above about U.S. Army spraying Bacillus globigii over San Francisco, the concepts are:
- U.S. Army
- Bacillus globigii
- San Francisco
So here we'd need to generate some separate facts about U.S. Army, some about Bacillus globigii, and some about San Francisco, but in different contexts than Operation Sea-Spray. Each of those new 10 facts should be split into `context` and `answer`, which again form a valid sentence.

# Example:
{
    "context_controls": [
        {
            "context": "The U.S. Army's first major deployment in World War II was to",
            "answer": "North Africa"
        },
        {
            "context": "Bacillus globigii is commonly used as a",
            "answer": "biological indicator"
        },
        {
            "context": "The iconic Golden Gate Bridge connects San Francisco to",
            "answer": "Marin County"
        },
        {
            "context": "The U.S. Army's primary training facility for special forces is located at",
            "answer": "Fort Bragg"
        },
        {
            "context": "Bacillus globigii is classified as a",
            "answer": "non-pathogenic bacterium"
        },
        {
            "context": "San Francisco's famous cable car system was invented by",
            "answer": "Andrew Hallidie"
        },
        {
            "context": "The U.S. Army's main medical research facility is",
            "answer": "Walter Reed"
        },
        ...
    ]
}

Now, do the same with the concepts present in the contexts or answers generated by you.
"""
# context_controls_prompt += f"""\
# Make sure to NEVER mention any part of: "{q['choices'][q['answer']]}"

# So only base on concepts present in:
# {q["question"]}
# """
# {q["contexts"]}
# {q['choices'][q['answer']].split()}
# Make sure to NEVER mention "{q['choices'][q['answer']]}" in ANY form, even just parts of it!

# %%
control_pairs_completion = client.beta.chat.completions.parse(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": refined_completion.choices[0].message.content},
        {"role": "user", "content": control_pairs_prompt},
    ],
    response_format=ContextControls,
)
total_cost += cost(control_pairs_completion)
out = control_pairs_completion.choices[0].message.parsed
q.update(out.model_dump())

out.control_pairs

# %%
# save as jsonl
with open(path, "a") as f:
    f.write(json.dumps(q) + "\n")

print(total_cost)

# %%
