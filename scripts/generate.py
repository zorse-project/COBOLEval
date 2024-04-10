import json
import os
from typing import List

import dotenv
import marko
import openai
from joblib import Memory
from loguru import logger
from marko.block import FencedCode
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils import Model, cleanup_dylib, cmd


dotenv.load_dotenv()


memory = Memory("cache", verbose=0)


OPENAI_SYSTEM_PROMPT = """```
{}
```

Complete the above program. It should consist of a single markdown code block following on from the lines above until the end of the program. It should terminate with `GOBACK`."""


@memory.cache
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def chat(messages, model) -> str:
    response = openai.chat.completions.create(
        model=model.name,
        messages=messages,
        temperature=model.temp,
    )
    return response.choices[0].message.content


def extract_code_block(src: str) -> List[str]:
    """
    Extract the first code block from markdown source
    """
    markdown = marko.parse(src)

    def search_for_code(element, code_blocks):
        if isinstance(element, FencedCode):
            code_blocks.append(element.children[0].children)
        elif hasattr(element, "children"):
            for child in element.children:
                search_for_code(child, code_blocks)

    code_blocks = []
    search_for_code(markdown, code_blocks)

    if len(code_blocks) > 1:
        logger.warning("Too many code blocks")

    block = code_blocks[0]
    return block


def swap_sections(src: str) -> str:
    """
    Swap the Working Storage and Linkage Sections
    """
    working_storage, linkage, procedure, begin = [], [], [], []
    current_section = begin

    for line in src.split("\n"):
        stripped_line = line.strip().upper()
        if stripped_line.startswith("WORKING-STORAGE SECTION."):
            current_section = working_storage
        elif stripped_line.startswith("LINKAGE SECTION."):
            current_section = linkage
        elif stripped_line.startswith("PROCEDURE DIVISION"):
            current_section = procedure
            line = "       PROCEDURE DIVISION USING LINKED-ITEMS."
        current_section.append(line)

    return "\n".join(begin + working_storage + linkage + procedure)


class LLMGenerator:
    def __init__(self, model: Model):
        self.model = model
        self.output_path = f"./preds/{model.name}"
        self.solutions_path = os.path.join(self.output_path, "solutions")
        os.makedirs(self.solutions_path, exist_ok=True)

        with open("./data/CobolEval.jsonl", "r") as f:
            self.evals = [json.loads(line) for line in f.readlines()]

        self.total = 0
        self.compiled = 0
        self.samples = []

    def eval(self):
        for e in self.evals:
            name = f"{e['entry_point']}"
            path = os.path.join(self.solutions_path, f"{e['entry_point']}.cbl")
            for k in range(self.model.samples_per_task):
                try:
                    program = self.solve(e, k)

                    self.samples.append(
                        {"sample_id": k, "task_id": e["task_id"], "completion": program}
                    )

                    logger.info(program)

                    with open(path, "w+") as f:
                        f.write(program)

                    compiles = cmd(f"cobc -fformat=variable {path}")
                    if compiles:
                        self.compiled += 1
                        cleanup_dylib(name)

                    self.total += 1

                except Exception as e:
                    logger.error(e)

        with open(f"{self.output_path}/samples.jsonl", "w+") as f:
            for s in self.samples:
                f.write(json.dumps(s) + "\n")

        logger.info(f"Compiled {self.compiled} out of {self.total} programs")
        return self.compiled

    def solve(self, eval, sample_id=0):
        raise NotImplementedError


class OpenAIChat(LLMGenerator):
    def __init__(self, model: Model):
        super().__init__(model)

    def solve(self, eval, sample_id=0):
        sol = chat(
            [
                {
                    "role": "system",
                    "content": OPENAI_SYSTEM_PROMPT.format(eval["prompt"]),
                }
            ],
            self.model,
        )
        sol = extract_code_block(sol)
        program = self.construct(eval["prompt"], sol)
        return program

    def construct(self, prompt: str, sol: str):
        if sol.strip().startswith("WORKING-STORAGE SECTION."):
            sol = sol.replace("WORKING-STORAGE SECTION.", "")

        prog = f"{prompt}\n{sol}"
        return swap_sections(prog)


class JsonComplete(LLMGenerator):
    """
    Save a program generated left-to-right in the format:

    {"completion": "<COBOL_PROGRAM>"}
    """

    def __init__(self, model: Model, jsonl_path: str):
        super().__init__(model)
        for i, e in enumerate(self.evals):
            e["id"] = i

        if self.model.samples_per_task > 1:
            self.completions = [[] * self.model.samples_per_task]
            for k in range(self.model.samples_per_task):
                self.completions.append([])
                with open(jsonl_path.split(".")[0] + f"_seed_{k}.jsonl", "r") as f:
                    for line in f.readlines():
                        self.completions[k].append(json.loads(line))
        else:
            self.completions = [[]]

            with open(jsonl_path, "r") as f:
                for line in f.readlines():
                    self.completions[0].append(json.loads(line))

    def construct(self, prompt: str, name: str, sol: str):
        prog = f"{prompt}{sol}"
        prog = swap_sections(prog)
        name = name.upper().replace("_", "-")
        prog += f"\n       END PROGRAM {name}.\n"
        return prog

    def solve(self, eval, sample_id=0):
        sol = self.completions[sample_id][eval["id"]]["completion"]
        program = self.construct(eval["prompt"], eval["entry_point"], sol)
        return program


class JsonProgram(LLMGenerator):
    """
    Save an infilled program in the format:

    {"completion": "<COBOL_PROGRAM>"}
    """

    def __init__(self, model: Model, jsonl_path: str):
        super().__init__(model)
        for i, e in enumerate(self.evals):
            e["id"] = i

        if self.model.samples_per_task > 1:
            self.completions = [[] * self.model.samples_per_task]
            for k in range(self.model.samples_per_task):
                self.completions.append([])
                with open(jsonl_path.split(".")[0] + f"_seed_{k}.jsonl", "r") as f:
                    for line in f.readlines():
                        self.completions[k].append(json.loads(line))
        else:
            self.completions = [[]]

            with open(jsonl_path, "r") as f:
                for line in f.readlines():
                    self.completions[0].append(json.loads(line))

    def solve(self, eval, sample_id=0):
        return self.completions[sample_id][eval["id"]]["completion"]
    
@memory.cache
def hf_complete(prompt, model, tokenizer, max_length=1024, eos_token=None) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    outputs = model.generate(inputs, max_new_tokens=max_length, use_cache=True, do_sample=False, repetition_penalty=1.1)
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=False)[len(prompt):]
    if eos_token and text_output.endswith(eos_token):
        text_output = text_output[: -len(eos_token)]
    return text_output
    

class HuggingfaceComplete(LLMGenerator):
    """
    Completes WORKING-STORAGE then PROCEDURE DIVISION with local Huggingface model
    """

    def __init__(self, model: Model):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        super().__init__(model)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model.name, device_map="cuda", torch_dtype=torch.bfloat16, cache_dir="/runs/cache")
        if model.tokenizer:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.tokenizer)
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.name)

    def solve(self, eval, sample_id=0):
        sol = hf_complete(eval["prompt"], self.hf_model, self.hf_tokenizer, eos_token=self.model.eos_token)
        program = self.construct(eval["prompt"], sol)
        return program

    def construct(self, prompt: str, sol: str):
        if sol.strip().startswith("WORKING-STORAGE SECTION."):
            sol = sol.replace("WORKING-STORAGE SECTION.", "")

        prog = f"{prompt}\n{sol}"
        return swap_sections(prog)


class HuggingfaceInfill(LLMGenerator):
    """
    Infills WORKING-STORAGE SECTION then completes PROCEDURE DIVISION with local Huggingface model

    Infilling strategy described here: https://bloop.ai/blog/evaluating-llms-on-cobol
    """

    def __init__(self, model: Model):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        super().__init__(model)
        self.hf_model = AutoModelForCausalLM.from_pretrained(model.name, device_map="cuda", torch_dtype=torch.bfloat16, cache_dir="/runs/cache")
        if model.tokenizer:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.tokenizer)
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model.name)
        self.prefix_token = model.prefix_token
        self.middle_token = model.middle_token
        self.suffix_token = model.suffix_token

    def solve(self, eval, sample_id=0):
        prefix, suffix = self.get_prefix_suffix(eval["prompt"])
        prompt = f"{self.prefix_token}{prefix}{self.suffix_token}{suffix}"
        sol = hf_complete(prompt, self.hf_model, self.hf_tokenizer, eos_token=self.model.eos_token)
        
        try:
            procedure, working_storage = sol.split(self.middle_token)
        except:
            procedure, working_storage = sol, "       WORKING-STORAGE SECTION.\n\n"

        program = f"{prefix}{working_storage}{suffix}{procedure}"
        return program

    def get_prefix_suffix(self, src: str):
        """
        Split the prompt into prefix and suffix strings
        """
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("DATA DIVISION."):
                data_division = i
            if line.strip().startswith("LINKAGE SECTION."):
                linkage_section = i
            if line.strip().startswith(
                "* Complete the WORKING-STORAGE SECTION and the PROCEDURE DIVISION"
            ):
                complete = i

        prefix = "\n".join(lines[: data_division + 1]) + "\n"
        store = "      * Store the result in the RESULT variable and mark the end of your program with END PROGRAM\n"
        suffix = "\n".join(lines[linkage_section:complete] + [store])
        return prefix, suffix


if __name__ == "__main__":
    model = Model(name="gpt-4", samples_per_task=1)
    runner = OpenAIChat(model)
    runner.eval()
