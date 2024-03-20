# COBOLEval: LLM Evaluation for COBOL

COBOLEval is a dataset to evaluate the code generation abilities of Large Language Models on the COBOL programming language. It is a transpilation of the widely-used [HumanEval](https://github.com/openai/human-eval) benchmark from Python into COBOL. This repo contains both the Python to COBOL transpiler, and an evaluation harness for the dataset.

## Installation

COBOLEval uses [GnuCOBOL](https://gnucobol.sourceforge.io/) to compile the generated COBOL solutions. Download version 3.2.0 here and follow the installation instructions: https://sourceforge.net/projects/gnucobol/files/.

Check that the installation was successful with:

```
>>> cobc -v
cobc (GnuCOBOL) 3.2.0
```

Using Python3.10 or later:

```
python -m venv coboleval
source coboleval/bin/activate
pip install -r requirements.txt
```

To run the Python to COBOL transpiler, you'll need to [install Rust](https://www.rust-lang.org/tools/install).

## Usage

**This program runs untrusted model-generated code. Users are strongly encouraged not to do so outside of a robust security sandbox. Following HumanEval, the [execution call](./scripts/evaluation.py#L100) in `evaluation.py` is deliberately commented out to ensure users read this disclaimer before running code in a potentially unsafe manner.**

### Generate completions

Configure the model and the number of samples-per-problem in `scripts/generate.py` then run. 

```python
if __name__ == "__main__":
    model = Model(name="gpt-4", samples_per_task=1)
    runner = OpenAIChat(model)
    runner.eval()
```

This will create a `samples.jsonl` file in `preds/gpt-4` which contains the generated COBOL solutions.

### Calculate Pass@k

Configure the model and the number of samples in the `entrypoint()` function in `scripts/evaluate_functional_correctness.py`:

```python
def entrypoint():
    all_results = []
    run_folders = ["gpt-4"]  # edit
    for folder in run_folders:
        all_results.append(eval(f"preds/{folder}", "1"))

    for res, folder in zip(all_results, run_folders):
        print(f"{folder}: {res}")
```

Outputs are written to `preds/gpt-4/samples_results.jsonl` and Pass@k is printed:

```
gpt-4: {'pass@1': 0.10273972602739725}
```