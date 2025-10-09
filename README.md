# RLLMGraphColoring

This repository hosts the code used to run the experiments in the paper "Reasoning Large Language Model Errors Arise from Hallucinating Critical Problem Features", as well as records of the problems, prompts, and responses involved in those experiments. This repository is a modified version of [the one for a related previous paper](https://github.com/AlexHeyman/LLMGraphColoring).

## Requirements

This codebase was developed for Python 3.11.7, with the packages Anthropic 0.49.0, Fireworks AI 0.15.3, Google Generative AI 0.7.2, and OpenAI 1.70.0 used for interfacing with RLLMs (see `models.py`), NumPy 2.3.1 used for managing stable matching preference matrices, and SciPy 1.16.0 and Matplotlib 3.9.2 used for generating plots (see `plot_problem_set.py` and `summarize.py` in both the `graph_coloring` and `stable_matching` directories). Earlier or later versions of the requirements may or may not work.

## Usage

The following files are executable code files with different functions (see their respective header docstrings for details):

* `test.py` in the main directory
* `generate.py`, `select_problems.py`, `categorize.py`, `plot_problem_set.py`, `calculate_uncolorability_distances.py`, `delete_not_found.py`, `evaluate.py`, `evaluate_response_features.py`, and `summarize.py` in the `graph_coloring` directory
* `generate.py`, `plot_problem_set.py`, `assess_grouping.py`, `evaluate.py`, `evaluate_response_features.py`, and `summarize.py` in the `stable_matching` directory

Also see `graph_coloring/metadata.py` and `stable_matching/metadata.py` for important parameters for their respective directories' executable files.

Note that the compressed data archives in `graph_coloring/data`, `graph_coloring/prompts`, and `graph_coloring/responses` must be uncompressed before the directory's executable code files can operate on the data properly, and likewise for the archives in `stable_matching/data`, `stable_matching/prompts`, and `stable_matching/responses`.