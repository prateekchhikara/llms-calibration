import json
from jinja2 import Template
import random

def determine_model_family(model_name):
    """Determine the model family based on model name."""
    if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
        return "openai"
    elif model_name in ["llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"]:
        return "groq"
    else:
        raise ValueError(f"Model {model_name} not supported")

def save_results(results, results_dir, model_name, approach, start_index, end_index):
    """Save evaluation results to JSON file."""
    # sort the results by the index
    sorted_results = dict(sorted(results.items(), key=lambda x: int(x[0])))
    
    filename = f"{results_dir}{model_name}_{approach}_{start_index}_{end_index}_results.json"
    with open(filename, 'w') as f:
        json.dump(sorted_results, f, indent=2)

def process_evaluation_output(label):
    """Convert evaluation label to numeric output."""
    if label == "CORRECT" or label == "A":
        return 1
    elif label == "INCORRECT" or label == "B":
        return 0
    else:
        return -1

def prepare_options(gold_answer, wrong_answers):
    """Prepare randomized options string."""
    all_answers = [gold_answer] + wrong_answers
    return "\n- " + "\n- ".join(random.sample(all_answers, len(all_answers))) 