import pandas as pd
from tqdm import tqdm
import concurrent.futures
import argparse
from dotenv import load_dotenv
from prompts import LLM_RESPONSE_PROMPT, LLM_JUDGE_PROMPT, GRADER_TEMPLATE, LLM_RESPONSE_PROMPT_DISTRACTORS
from jinja2 import Template

from llms import LLMClient
from utils import determine_model_family, save_results, process_evaluation_output, prepare_options

load_dotenv()

class LLMCalibratorEvaluator:
    """
    A class to evaluate LLM responses and their confidence calibration.
    
    This class handles getting responses from different LLM models (OpenAI or Groq),
    evaluating their answers against gold standards, and tracking confidence scores.
    """

    def __init__(self, model_name, input_file, results_dir, approach, start_index, end_index):
        """
        Initialize the evaluator with model and data parameters.

        Args:
            model_name (str): Name of the LLM model to use
            number_of_samples (int): Number of samples to evaluate
            input_file (str): Path to input dataset
            results_dir (str): Directory to save results
            approach (str): Evaluation approach ('normal' or 'distractors')
        """
        self.model_name = model_name
        self.start_index = start_index
        self.input_file = input_file
        self.results_dir = results_dir
        self.approach = approach
        self.end_index = end_index
        self.model_family = determine_model_family(model_name)
        self.data = pd.read_csv(self.input_file)
        self.result_dict = None
        self.final_sample_count = len(self.data)
        self.llm_client = LLMClient()

    def get_answer_and_confidence(self, question, options):
        """
        Get answer and confidence score for a given question.

        Args:
            question (str): The question to answer

        Returns:
            tuple: (answer, confidence_score)
        """
        if self.approach == "normal":
            template = Template(LLM_RESPONSE_PROMPT)
            message = template.render(question=question)
        elif self.approach == "distractors":
            template = Template(LLM_RESPONSE_PROMPT_DISTRACTORS)
            message = template.render(question=question, options=options)
            
        response = self.llm_client.get_llm_response(message, self.model_name, self.model_family)
        return response["answer"], response["confidence_score"]

    def evaluate_answer(self, question, gold_answer, generated_answer):
        """
        Evaluate generated answer against gold standard.

        Args:
            question (str): Original question
            gold_answer (str): Correct answer
            generated_answer (str): Model generated answer

        Returns:
            str: Evaluation label
        """
        template = Template(GRADER_TEMPLATE)
        message = template.render(
            question=question,
            target=gold_answer,
            predicted_answer=generated_answer
        )
        response = self.llm_client.get_llm_response(message, "gpt-4o-mini", "openai")
        return response["label"]
    
    def process_question(self, i):
        question = self.data["problem"].iloc[i]
        gold_answer = self.data["answer"].iloc[i]
        wrong_answers = [
            str(self.data["wrong_answer_1"].iloc[i]),
            str(self.data["wrong_answer_2"].iloc[i]),
            str(self.data["wrong_answer_3"].iloc[i])
        ]
        
        options = prepare_options(gold_answer, wrong_answers)
        answer, confidence = self.get_answer_and_confidence(question, options)
        label = self.evaluate_answer(question, gold_answer, answer)
        output = process_evaluation_output(label)

        return int(i), {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": answer,
            "confidence": int(confidence),
            "output": int(output)
        }

    def eval(self, workers=2):
        """
        Run evaluation on the dataset.
        
        Processes each question, gets model answers and confidence scores,
        evaluates correctness, and stores results.

        Args:
            workers (int): Number of worker threads to use for parallel processing
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.process_question, i) 
                      for i in range(self.start_index, self.end_index)]
            
            for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                          total=self.end_index - self.start_index)):
                index, result = future.result()
                results[index] = result
                tqdm.write(f"Processed question {i + 1}/{self.end_index - self.start_index}")

                if (i + 1) % 2 == 0:
                    self.result_dict = results
                    save_results(results, self.results_dir, self.model_name, 
                               self.approach, self.start_index, self.end_index)

        self.result_dict = results
        save_results(results, self.results_dir, self.model_name, 
                    self.approach, self.start_index, self.end_index)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate LLM responses and confidence calibration")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Name of the LLM model")
    parser.add_argument("--end_index", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--input_file", type=str, default="dataset/simpleQA_with_distractors.csv", help="Input dataset path")
    parser.add_argument("--results_dir", type=str, default="results/", help="Directory to save results")
    parser.add_argument("--approach", type=str, default="normal", help="Evaluation approach (normal or distractors)")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker threads")
    
    args = parser.parse_args()

    evaluator = LLMCalibratorEvaluator(args.model_name, args.input_file, args.results_dir, args.approach, args.start_index, args.end_index)
    evaluator.eval(workers=args.workers)