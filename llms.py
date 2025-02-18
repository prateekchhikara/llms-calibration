from openai import OpenAI
from groq import Groq
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import os

class LLMClient:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=60, max=60))
    def get_llm_response(self, base_prompt, model_name, model_family):
        """
        Get response from LLM with retry logic for API failures.

        Args:
            base_prompt (str): The prompt to send to the LLM
            model_name (str): Name of the model to use
            model_family (str): Family of the model (openai or groq)

        Returns:
            dict: Parsed JSON response from the LLM
        """
        if model_family == "openai":
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": base_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        
        if model_family == "groq":
            response = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": base_prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content) 