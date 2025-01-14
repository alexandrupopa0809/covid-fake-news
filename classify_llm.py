import json
import logging
import os

import openai
from dotenv import load_dotenv
from utils import read_json, write_to_json, get_gpt_completion, PROMPTS_OBJS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
openai.api_key = OPENAI_API_KEY

logging.basicConfig(
    level=logging.INFO,
    filename="time_logs.log",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_results(data):
    for prompt_obj in PROMPTS_OBJS:
        results = []
        for idx, article in enumerate(data):
            try:
                if idx % 2 == 0:
                    user_prompt = prompt_obj + f"\Title: {article['title']}"
                else:
                    user_prompt = prompt_obj + f"\nArticle: {article['text']}"
                
                # TODO: add/read claims
                if prompt_obj["claims"]:
                    user_prompt += f"\nAdevarul despre COVID-19: {''}"

                resp = get_gpt_completion(prompt_obj["system_prompt"], user_prompt)
                predicted_obj = json.loads(resp["content"])
                predicted_label = predicted_obj["predicted_label"]
                actual_label = article["label"]
                reason = predicted_obj["reason"] if "reason" in predicted_obj else "necunoscut"

                # TODO: compute accuracy

            except Exception as e:
                logging.info(f"Invalid JSON format. Details: {str(e)}")


if __name__ == "__main__":
    # TODO 1: add path
    data = read_json()


"""
1. Rezultate zero-shot doar cu titlul.
2. Rezultate zero-shot doar cu articol.
3. Rezultate cu titlul + tot setul de claim-uri.
4. Rezultate cu articolul + tot setul de claim-uri.
"""