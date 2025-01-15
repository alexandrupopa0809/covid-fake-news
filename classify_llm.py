import json
import logging
import os

import openai
from dotenv import load_dotenv

from prompts import PROMPTS_OBJS
from utils import get_gpt_completion, read_json, write_to_json

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


def get_results(data, claims):
    for idx, prompt_obj in enumerate(PROMPTS_OBJS):
        results = []
        correct_preds = 0
        total_preds = 0
        invalid_preds = 0

        for _, article in enumerate(data, start=1):
            try:
                if idx % 2 == 0:
                    user_prompt = (
                        prompt_obj["user_prompt"]
                        + f" \n### Title: {article['title']} ###"
                    )
                else:
                    user_prompt = (
                        prompt_obj["user_prompt"]
                        + f" \n### Article: {article['text']} ###"
                    )
                    if article["text"] == "necunoscut":
                        continue

                if prompt_obj["claims"]:
                    user_prompt += f"\n\n\nAcesta este adevarul despre COVID-19: {claims}"

                resp = get_gpt_completion(prompt_obj["system_prompt"], user_prompt)
                predicted_obj = json.loads(resp["content"])
                predicted_label = predicted_obj["predicted_label"]
                actual_label = article["label"]
                reason = (
                    predicted_obj["reason"]
                    if "reason" in predicted_obj
                    else "necunoscut"
                )

                if predicted_label == actual_label:
                    correct_preds += 1
                total_preds += 1

                results.append(
                    {
                        "title": article["title"],
                        "predicted_label": predicted_label,
                        "actual_label": actual_label,
                        "reason": reason,
                    }
                )

            except Exception as e:
                logging.info(f"Invalid JSON format. Details: {str(e)}")
                invalid_preds += 1
        results.append(
            {
                "accuracy": correct_preds / total_preds if total_preds > 0 else 0,
                "total_preds": total_preds,
                "num_invalid_predictions": invalid_preds,
            }
        )
        write_to_json(results, f"results/{prompt_obj['output_file']}")


if __name__ == "__main__":
    data = read_json("data/bi_articles_ds.json")
    with open("data/source.txt", "r") as f:
        claims = f.read()
    get_results(data, claims)
