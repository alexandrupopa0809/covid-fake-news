import json
import logging
import os

import openai
from dotenv import load_dotenv

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


def write_to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_gpt_completion(system_prompt, prompt, model="gpt-4o-mini"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    content = response.choices[0].message.content
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    logging.info(
        "Total tokens used {usage['total_tokens']} - input:"
        f"{usage['prompt_tokens']}, output: {usage['completion_tokens']}"
    )
    logging.info(f"Returned text: [{content}]")
    return {
        "content": content,
        "usage": usage,
    }
