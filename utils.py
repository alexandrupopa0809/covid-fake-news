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

PROMPTS_OBJS = [
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea titlurilor stirilor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un titlu de stire referitoare la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa.

            Vei intoarce un obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake".
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_titles_only.json",
        "claims": False
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea articolelor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un articol referitor la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa.

            Vei intoarce un obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake".
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_texts_only.json",
        "claims": False
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea titlurilor stirilor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un titlu de stire referitoare la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. De asemenea, ca ajutor vei primi si un text care
            reprezinta adevarul despre COVID-19. Te poti ajuta de acest text, dar si de alte informatii
            pe care deja le cunosti.

            Vei intoarce un obiect json cu cheile: "predicted_label", care poate avea valorile "real" sau "fake"
            si cheia "reason", care reprezinta motivul pentru care stirea este falsa sau adevarata.
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_titles_claims.json",
        "claims": True
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea articolelor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un articol referitor la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. De asemenea, ca ajutor vei primi si un text care
            reprezinta adevarul despre COVID-19. Te poti ajuta de acest text, dar si de alte informatii
            pe care deja le cunosti.

            Vei intoarce un obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake"
            si cheia "reason", care reprezinta motivul pentru care stirea este falsa sau adevarata.
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_articles_claims.json",
        "claims": True
    },
]

"""
1. Rezultate zero-shot doar cu titlul.
2. Rezultate zero-shot doar cu articol.
3. Rezultate cu titlul + tot setul de claim-uri.
4. Rezultate cu articolul + tot setul de claim-uri.
"""