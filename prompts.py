PROMPTS_OBJS = [
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea titlurilor stirilor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un titlu de stire referitoare la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. Acest titlu este delimitat de 3 semne #.

            Vei intoarce doar un singur obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake".
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_titles_only.json",
        "claims": False,
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea articolelor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un articol referitor la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. Acest articol este delimitat de 3 semne #

            Vei intoarce doar un singur obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake".
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_texts_only.json",
        "claims": False,
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea titlurilor stirilor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un titlu de stire referitoare la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. Acest titlu este delimitat de 3 semne #.
            De asemenea la finalul promptului, ca ajutor vei primi si un text care reprezinta adevarul despre COVID-19.
            Te poti ajuta de acest text, dar si de alte informatii pe care deja le cunosti.

            Vei intoarce doar un singur obiect json cu cheile: "predicted_label", care poate avea valorile "real" sau "fake"
            si cheia "reason", care reprezinta motivul pentru care stirea este falsa sau adevarata.
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_titles_claims.json",
        "claims": True,
    },
    {
        "system_prompt": """
            Esti un asistent util pentru intelegerea articolelor referitoare la COVID-19.
        """,
        "user_prompt": """
            Vei primi un articol referitor la pandemia de COVID-19 si trebuie sa spui daca
            stirea respectiva este reala sau falsa. Acest articol este delimitat de 3 semne #.
            De asemenea la finalul promptului, ca ajutor vei primi si un text care
            reprezinta adevarul despre COVID-19. Te poti ajuta de acest text, dar si de alte informatii
            pe care deja le cunosti.

            Vei intoarce doar un singur obiect json cu cheia: "predicted_label", care poate avea valorile "real" sau "fake"
            si cheia "reason", care reprezinta motivul pentru care stirea este falsa sau adevarata.
            Trebuie sa intorci doar obiectul json. Nu folosi elemente de markdown!
        """,
        "output_file": "res_articles_claims.json",
        "claims": True,
    },
]

"""
1. Rezultate zero-shot doar cu titlul.
2. Rezultate zero-shot doar cu articol.
3. Rezultate cu titlul + tot setul de claim-uri.
4. Rezultate cu articolul + tot setul de claim-uri.
"""
