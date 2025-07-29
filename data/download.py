# llm/data/download.py

import os
import urllib.request

def download_text(path: str) -> str:
    if not os.path.exists(path):
        with urllib.request.urlopen(path) as response:
            text_data = response.read().decode('utf-8')
        with open(path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(path, "r", encoding="utf-8") as file:
            text_data = file.read()
    return text_data
