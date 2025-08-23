import os, re, requests
from bs4 import BeautifulSoup
from typing import List

def fetch_and_clean(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove nav/aside/script/style
    for bad in soup(["nav","aside","script","style","footer","header"]):
        bad.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def fetch_corpus(urls: List[str]) -> str:
    parts = []
    for u in urls:
        try:
            parts.append(f"### Source: {u}\n\n{fetch_and_clean(u)}")
        except Exception as e:
            print(f"[WARN] Failed {u}: {e}")
    return "\n\n".join(parts)
