"""Run this script to ingest Wikipedia pages listed in `data/wikipedia_pages.txt`
and build a persisted Llama-Index in `data/index`.
"""
from pathlib import Path
from api.service.llama_index_updater import build_index_from_titles

with open("concepts.txt", "r") as f:
    concepts = [line.strip() for line in f.readlines() if line.strip()]

print("Concepts to ingest:", len(concepts))

def main():
    # titles = load_wikipedia_page_titles("data/wikipedia_pages.txt")
    build_index_from_titles(concepts, index_path="data/index")


if __name__ == "__main__":
    main()
