"""
RAG query system using SBERT embeddings, FAISS, and SQLite.
"""

import sqlite3
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple
import sys


FAISS_INDEX_PATH = "./recipe_faiss_index_full.idx"
METADATA_DB_PATH = "./recipe_metadata.sqlite3"
MODEL_NAME = "all-MiniLM-L6-v2"


class ModalRecipeRAG:
    _global_instance = None

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.db_connection = None
        self.is_loaded = False

    @classmethod
    def get_instance(cls):
        if cls._global_instance is None:
            cls._global_instance = ModalRecipeRAG()
        return cls._global_instance

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            print("Loading model", flush=True)
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded", flush=True)
            return True
        except Exception as e:
            print(f"Error loading model: {e}", flush=True)
            return False

    def load_index(self) -> bool:
        if not os.path.exists(FAISS_INDEX_PATH):
            print(f"FAISS index not found at {FAISS_INDEX_PATH}")
            return False

        if not os.path.exists(METADATA_DB_PATH):
            print(f"Metadata database not found at {METADATA_DB_PATH}")
            return False

        try:
            print("Loading FAISS index", flush=True)
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            print(f"Loaded {self.index.ntotal} recipes", flush=True)

            self.db_connection = sqlite3.connect(METADATA_DB_PATH, check_same_thread=False)
            self.db_connection.row_factory = sqlite3.Row

            cursor = self.db_connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM recipes")
            count = cursor.fetchone()[0]
            print(f"Connected to database with {count} recipes", flush=True)

            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading: {e}", flush=True)
            return False

    def search_recipes(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.model is None:
            if not self.load_model():
                return []

        if not self.is_loaded:
            if not self.load_index():
                return []

        try:
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = min(64, max(1, int(self.index.nprobe)))

            print(f"Searching: {query}", flush=True)
            query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            distances, indices = self.index.search(query_embedding, top_k)
            distances = distances[0]
            indices = indices[0]

            results = []
            cursor = self.db_connection.cursor()

            for idx, (recipe_id, distance) in enumerate(zip(indices, distances)):
                similarity_score = float(distance)
                cursor.execute(
                    "SELECT id, title, ingredients_json, link, source FROM recipes WHERE id = ?",
                    (int(recipe_id),)
                )
                row = cursor.fetchone()

                if row:
                    try:
                        ingredients = json.loads(row['ingredients_json'])
                    except Exception:
                        ingredients = []

                    results.append({
                        'recipe_id': row['id'],
                        'title': row['title'],
                        'ingredients': ingredients,
                        'link': row['link'],
                        'source': row['source'],
                        'similarity_score': similarity_score,
                    })

            return results

        except Exception as e:
            print(f"Error searching: {e}", flush=True)
            return []

    def close(self):
        if self.db_connection:
            self.db_connection.close()


def search_recipes_cli(ingredients_list: list, top_k: int = 5):
    rag = ModalRecipeRAG.get_instance()
    query = f"I have these ingredients: {', '.join(ingredients_list)}. What recipes can I make?"
    return rag.search_recipes(query, top_k=top_k)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python rag_modal_query.py 'ingredient1, ingredient2' [top_k]", flush=True)
        return

    if sys.argv[1] == "--interactive":
        rag = ModalRecipeRAG.get_instance()
        while True:
            user_input = input("Ingredients (comma-separated): ").strip()
            if user_input.lower() in ("quit", "exit"):
                break
            if not user_input:
                continue
            ingredients = [ing.strip() for ing in user_input.split(',') if ing.strip()]
            top_k = int(input("Top K results [5]: ") or 5)
            results = search_recipes_cli(ingredients, top_k)
            if results:
                for i, recipe in enumerate(results, 1):
                    print(f"\n{i}. {recipe['title']}", flush=True)
                    print(f"   Score: {recipe['similarity_score']:.4f}", flush=True)
                    if recipe['link']:
                        print(f"   Link: {recipe['link']}", flush=True)
            else:
                print("No recipes found.", flush=True)
            print("\n" + "=" * 70, flush=True)
        rag.close()
        return

    ingredients_str = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ingredients = [ing.strip() for ing in ingredients_str.split(',') if ing.strip()]

    results = search_recipes_cli(ingredients, top_k)

    if results:
        for i, recipe in enumerate(results, 1):
            print(f"\n{i}. {recipe['title']}", flush=True)
            print(f"   Score: {recipe['similarity_score']:.4f}", flush=True)
            if recipe['link']:
                print(f"   Link: {recipe['link']}", flush=True)
    else:
        print("No recipes found.", flush=True)


if __name__ == "__main__":
    main()
