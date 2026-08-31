"""
Small generic cache: save/load any picklable object under a string key.
Used to avoid recomputing tile resizing (and later, CNN embeddings) every run.
"""

import os
import pickle


def cache_path(cache_dir, key):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pkl")


def load(cache_dir, key):
    path = cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def save(cache_dir, key, obj):
    path = cache_path(cache_dir, key)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path
