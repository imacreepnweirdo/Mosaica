"""
Finds candidate tile images in a folder and fingerprints the folder's
contents so caches can detect when they're stale.
"""

import os
import hashlib

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_images(folder):
    """Sorted list of full paths to supported images directly inside `folder`."""

    names = sorted(
        n for n in os.listdir(folder) if n.lower().endswith(SUPPORTED_EXTENSIONS)
    )
    return [os.path.join(folder, n) for n in names]


def folder_signature(folder):
    """
    A short hash representing "what's in this folder right now": filenames,
    sizes, and modification times. Used as a cache key — if you add, remove,
    or overwrite a photo, this changes and the cache is rebuilt automatically.
    """

    entries = []
    for path in find_images(folder):
        stat = os.stat(path)
        entries.append(f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime)}")

    signature = "|".join(entries)
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()
