"""Pinned OPUS Books bilingual corpus reader with document-grouped train/dev/test.

Research-only. No raw copyright-restricted data are committed or uploaded.
"""
from __future__ import annotations

import hashlib
import io
import random
from pathlib import Path
import zipfile

BOOKS = {
    "french": {
        "pair": "en-fr", "source": "fr",
        "url": "https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-fr.txt.zip",
        "sha256": "3b9eade0b11d6cc280502cbe52908ec476d9faa3bb589e70cf3ee1e44fcfc473",
        "bytes": 12009501,
    },
    "spanish": {
        "pair": "en-es", "source": "es",
        "url": "https://object.pouta.csc.fi/OPUS-Books/v1/moses/en-es.txt.zip",
        "sha256": "81917fa2df54b0472434d9eae7d1dc54556d5d0bf6ea265ac5a9edb30e7acfc2",
        "bytes": 9257150,
    },
}
SEED = 202611171
TRAIN_COUNT = 1536
DEV_COUNT = 128
TEST_COUNT = 256


def norm(s: str) -> str:
    return " ".join(s.casefold().strip().split())


def read_parallel(data: bytes, language: str):
    if language not in BOOKS:
        raise ValueError("Unsupported language")
    spec = BOOKS[language]
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("OPUS Books archive digest mismatch; aborting rather than changing data silently")
    if len(data) != spec["bytes"]:
        raise ValueError("OPUS Books archive byte count mismatch")
    return _parse_archived(data, spec)


def _parse_archived(data: bytes, spec: dict):
    prefix = "Books." + spec["pair"]
    wanted = [prefix + "." + ext for ext in (spec["source"], "en", "ids")]
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = {}
        for item in z.infolist():
            short = Path(item.filename).name
            if short in wanted:
                if short in files or item.file_size > 100_000_000:
                    raise ValueError("Duplicate or unexpectedly large OPUS member: " + short)
                files[short] = item.filename
        if set(files) != set(wanted):
            raise ValueError("Missing OPUS Books aligned text or document identifiers")
        src, ref, ids = [z.read(files[name]).decode("utf-8-sig").splitlines()
                         for name in wanted]
    if not (len(src) == len(ref) == len(ids)):
        raise ValueError("OPUS Books parallel lines and document metadata are not aligned")

    grouped, seen_src, seen_ref = {}, set(), set()
    for source, english, meta in zip(src, ref, ids):
        source, english = source.strip(), english.strip()
        if not (3 <= len(source.split()) <= 32 and 3 <= len(english.split()) <= 36):
            continue
        s, e = norm(source), norm(english)
        if s in seen_src or e in seen_ref:
            continue
        parts = meta.split()
        if len(parts) < 2:
            continue
        doc = "|".join(parts[:2])
        grouped.setdefault(doc, []).append((source, english))
        seen_src.add(s); seen_ref.add(e)
    if len(grouped) < 3:
        raise ValueError("Need at least three separately identifiable books")
    return grouped


def split_books(grouped, *, seed=SEED, train_count=TRAIN_COUNT,
                dev_count=DEV_COUNT, test_count=TEST_COUNT):
    """Avoid cross-document leakage, even where a book has many aligned fragments."""
    if len(grouped) < 3 or not all(grouped.values()):
        raise ValueError("Need at least 3 nonempty book groups")
    groups = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(groups)
    total = sum(len(grouped[k]) for k in groups)
    if total < train_count + dev_count + test_count:
        raise ValueError("Insufficient OPUS Books examples after filtering")

    held = {"dev": [], "test": []}
    reserved = {"dev": set(), "test": set()}
    remaining = total
    for key in groups:
        count = len(grouped[key])
        # Keep the training pool large enough even when a book is extremely large.
        for kind, need, keep in (("dev", dev_count, test_count),
                                 ("test", test_count, 0)):
            if len(held[kind]) < need and remaining - count >= train_count + keep:
                held[kind].extend(grouped[key])
                reserved[kind].add(key)
                remaining -= count
                break
    if len(held["dev"]) < dev_count or len(held["test"]) < test_count:
        raise ValueError("Not enough document-disjoint development and test material")

    train_pool = [row for key in groups if key not in reserved["dev"] | reserved["test"]
                  for row in grouped[key]]
    if len(train_pool) < train_count:
        raise ValueError("Not enough training material outside held-out books")
    rng.shuffle(train_pool)
    for kind in held:
        rng.shuffle(held[kind])
    train, dev, test = (train_pool[:train_count],
                        held["dev"][:dev_count], held["test"][:test_count])
    def identity(rows, column):
        return {norm(row[column]) for row in rows}
    for col in (0, 1):
        assert not identity(train, col) & identity(dev, col)
        assert not identity(train, col) & identity(test, col)
        assert not identity(dev, col) & identity(test, col)
    result = {
        "train": train, "dev": dev, "test": test,
        "book_groups_total": len(grouped),
        "doc_group_sha256": {
            kind: [hashlib.sha256(s.encode("utf-8")).hexdigest()
                   for s in sorted(reserved[kind])]
            for kind in reserved
        },
    }
    return result
