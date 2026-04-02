"""Semantic chunking and atomic FAISS/JSON persistence without pickle."""
import hashlib
import json
import re
from uuid import uuid4
import warnings
from zipfile import BadZipFile

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INDEX_SETTINGS = {"schema": 1, "algorithm": "sentence-window-cosine-v1",
                  "buffer": 1, "percentile": 85, "max_chars": 1000, "overlap": 100}


def semantic_chunks(text, embeddings):
    """Split at large changes between adjacent sentence-window embeddings.

    Sentence windows supply local context. A final character limit bounds long
    scenes and texts with no sentence punctuation. No source sentences are dropped.
    """
    settings = INDEX_SETTINGS
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings["max_chars"],
                                               chunk_overlap=0)
    sentences = [piece for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
                 for piece in splitter.split_text(sentence) if piece.strip()]
    if len(sentences) <= 1:
        return sentences
    buffer = settings["buffer"]
    windows = [" ".join(sentences[max(0, i-buffer):i+buffer+1])
               for i in range(len(sentences))]
    vectors = np.asarray(embeddings.embed_documents(windows), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(windows) or not np.isfinite(vectors).all():
        raise ValueError("Invalid semantic embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)
    distances = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distances, settings["percentile"])
    groups, current = [], []
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        if i == len(sentences)-1 or distances[i] > threshold:
            groups.append(" ".join(current))
            current = []
    fallback = RecursiveCharacterTextSplitter(chunk_size=settings["max_chars"],
                                               chunk_overlap=settings["overlap"])
    return [chunk for group in groups for chunk in fallback.split_text(group)]


def build_index(text, embeddings, source_hash):
    chunks = semantic_chunks(text, embeddings)
    metadata = [{"chunk_id": i, "source_sha256": source_hash} for i in range(len(chunks))]
    return FAISS.from_texts(chunks, embeddings, metadatas=metadata)


def save_index(path, identity, store):
    """Publish one complete archive atomically; partial writes are never loaded."""
    data = faiss.serialize_index(store.index)
    docs = [store.docstore.search(store.index_to_docstore_id[i])
            for i in range(store.index.ntotal)]
    manifest = dict(identity=identity, index_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                    documents=[{"text": d.page_content, "metadata": d.metadata} for d in docs])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{uuid4().hex}.tmp")
    try:
        with temp.open("wb") as output:
            np.savez_compressed(output, index=data,
                                manifest=json.dumps(manifest, ensure_ascii=False))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_index(path, identity, embeddings):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
            data = archive["index"]
        if manifest["identity"] != identity:
            raise ValueError("Cache identity mismatch")
        if data.dtype != np.uint8 or data.ndim != 1:
            raise ValueError("Invalid index bytes")
        if hashlib.sha256(data.tobytes()).hexdigest() != manifest["index_sha256"]:
            raise ValueError("Cache checksum mismatch")
        docs = manifest["documents"]
        if not docs or any(not isinstance(d["text"], str) or not d["text"].strip()
                           or d["metadata"]["chunk_id"] != i
                           or d["metadata"]["source_sha256"] != identity["source_sha256"]
                           for i, d in enumerate(docs)):
            raise ValueError("Invalid cached passages")
        index = faiss.deserialize_index(data)
        if index.ntotal != len(docs) or index.d < 1:
            raise ValueError("Cache size mismatch")
        mapping = {i: str(i) for i in range(len(docs))}
        docstore = InMemoryDocstore({str(i): Document(page_content=d["text"], metadata=d["metadata"])
                                    for i, d in enumerate(docs)})
        return FAISS(embeddings, index, docstore, mapping)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, BadZipFile) as error:
        warnings.warn(f"Ignoring unusable index cache: {type(error).__name__}")
        return None
