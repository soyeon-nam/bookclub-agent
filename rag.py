"""Book retrieval with semantic passages and reusable, content-addressed indexes."""
from pathlib import Path
from typing import Optional
import urllib.request

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from rag_index import INDEX_SETTINGS, build_index, load_index, save_index
import hashlib
import json
import warnings


BOOKS_DIR = Path(__file__).resolve().parent / "books"
BOOK_URLS = {
    "cinderella": "https://www.gutenberg.org/cache/epub/23303/pg23303.txt",
    "pinocchio": "https://www.gutenberg.org/cache/epub/500/pg500.txt",
}
BOOK_FALLBACK_URLS = {
    "cinderella": "https://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/2/3/3/0/23303/23303.txt",
    "pinocchio": "https://www.mirrorservice.org/sites/ftp.ibiblio.org/pub/docs/books/gutenberg/5/0/500/500-0.txt",
}
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
_vectorstore: Optional[FAISS] = None
index_info = {}


def _fetch_text(book_name: str) -> Optional[str]:
    """Use a local copy first; otherwise fetch a supported Gutenberg source."""
    name = book_name.strip()
    for filename in dict.fromkeys((name, name.lower())):
        path = BOOKS_DIR / f"{filename}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8-sig")
    url = BOOK_URLS.get(name.lower())
    if url is None:
        return None
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8-sig")
    except (OSError, UnicodeError):
        print("[RAG] Primary source unavailable; trying the same ebook on the mirror.")
    try:
        with urllib.request.urlopen(BOOK_FALLBACK_URLS[name.lower()], timeout=30) as response:
            return response.read().decode("utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise OSError(
            f"Cannot download '{book_name}'. Place a UTF-8 copy in {BOOKS_DIR / (name + '.txt')}."
        ) from error


def init_rag(book_name: str) -> bool:
    """Reuse an index only when the source, model and chunk settings match."""
    global _vectorstore, index_info
    _vectorstore = None
    index_info = {}
    text = _fetch_text(book_name)
    if not text or not text.strip():
        print(f"[RAG] No source for '{book_name}'; continuing without retrieval.")
        return False
    identity = {
        "book": book_name.strip().lower(),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "embedding_model": EMBEDDING_MODEL,
        "settings": dict(INDEX_SETTINGS),
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = INDEX_DIR / f"{key}.npz"
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = load_index(path, identity, embeddings)
    cache_hit = store is not None
    if store is None:
        store = build_index(text, embeddings, identity["source_sha256"])
        try:
            save_index(path, identity, store)
        except OSError as error:
            warnings.warn(f"Index cache could not be saved: {type(error).__name__}")
    _vectorstore = store
    index_info = dict(identity, cache_key=key, cache_hit=cache_hit,
                      chunk_count=store.index.ntotal)
    print(f"[RAG] {'Loaded' if cache_hit else 'Built'} '{book_name}' with {store.index.ntotal} chunks.")
    return True


def retrieve(query: str, k: int = 3) -> str:
    """Return related passages, or an empty string when RAG is unavailable."""
    if k < 1:
        raise ValueError("k must be positive")
    if _vectorstore is None or not query.strip():
        return ""
    documents = _vectorstore.similarity_search(query, k=k)
    return "\n---\n".join(document.page_content for document in documents)
