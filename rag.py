"""Book retrieval with semantic passages and reusable, content-addressed indexes."""
from pathlib import Path
from typing import Optional
import urllib.request

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_search import KeywordIndex, search
import os
from rag_index import INDEX_SETTINGS, build_index, load_index, save_index
import hashlib
import json
import warnings
from runtime_cache import MEMO, cache_key


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
_keywords = None
_llm = None
_llm_model = None


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


def retrieval_mode():
    mode = os.getenv('RAG_MODE', 'faiss').strip().lower()
    if mode not in ('faiss', 'advanced'):
        raise ValueError('RAG_MODE must be faiss or advanced')
    return mode


def init_rag(book_name: str) -> bool:
    """Reuse an index only when the source, model and chunk settings match."""
    global _vectorstore, index_info, _keywords
    retrieval_mode()  # Reject invalid configuration before downloads or API calls.
    MEMO.clear()
    _vectorstore = None
    index_info = {}
    _keywords = None
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
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, request_timeout=30, max_retries=1)
    store = load_index(path, identity, embeddings)
    cache_hit = store is not None
    if store is None:
        store = build_index(text, embeddings, identity["source_sha256"])
        try:
            save_index(path, identity, store)
        except OSError as error:
            warnings.warn(f"Index cache could not be saved: {type(error).__name__}")
    documents = [store.docstore.search(store.index_to_docstore_id[i])
                 for i in range(store.index.ntotal)]
    _keywords = KeywordIndex(documents)
    _vectorstore = store
    index_info = dict(identity, cache_key=key, cache_hit=cache_hit,
                      chunk_count=store.index.ntotal)
    print(f"[RAG] {'Loaded' if cache_hit else 'Built'} '{book_name}' with {store.index.ntotal} chunks.")
    return True


def _get_llm(model=None):
    global _llm, _llm_model
    model = model or os.getenv("GPT_MODEL", "gpt-4o-mini")
    if _llm is None or _llm_model != model:
        _llm_model = model
        _llm = ChatOpenAI(model=model, temperature=0,
                          timeout=30, max_retries=1)
    return _llm


def _search_request(query, k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    mode = retrieval_mode()
    if _vectorstore is None or not query.strip():
        return None
    # Capture the actual book objects; a pending search cannot switch to a new book.
    store, keywords, identity = _vectorstore, _keywords, dict(index_info)
    model = os.getenv("GPT_MODEL", "gpt-4o-mini") if mode == "advanced" else None
    key = cache_key('retrieval-v2', {'index':identity.get('cache_key'), 'model':model, 'mode':mode,
                                    'query':query.strip(), 'k':k})
    def compute():
        trace = {'status':'searching', 'cache_key':identity.get('cache_key'), 'model':model, 'mode':mode}
        if mode == 'faiss':
            docs = store.similarity_search(query.strip(), k=k)
            text = '\n---\n'.join(doc.page_content for doc in docs)
            trace.update(original_query=query.strip(), rewritten_query=query.strip(),
                         rewrite_status='skipped', rerank_status='skipped',
                         selected_chunk_ids=[doc.metadata['chunk_id'] for doc in docs],
                         candidates=[{'chunk_id':doc.metadata['chunk_id'],'text':doc.page_content} for doc in docs],
                         retrieval_chat_calls=0)
        else:
            text = search(store, keywords, query.strip(), identity.get('book',''), k, lambda:_get_llm(model), trace)
        trace['status'] = 'passed'
        return {'text':text, 'trace':trace}
    return key, compute


def _cacheable(result):
    trace = result['trace']
    return trace.get('rewrite_status') != 'fallback' and trace.get('rerank_status') != 'fallback'


def retrieve(query: str, k: int = 3, *, trace=None) -> str:
    """Reuse a completed or in-flight search, with a private trace for this turn."""
    details = trace if trace is not None else {}
    details.clear()
    request = _search_request(query, k)
    if request is None:
        details['status'] = 'unavailable' if _vectorstore is None else 'empty_query'
        return ''
    result, status = MEMO.get(*request, cache_if=_cacheable)
    details.update(result['trace'], cache_status=status)
    return result['text']


def prefetch_retrieve(query: str, k: int = 3):
    request = _search_request(query, k)
    if request is None:
        return 'unavailable'
    return MEMO.prefetch(*request, cache_if=_cacheable)
