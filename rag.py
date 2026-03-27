"""Basic book retrieval: source text -> overlapping chunks -> FAISS passages."""
from pathlib import Path
from typing import Optional
import urllib.request

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
_vectorstore: Optional[FAISS] = None


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
    """Build a fresh index; missing/empty sources disable RAG for this session.

    Download, embedding and index errors propagate instead of silently pretending
    retrieval is working. Clear the old book before attempting initialization.
    """
    global _vectorstore
    _vectorstore = None
    text = _fetch_text(book_name)
    if not text or not text.strip():
        print(f"[RAG] No source for '{book_name}'; continuing without retrieval.")
        return False
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
    ).split_text(text)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    _vectorstore = FAISS.from_texts(chunks, embeddings)
    print(f"[RAG] Initialized '{book_name}' with {len(chunks)} chunks.")
    return True


def retrieve(query: str, k: int = 3) -> str:
    """Return related passages, or an empty string when RAG is unavailable."""
    if k < 1:
        raise ValueError("k must be positive")
    if _vectorstore is None or not query.strip():
        return ""
    documents = _vectorstore.similarity_search(query, k=k)
    return "\n---\n".join(document.page_content for document in documents)
