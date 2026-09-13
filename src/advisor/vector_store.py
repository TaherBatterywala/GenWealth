"""
GenWealth Advisor Engine — Phase 3, Step 3.1
============================================
Module: src/advisor/vector_store.py

Implements the Hybrid RAG (Retrieval-Augmented Generation) knowledge base:
  - MongoDB Atlas as the persistent vector store
  - sentence-transformers (all-MiniLM-L6-v2) for local, keyless embeddings
  - DuckDuckGo Search (DDGS) for real-time, keyless market news retrieval
  - Cosine-similarity ranking performed in-application (pre-Atlas Vector Index)

Architecture Notes:
  • Embeddings are 384-dimensional dense float vectors (all-MiniLM-L6-v2).
  • For production scale, enable MongoDB Atlas Vector Search index (knnBeta)
    on the `embedding` field to push similarity ranking server-side.
  • This module is fully self-contained and requires no paid API keys for
    embedding or news retrieval.

Dependencies (add to requirements.txt):
  pymongo[srv]>=4.6
  sentence-transformers>=2.7
  python-dotenv>=1.0
  ddgs>=9.0
  numpy>=1.26
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from ddgs import DDGS

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("genwealth.vector_store")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384          # Dimension of all-MiniLM-L6-v2 output vectors
DB_NAME: str = "genwealth_db"
COLLECTION_NAME: str = "financial_knowledge"
# Maximum age (in hours) before stored articles are considered stale and refreshed
FRESHNESS_HOURS: int = 24

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (initialised on first use to avoid startup overhead)
# ---------------------------------------------------------------------------
_embedding_model: Optional[Any] = None  # SentenceTransformer or None if torch unavailable
_mongo_client: Optional[MongoClient] = None
_collection: Optional[Collection] = None


# ===========================================================================
# 1. Environment & Database Initialisation
# ===========================================================================

def _load_env() -> str:
    """
    Load environment variables from .env and return the MongoDB URI.

    Returns:
        str: The MongoDB connection URI.

    Raises:
        EnvironmentError: If MONGODB_URI is not found in the environment.
    """
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise EnvironmentError(
            "MONGODB_URI not found. Ensure it is set in your .env file.\n"
            "Expected key: MONGODB_URI=mongodb+srv://<user>:<password>@..."
        )
    return uri


def get_mongo_collection() -> Collection:
    """
    Return a cached MongoDB collection handle, creating the connection on
    first invocation.

    The function uses a module-level singleton pattern to avoid opening
    multiple connections across repeated calls within the same process.

    Returns:
        Collection: pymongo Collection object for `financial_knowledge`.

    Raises:
        ConnectionFailure: If the Atlas cluster is unreachable.
        EnvironmentError: If MONGODB_URI is missing from the environment.
    """
    global _mongo_client, _collection

    if _collection is not None:
        return _collection

    uri = _load_env()
    logger.info("Connecting to MongoDB Atlas...")
    try:
        # serverSelectionTimeoutMS prevents indefinite blocking on bad URIs
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=8_000)
        # Force an actual network round-trip to validate credentials
        _mongo_client.admin.command("ping")
        logger.info("MongoDB Atlas connection established successfully.")
    except (ConnectionFailure, ServerSelectionTimeoutError) as exc:
        logger.error(
            "Could not connect to MongoDB Atlas.\n"
            "   Check MONGODB_URI, network access rules, and cluster status.\n"
            "   Error: %s", exc
        )
        raise

    db = _mongo_client[DB_NAME]
    _collection = db[COLLECTION_NAME]

    # Ensure indexes exist for efficient ticker-based lookups
    _collection.create_index("ticker", background=True)
    _collection.create_index([("ticker", 1), ("source_type", 1)], background=True)
    logger.info(
        "Collection '%s.%s' ready with indexes.", DB_NAME, COLLECTION_NAME
    )
    return _collection


# ===========================================================================
# 2. Local Vector Embeddings
# ===========================================================================

def _get_embedding_model() -> Any:
    """
    Lazy-load and cache the sentence-transformer model.

    The model is downloaded once on first access (~90 MB) and reused for all
    subsequent calls in the same process. Returns None gracefully if
    sentence-transformers / torch is not installed.

    Returns:
        SentenceTransformer instance, or None if unavailable.
    """
    global _embedding_model
    if _embedding_model is None:
        if not _SENTENCE_TRANSFORMERS_AVAILABLE or _SentenceTransformer is None:
            logger.warning(
                "sentence-transformers not installed; embeddings will be zero vectors. "
                "Run: pip install sentence-transformers"
            )
            return None
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        _embedding_model = _SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info(
            "Embedding model loaded. Output dimension: %d.", EMBEDDING_DIM
        )
    return _embedding_model


def generate_embedding(text: str) -> list[float]:
    """
    Convert a text string into a 384-dimensional float embedding vector.

    Uses sentence-transformers `all-MiniLM-L6-v2` running **locally** --
    no external API call or API key required.

    Args:
        text (str): The input text to embed. Should be < 512 word-pieces for
                    best accuracy (the model silently truncates beyond that).

    Returns:
        list[float]: A 384-dimensional list of floats representing the semantic
                     meaning of the input text.

    Example:
        >>> vec = generate_embedding("NVIDIA Q2 earnings beat expectations")
        >>> len(vec)
        384
    """
    if not text or not text.strip():
        logger.warning("generate_embedding received empty text; returning zero vector.")
        return [0.0] * EMBEDDING_DIM

    model = _get_embedding_model()
    if model is None:
        logger.warning("Embedding model unavailable; returning zero vector for: %s", text[:60])
        return [0.0] * EMBEDDING_DIM
    # normalize_embeddings=True gives unit vectors -- ideal for cosine similarity
    vector: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ===========================================================================
# 3. Real-Time Web Search (DuckDuckGo -- Keyless)
# ===========================================================================

def get_live_stock_news(ticker: str, max_results: int = 3) -> list[dict]:
    """
    Fetch real-time market news for a stock ticker using DuckDuckGo Search.

    This function requires **no API key** and imposes **no rate-limit charges**.
    Results reflect the most recent news indexed by DuckDuckGo.

    Args:
        ticker (str):      Stock ticker symbol, e.g. "NVDA", "AAPL", "TSLA".
        max_results (int): Maximum number of news articles to return (default 3).

    Returns:
        list[dict]: Each element is a dict with keys:
            - ``title``   (str): Headline of the article.
            - ``body``    (str): Snippet / summary of the article body.
            - ``url``     (str): Direct URL to the full article.
            - ``source``  (str): Publisher name extracted from URL.
            - ``ticker``  (str): The requested ticker symbol (for traceability).

    Example:
        >>> articles = get_live_stock_news("NVDA", max_results=2)
        >>> articles[0]["title"]
        'Nvidia posts record revenue, shares surge after earnings'
    """
    query = f"{ticker} stock news latest earnings market"
    logger.info("Fetching live news for '%s' via DuckDuckGo...", ticker)
    articles: list[dict] = []

    try:
        with DDGS() as ddgs:
            # ddgs>=9.0 signature: news(query: str, **kwargs)
            # `query` is a required positional arg — do NOT use keywords=
            raw_results = ddgs.news(
                query,
                max_results=max_results,
                safesearch="off",
            )
            for item in raw_results:
                # DDGS news keys: title, body, url, source, date, image
                articles.append(
                    {
                        "title":  item.get("title", "No title"),
                        "body":   item.get("body", ""),
                        "url":    item.get("url", ""),
                        "source": item.get("source", "Unknown"),
                        "ticker": ticker.upper(),
                    }
                )
    except Exception as exc:
        logger.error(
            "DuckDuckGo news search failed for ticker '%s': %s", ticker, exc
        )
        # Return empty list rather than crashing the pipeline
        return []

    logger.info("Found %d article(s) for '%s'.", len(articles), ticker)
    return articles


# ===========================================================================
# 4. Vector Knowledge Storage & Search
# ===========================================================================

def store_knowledge_item(
    ticker: str,
    text_content: str,
    source_type: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Embed a text document and upsert it into the MongoDB knowledge base.

    Uses ``upsert`` semantics keyed on (ticker, text_content) so that
    re-ingesting the same article updates its timestamp rather than creating
    duplicates.

    Args:
        ticker       (str):  Stock ticker associated with this knowledge item.
        text_content (str):  The full text body to embed and store.
        source_type  (str):  Category label, e.g. "live_news", "sec_filing",
                              "analyst_report", "earnings_transcript".
        metadata     (dict): Optional free-form key-value pairs to attach
                              (e.g., {"url": "...", "source": "Reuters"}).

    Returns:
        str: MongoDB ObjectId of the upserted document (as a string).

    Raises:
        OperationFailure: On a MongoDB write error.
    """
    collection = get_mongo_collection()
    embedding = generate_embedding(text_content)
    now_utc = datetime.now(timezone.utc)

    document = {
        "ticker":       ticker.upper(),
        "text_content": text_content,
        "embedding":    embedding,        # list[float], length 384
        "source_type":  source_type,
        "timestamp":    now_utc,
        "metadata":     metadata or {},
    }

    # Upsert: match on ticker + exact text to avoid duplicates
    filter_doc = {
        "ticker":       ticker.upper(),
        "text_content": text_content,
    }

    try:
        result = collection.update_one(
            filter_doc,
            {"$set": document},
            upsert=True,
        )
        upserted_id = result.upserted_id or "existing-doc-updated"
        logger.info(
            "Stored '%s' item for '%s' (upserted_id=%s).",
            source_type, ticker, upserted_id,
        )
        return str(upserted_id)
    except OperationFailure as exc:
        logger.error("MongoDB write error: %s", exc)
        raise


def _make_utc_aware(dt: datetime) -> datetime:
    """
    Ensure a datetime object is UTC offset-aware.

    MongoDB may return either offset-aware (timezone.utc) or offset-naive
    datetimes depending on driver version and document origin. Comparing a
    naive datetime against an aware one raises a ``TypeError`` at runtime.
    This helper normalises any incoming datetime to UTC-aware.

    Args:
        dt (datetime): A datetime that may or may not carry tzinfo.

    Returns:
        datetime: The same point in time with ``tzinfo=timezone.utc`` set.
    """
    if dt.tzinfo is None:
        # Treat naive timestamps as UTC (MongoDB stores in UTC by default)
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def query_knowledge_base(
    ticker: str,
    query_text: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the most semantically relevant documents for a ticker + query.

    **Search Strategy (Hybrid Fallback + Freshness Gate)**:
    1. Retrieve all documents in MongoDB matching the given ticker.
    2. **Freshness Check** — if the newest stored document is older than
       ``FRESHNESS_HOURS`` (24 h), automatically fetch fresh news via
       DuckDuckGo and upsert the results before ranking.
    3. If **no documents exist** for this ticker (cold-start), automatically:
       a. Call ``get_live_stock_news(ticker)`` to fetch fresh articles.
       b. Store each article via ``store_knowledge_item()``.
       c. Return those freshly ingested articles as the result set.
    4. Rank all stored documents by cosine similarity (in-application).

    Timezone safety: all timestamp comparisons use UTC-aware datetimes to
    prevent ``TypeError`` from offset-naive vs. offset-aware mismatches.

    Args:
        ticker     (str): Stock ticker to query, e.g. "AAPL".
        query_text (str): Natural-language query to match against stored text.
        top_k      (int): Number of top results to return (default 3).

    Returns:
        list[dict]: Top-k documents sorted by descending cosine similarity.
                    Each dict contains ``ticker``, ``text_content``,
                    ``source_type``, ``timestamp``, ``metadata``,
                    and a computed ``similarity_score`` (float, 0--1).

    Example:
        >>> docs = query_knowledge_base("NVDA", "GPU demand outlook 2025")
        >>> docs[0]["similarity_score"]
        0.87
    """
    collection = get_mongo_collection()
    ticker_upper = ticker.upper()
    now_utc = datetime.now(timezone.utc)

    # -- Step 1: Check if we have any stored documents for this ticker --------
    stored_count = collection.count_documents({"ticker": ticker_upper})
    logger.info(
        "Found %d stored document(s) for ticker '%s'.",
        stored_count, ticker_upper,
    )

    if stored_count == 0:
        # -- Cold-start fallback: fetch live news and seed the DB -------------
        logger.info(
            "No existing knowledge for '%s'. Fetching live news...", ticker_upper
        )
        live_articles = get_live_stock_news(ticker_upper, max_results=top_k)

        if not live_articles:
            logger.warning("No live news found for '%s'. Returning empty.", ticker_upper)
            return []

        for article in live_articles:
            combined_text = f"{article['title']}. {article['body']}"
            store_knowledge_item(
                ticker=ticker_upper,
                text_content=combined_text,
                source_type="live_news",
                metadata={
                    "url":    article["url"],
                    "source": article["source"],
                    "title":  article["title"],
                },
            )
        logger.info(
            "Seeded %d live article(s) for '%s' into MongoDB.",
            len(live_articles), ticker_upper,
        )

    else:
        # -- Step 2: Freshness Gate -- refresh if newest doc is > 24h old -----
        newest_doc = collection.find_one(
            {"ticker": ticker_upper},
            sort=[("timestamp", -1)],
            projection={"timestamp": 1, "_id": 0},
        )
        if newest_doc and newest_doc.get("timestamp"):
            # Normalise to UTC-aware to prevent naive/aware TypeError
            newest_ts = _make_utc_aware(newest_doc["timestamp"])
            age_hours = (now_utc - newest_ts).total_seconds() / 3600.0
            logger.info(
                "Newest document for '%s' is %.1f hour(s) old (threshold: %d h).",
                ticker_upper, age_hours, FRESHNESS_HOURS,
            )
            if age_hours > FRESHNESS_HOURS:
                logger.info(
                    "Stale knowledge detected for '%s'. Fetching fresh news...",
                    ticker_upper,
                )
                fresh_articles = get_live_stock_news(ticker_upper, max_results=top_k)
                if fresh_articles:
                    for article in fresh_articles:
                        combined_text = f"{article['title']}. {article['body']}"
                        store_knowledge_item(
                            ticker=ticker_upper,
                            text_content=combined_text,
                            source_type="live_news",
                            metadata={
                                "url":    article["url"],
                                "source": article["source"],
                                "title":  article["title"],
                            },
                        )
                    logger.info(
                        "Refreshed %d article(s) for '%s' into MongoDB.",
                        len(fresh_articles), ticker_upper,
                    )
                else:
                    logger.warning(
                        "Freshness refresh: no new articles found for '%s'. "
                        "Serving stale cache.",
                        ticker_upper,
                    )
            else:
                logger.info(
                    "Knowledge for '%s' is fresh (%.1f h < %d h threshold). "
                    "Skipping live fetch.",
                    ticker_upper, age_hours, FRESHNESS_HOURS,
                )

    # -- Step 3: Retrieve all ticker documents and rank by similarity ----------
    query_embedding = np.array(generate_embedding(query_text), dtype=np.float32)

    cursor = collection.find(
        {"ticker": ticker_upper},
        {
            "embedding":    1,
            "text_content": 1,
            "source_type":  1,
            "timestamp":    1,
            "metadata":     1,
            "_id":          0,
        },
    )

    ranked: list[tuple[float, dict]] = []
    for doc in cursor:
        stored_vec = np.array(doc.get("embedding", []), dtype=np.float32)
        if stored_vec.size == 0:
            continue
        # Cosine similarity: dot product of two L2-normalised unit vectors
        # (embeddings are already normalised by generate_embedding)
        score: float = float(np.dot(query_embedding, stored_vec))
        result_doc = {
            "ticker":           ticker_upper,
            "text_content":     doc.get("text_content", ""),
            "source_type":      doc.get("source_type", ""),
            "timestamp":        doc.get("timestamp"),
            "metadata":         doc.get("metadata", {}),
            "similarity_score": round(score, 4),
        }
        ranked.append((score, result_doc))

    # Sort descending by similarity score
    ranked.sort(key=lambda x: x[0], reverse=True)
    top_results = [doc for _, doc in ranked[:top_k]]

    logger.info(
        "Returning top-%d result(s) for query '%s...' (ticker=%s).",
        len(top_results), query_text[:50], ticker_upper,
    )
    return top_results


def bulk_store_news_articles(articles: list[dict]) -> list[str]:
    """
    Efficiently store multiple news articles using bulk write operations.

    This is a utility helper for batch-seeding the knowledge base during
    initial setup or scheduled refresh jobs.

    Args:
        articles (list[dict]): List of article dicts as returned by
                                ``get_live_stock_news()``. Each must contain
                                ``ticker``, ``title``, ``body``, ``url``, and
                                ``source`` keys.

    Returns:
        list[str]: List of upserted document IDs (as strings).
    """
    ids: list[str] = []
    for article in articles:
        ticker = article.get("ticker", "UNKNOWN")
        combined_text = f"{article.get('title', '')}. {article.get('body', '')}"
        doc_id = store_knowledge_item(
            ticker=ticker,
            text_content=combined_text,
            source_type="live_news",
            metadata={
                "url":    article.get("url", ""),
                "source": article.get("source", ""),
                "title":  article.get("title", ""),
            },
        )
        ids.append(doc_id)
    return ids


# ===========================================================================
# 5. Self-Test Execution Block
# ===========================================================================

if __name__ == "__main__":
    """
    Self-test suite for vector_store.py.

    Run from the project root:
        python -m src.advisor.vector_store
        # or
        python src/advisor/vector_store.py

    Tests performed:
        1. MongoDB Atlas connection health check
        2. Local embedding model load + shape validation
        3. Live news retrieval via DuckDuckGo for NVDA
        4. Batch storage of fetched articles into MongoDB
        5. Vector similarity search on stored knowledge
    """
    import sys

    TICKER = "NVDA"
    QUERY  = "NVIDIA GPU demand AI data center revenue outlook"

    print("\n" + "=" * 65)
    print("  GenWealth -- Phase 3 | vector_store.py Self-Test")
    print("=" * 65 + "\n")

    # -- Test 1: Database Connectivity ----------------------------------------
    print("[TEST 1] MongoDB Atlas Connection...")
    try:
        col = get_mongo_collection()
        doc_count = col.count_documents({})
        print(f"  PASS  Connected. Total documents in collection: {doc_count}\n")
    except Exception as e:
        print(f"  FAIL  Connection failed: {e}")
        sys.exit(1)

    # -- Test 2: Embedding Generation -----------------------------------------
    print("[TEST 2] Local Embedding Generation...")
    sample_text = "NVIDIA reported record quarterly revenue driven by AI chip demand."
    vec = generate_embedding(sample_text)
    assert len(vec) == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM} dims, got {len(vec)}"
    )
    print(f"  PASS  Embedding generated. Dimensions: {len(vec)}")
    print(f"        First 5 values: {[round(v, 4) for v in vec[:5]]}\n")

    # -- Test 3: Live News Retrieval ------------------------------------------
    print(f"[TEST 3] DuckDuckGo Live News Retrieval for '{TICKER}'...")
    news_articles = get_live_stock_news(TICKER, max_results=3)
    if news_articles:
        for i, article in enumerate(news_articles, 1):
            print(f"  [{i}] {article['title']}")
            print(f"       Source : {article['source']}")
            print(f"       URL    : {article['url']}")
            print(f"       Snippet: {article['body'][:120]}...\n")
    else:
        print("  WARN  No articles retrieved (check network access).\n")

    # -- Test 4: Batch Storage into MongoDB -----------------------------------
    print(f"[TEST 4] Storing articles for '{TICKER}' into MongoDB...")
    if news_articles:
        stored_ids = bulk_store_news_articles(news_articles)
        print(f"  PASS  Stored {len(stored_ids)} article(s).")
        for sid in stored_ids:
            print(f"        ID: {sid}")
        print()
    else:
        print("  WARN  Skipping storage (no articles to store).\n")

    # -- Test 5: Vector Similarity Search -------------------------------------
    print("[TEST 5] Querying Knowledge Base...")
    print(f"  Ticker : {TICKER}")
    print(f"  Query  : '{QUERY}'\n")
    results = query_knowledge_base(TICKER, QUERY, top_k=3)

    if results:
        for rank, doc in enumerate(results, 1):
            print(f"  Rank {rank} | Score: {doc['similarity_score']:.4f}")
            print(f"    Source Type : {doc['source_type']}")
            print(f"    Text Preview: {doc['text_content'][:120]}...")
            print(f"    URL         : {doc['metadata'].get('url', 'N/A')}\n")
    else:
        print("  WARN  No results returned from knowledge base query.\n")

    print("=" * 65)
    print("  Self-test complete.")
    print("=" * 65 + "\n")
