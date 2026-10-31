import re

from langchain.tools import tool
from langchain_milvus import BM25BuiltInFunction, Milvus
from pymilvus import Function, FunctionType, MilvusClient

from llmtui.config import (
    DENSE_INDEX_TYPE,
    DENSE_METRIC_TYPE,
    EMBEDDING_MODEL_KEY,
    MILVUS_COLLECTION_NAME,
    MILVUS_GRPC_OPTIONS,
    SPARSE_INDEX_TYPE,
    SPARSE_METRIC_TYPE,
)
from llmtui.config import MILVUS_DB_PATH as MILVUS_DB
from llmtui.models_config import build_embeddings, load_embeddings


def _pick_embedding_entry():
    """The embeddings entry matching the vectors in the current store.

    The store records which entry built it (set by scripts/ingest.py as a
    collection property on the LangChain collection). If it is missing or no
    longer in models.json, the dense arm would silently be nonsense -- so a
    single entry is used as the only sane guess, and anything else fails
    loudly instead of returning wrong passages.
    """

    entries = load_embeddings()
    client = MilvusClient(uri=MILVUS_DB, grpc_options=MILVUS_GRPC_OPTIONS)
    try:
        stored = None
        if client.has_collection(MILVUS_COLLECTION_NAME):
            desc = client.describe_collection(MILVUS_COLLECTION_NAME) or {}
            value = (desc.get("properties") or {}).get(EMBEDDING_MODEL_KEY)
            stored = value if isinstance(value, str) and value else None
    finally:
        client.close()
    if stored is None:
        # no store yet, or a store predating the tagging: nothing to match
        # against, so a single entry is the only sane guess
        if len(entries) == 1:
            return next(iter(entries.values()))
        raise RuntimeError(
            f"vector store {MILVUS_DB} has no recorded embedding model and"
            " models.json lists more than one -- run scripts/ingest.py to"
            " rebuild it and tag which one it was built with"
        )
    if stored in entries:
        return entries[stored]
    raise RuntimeError(
        f"vector store {MILVUS_DB} was built with embeddings entry"
        f" {stored!r}, which is no longer in models.json -- re-ingest"
        " or restore the entry"
    )


# different then model sampling TOP_K.
# 5 retrievals for now, i dont want to flood the context.
# but after implementing context management maybe 10 could fit
TOP_K = 5
# amount of candidates dense and arm both return separately BEFORE they get fused.
# langchain defaults this to 4, so the ranker is pretty useless at that number
# per query impact is low, the embedding of the query is the bottleneck, not fetching candidates after
FETCH_K = 50
# rrf scores a chunk as the sum of 1/(RRF_K + rank) over both arms. lower
# favours a strong hit in one arm, higher favours agreement between the two.
RRF_K = 60

# has to mirror the schema ingest.py builds, or the collection won't load.
# BM25 is exact-term only here -- see the note in ingest.py
_store: Milvus | None = None


def vector_store() -> Milvus:
    """The live store, built on the first search and kept from then on.

    Built here rather than at import: importing this module must not need an
    ingested store or a models.json (the TUI and the tests import it), and
    building opens a milvus-lite server that then holds the db dir open for
    the life of the process.
    """

    global _store
    if _store is None:
        _store = Milvus(
            embedding_function=build_embeddings(_pick_embedding_entry()),
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            # must match ingest.py -- see the note there on why
            # SPARSE_INVERTED_INDEX is spelled out instead of letting BM25
            # default to AUTOINDEX
            index_params=[
                {
                    "metric_type": DENSE_METRIC_TYPE,
                    "index_type": DENSE_INDEX_TYPE,
                    "params": {},
                },
                {
                    "metric_type": SPARSE_METRIC_TYPE,
                    "index_type": SPARSE_INDEX_TYPE,
                    "params": {},
                },
            ],
            # milvus grpc sends keep_alive messages, adding noise to the frontend
            connection_args={"uri": MILVUS_DB, "grpc_options": MILVUS_GRPC_OPTIONS},
            # we never read ids back, so let milvus mint them: also silences the
            # "No ids provided and auto_id is False" UserWarning on every add
            auto_id=True,
        )
    return _store


# https://milvus.io/docs/rrf-ranker.md
# reranker to merge the proposed candidates from the dense and sparse search returns
# grabs the best candidate not based on the score from each search, but by how many times
# they are proposed and at what ranks. it is rank based reranker, not scale based
# a requirement for hybrid retrieval, and pretty fucking cool

reranker = Function(
    name="rrf",
    function_type=FunctionType.RERANK,
    input_field_names=[],
    params={"reranker": "rrf", "k": RRF_K},
)


def book_pattern(book: str) -> str:
    """A `like` pattern matching a filename however the model spelled the title.

    Filenames are underscore-separated, but the model writes titles with spaces
    about as often, so every run of separators becomes a wildcard: "against the
    cult" and "against_the_cult" both find
    n1_against_the_cult_of_the_reptile_god.pdf.

    Quotes and backslashes would break out of the expression, and % is the
    wildcard itself, so those three go.
    """

    cleaned = re.sub(r'["\\%]', "", book).strip()
    return "%" + re.sub(r"[\s_-]+", "%", cleaned) + "%"


@tool(response_format="content_and_artifact")
def search_books(query: str, book: str = "") -> tuple[str, dict]:
    """Search the ingested document corpus for passages relevant to the query.
    Use this for questions about roleplaying games, their rules, and anything
    else that would be mentioned in their books.

    This search always returns passages when the library is non-empty -- it
    returns the closest matches, not the good ones. Passages about an unrelated
    subject are what "not covered" looks like here; there is no empty result to
    signal it. So read what comes back before using it: if the passages do not
    actually address the question, say so plainly rather than stretching them
    into an answer, and do not fall back on general knowledge.

    Args:
        query: What to look for. Use the subject matter only -- do not put the
            book title in here, use the book argument for that.
        book: Optional. Part of a book's filename, to restrict the search to
            that book. Leave empty to search the whole library.

    Returns the passages and, alongside them, how many there were. The count
    goes in the artifact rather than the text, so the tool row can show that a
    search came back empty without that number reaching the model. Cite the
    passages you use by copying each result's opening marker -- [file.pdf p.N],
    with the path exactly as shown -- verbatim into the answer: the interface
    turns exactly that syntax into a link that opens the PDF at page N. Never
    alter the filename or page, never merge two results into one marker.
    """

    # clean the arg from query chars. milvus uses its own expression language
    book = book.strip()

    # Note: the hybrid search guide https://milvus.io/docs/milvus_hybrid_search_retriever.md#Specify-the-index-params-for-multi-vector-fields
    # says to use the "ranker_type" parameter, but i got deprecation warnings, so i used this custom function as per the warning
    kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "reranker": reranker}

    # we send an extra filter.
    if book:
        kwargs["expr"] = f'source like "{book_pattern(book)}"'

    # hybrid search using dense semantic similarity + bm25 sparse search merged and reranked with rrf
    results = vector_store().similarity_search(query, **kwargs)

    # sending back prompt that explains why a result was not found
    if not results:
        if book:
            return (
                f"No passages matched in books whose filename contains '{book}'. "
                f"Either that book is not in the library, or it is there and has "
                f"nothing on this topic -- this result cannot tell you which. "
                f"Next step: search again with the same query and no book "
                f"argument. If that also finds nothing, stop searching and tell "
                f"the user; ask them for the book's exact title only if you "
                f"think the name was the problem. "
                f"Do not answer from general knowledge."
            ), {"passages": 0}
        return (
            "No passages matched anywhere in the document library. If you have "
            "already tried a different phrasing of this search, stop here and "
            "tell the user the library does not appear to cover it. "
            "Do not answer from general knowledge."
        ), {"passages": 0}

    # alongside the matched chunk, send metadata like the book title, page number.
    # the full path survives on purpose: the library nests in per-game folders, and
    # citations resolve the file from this path -- a basename alone could not be
    # found again
    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        # PyMuPDFLoader starts counting at 0, opens one page before when clicking, so:
        if isinstance(page, int):
            page += 1
        formatted.append(f"[{source} p.{page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted), {"passages": len(results)}
