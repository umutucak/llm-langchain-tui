import os

from dotenv import load_dotenv

from langchain.tools import tool
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_milvus import Milvus, BM25BuiltInFunction
from pymilvus import Function, FunctionType

load_dotenv()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
MILVUS_DB = os.getenv("MILVUS_PROD_DB_PATH")
DENSE_METRIC_TYPE = os.getenv("DENSE_METRIC_TYPE")
DENSE_INDEX_TYPE = os.getenv("DENSE_INDEX_TYPE")
SPARSE_METRIC_TYPE = os.getenv("SPARSE_METRIC_TYPE")
SPARSE_INDEX_TYPE = os.getenv("SPARSE_INDEX_TYPE")
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
vector_store = Milvus(
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
    builtin_function=BM25BuiltInFunction(),
    vector_field=["dense", "sparse"],
    # must match ingest.py -- see the note there on why SPARSE_INVERTED_INDEX
    # is spelled out instead of letting BM25 default to AUTOINDEX
    index_params=[
        {"metric_type": DENSE_METRIC_TYPE, "index_type": DENSE_INDEX_TYPE, "params": {}},
        {"metric_type": SPARSE_METRIC_TYPE, "index_type": SPARSE_INDEX_TYPE, "params": {}},
    ],
    connection_args={"uri": MILVUS_DB}
)

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


@tool
def search_books(query: str, book: str = "") -> str:
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
    """

    # clean the arg from query chars. milvus uses its own expression language
    book = book.replace('"', "").replace("\\", "").replace("%", "").replace("_", "").strip()

    # Note: the hybrid search guide https://milvus.io/docs/milvus_hybrid_search_retriever.md#Specify-the-index-params-for-multi-vector-fields
    # says to use the "ranker_type" parameter, but i got deprecation warnings, so i used this custom function as per the warning 
    kwargs = {"k": TOP_K, "fetch_k": FETCH_K, "reranker": reranker}

    # we send an extra filter. 
    if book:
        kwargs["expr"] = f'source like "%{book}%"'

    # hybrid search using dense semantic similarity + bm25 sparse search merged and reranked with rrf
    results = vector_store.similarity_search(query, **kwargs)

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
            )
        return (
            "No passages matched anywhere in the document library. If you have "
            "already tried a different phrasing of this search, stop here and "
            "tell the user the library does not appear to cover it. "
            "Do not answer from general knowledge."
        )

    # alongside the matched chunk, send metadata like the book title, page number
    formatted = []
    for doc in results:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{source} p.{page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)
