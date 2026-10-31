import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from tqdm import tqdm

# this is run straight from a shell, so the project root has to go on the path
# before anything under llmtui can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# silent harmless warnings
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
pymupdf.TOOLS.mupdf_display_errors(None)

from llmtui.config import (
    DENSE_INDEX_TYPE,
    DENSE_METRIC_TYPE,
    EMBEDDING_MODEL_KEY,
    MILVUS_BACKUP_DIR,
    MILVUS_COLLECTION_NAME,
    MILVUS_DB_PATH,
    MILVUS_GRPC_OPTIONS,
    SPARSE_INDEX_TYPE,
    SPARSE_METRIC_TYPE,
)
from llmtui.models_config import build_embeddings, load_embeddings
from llmtui.store_backup import backup_store

MILVUS_DB = MILVUS_DB_PATH

# which embeddings entry built a store: recorded as a collection property so
# search_books can pick the matching entry and a future ingest can show which
# one is live. pymilvus 3.0 exposes this as collection properties:
# alter_collection_properties to write, describe_collection to read. a store
# built before this tagging exists has no such property -- describe returns
# {} for it, so the read is None, not an error


def _store_client() -> MilvusClient:
    return MilvusClient(uri=MILVUS_DB, grpc_options=MILVUS_GRPC_OPTIONS)


def read_embedding_model() -> str | None:
    """The embedding entry name recorded on the store, or None.

    None means "not tagged": the database has no collection yet (ingest
    never ran against it) or predates the tagging.
    """

    client = _store_client()
    try:
        if not client.has_collection(MILVUS_COLLECTION_NAME):
            return None
        desc = client.describe_collection(MILVUS_COLLECTION_NAME) or {}
        value = (desc.get("properties") or {}).get(EMBEDDING_MODEL_KEY)
        return value if isinstance(value, str) and value else None
    finally:
        client.close()


def read_dense_dim() -> int | None:
    """The dim of the store's dense vector field, or None if no collection.

    The schema is fixed at collection creation: a model with a different dim
    cannot insert into it, so this is what the pre-wipe check compares the
    picked entry against.
    """

    client = _store_client()
    try:
        if not client.has_collection(MILVUS_COLLECTION_NAME):
            return None
        desc = client.describe_collection(MILVUS_COLLECTION_NAME) or {}
        for field in desc.get("fields") or []:
            if field.get("name") == "dense":
                dim = (field.get("params") or {}).get("dim")
                return int(dim) if dim is not None else None
        return None
    finally:
        client.close()


def write_embedding_model(name: str) -> None:
    """Record which embeddings entry produced this store's vectors.

    Called once after the first batch lands (the collection exists by then);
    a later call just overwrites the same property.
    """

    client = _store_client()
    try:
        client.alter_collection_properties(
            MILVUS_COLLECTION_NAME, {EMBEDDING_MODEL_KEY: name}
        )
    finally:
        client.close()


def embed_and_insert_batches(vector_store, batches, workers: int) -> None:
    """Insert every batch and fail if a worker cannot complete."""
    if not batches:
        return

    # i frontload and create the collection and its schema.
    # otherwise there is a race condition below raising DataNotMatchException
    vector_store.add_documents(batches[0])
    # collect failures instead of logging and ignoring them: a store with a
    # missing batch must not be stamped usable
    failures = []
    # cheeky little parallelisation again. god forbid i doom scroll too long
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(vector_store.add_documents, batch): number
            for number, batch in enumerate(batches[1:], start=1)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(batches),
            initial=1,
            desc="Embedding+inserting",
            unit="batch",
        ):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], exc))
                tqdm.write(f"[FAILED] batch {futures[future]}: {exc}")

    if failures:
        details = ", ".join(f"batch {number}: {exc}" for number, exc in failures)
        raise RuntimeError(
            f"embedding ingest failed; the index is incomplete ({details})"
        )


CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150
# art-heavy pages extract to nothing but a running header + page number
# those tiny chunks embed almost purely as the book title, so any query
# naming the book matches them ahead of real content.
# i drop them by filtering by chunk size
# it might be defeating the purpose of BM25 document length normalisation, i dont know
MIN_CHUNK_CHARS: int = 50
CPU_COUNT: int = os.cpu_count()


def load_pdf(path: Path) -> list[Document]:
    return PyMuPDFLoader(str(path)).load()


def pick_embedding_entry(entries: dict[str, object], current: str | None) -> object:
    """Ask which embeddings entry to build with; auto-pick when unambiguous.

    One entry configured: no question, it is the only choice. Several:
    list them numbered, marking the one that built the current store, and
    read a choice from the user. The stored tag is informational -- picking
    a different entry still wipes and rebuilds with the new model.
    """

    if len(entries) == 1:
        name = next(iter(entries))
        print(f"Using embedding model: {name}")
        return entries[name]

    print("Embedding models in models.json:")
    names = list(entries)
    for i, name in enumerate(names, start=1):
        marker = "  (current store)" if name == current else ""
        print(f"  {i}. {name}{marker}")
    while True:
        choice = input(f"Which one to build with? [1-{len(names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            name = names[int(choice) - 1]
            print(f"Using embedding model: {name}")
            return entries[name]
        print(f"Enter a number between 1 and {len(names)}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a directory of PDFs into the live vector store."
    )
    parser.add_argument("pdf_dir", help="directory to search recursively for PDFs")
    args = parser.parse_args()
    PDF_DIR: str = args.pdf_dir

    # checked before the wipe prompt: a bad path should not cost you a database
    if not Path(PDF_DIR).is_dir():
        parser.error(f"{PDF_DIR} is not a directory")

    # pick the embedding model before anything destructive: the vectors it
    # produces are what the wipe throws away, so the choice comes first
    entries = load_embeddings()
    current = read_embedding_model() if Path(MILVUS_DB).exists() else None
    entry = pick_embedding_entry(entries, current)

    # a preexisting store is moved to the backup dir before building: the
    # move is the backup (issue #24), and building into a fresh location
    # sidesteps the dim-mismatch trap of rebuilding over an old schema
    if Path(MILVUS_DB).exists():
        confirm = input(
            f"{MILVUS_DB} already exists. Back it up and rebuild from scratch? (y/N): "
        )
        if confirm.strip().lower() != "y":
            sys.exit("Aborting.")
        dest = backup_store(MILVUS_DB, MILVUS_BACKUP_DIR)
        print(f"Backed up existing store to {dest}")
        if Path(MILVUS_DB).exists():
            sys.exit(
                f"ERROR: {MILVUS_DB} could not be fully moved -- a live "
                "connection is still holding the DB open. Close it and retry."
            )

    embeddings = build_embeddings(entry)
    # model-bound ingest tuning, not script-bound: how many documents one
    # /v1/embeddings call carries (bounded by the model's context window)
    # and how many calls run in parallel (GPU-bound locally, rate-limit
    # bound remotely) are required keys on the embeddings section of
    # models.json -- no number is right for every model, so no defaults
    BATCH_SIZE: int = entry.batch_size
    EMBED_WORKERS: int = entry.concurrency

    # dense + sparse
    #
    vector_store = Milvus(
        embedding_function=embeddings,
        # milvus package used does exact matches only on sparse search
        # (for example: the query "grappling" does not hit "grapple")
        # still, hybrid retrieval ¯\_(ツ)_/¯
        builtin_function=BM25BuiltInFunction(),
        vector_field=["dense", "sparse"],
        # Milvus has a BITCH ASS bug right here.
        # unles declared explicitly below, BM25 defaults to AUTOINDEX, which
        # incorrectly picks a DENSE index, not a SPARSE one, returning type mismatches
        # because dense vectors are static-sized while sparse vectors are varying.
        # cost me hours. fuck this guy.
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

    pdf_paths = list(Path(PDF_DIR).rglob("*.pdf"))
    print(f"Detected {CPU_COUNT} CPU cores, loading {len(pdf_paths)} PDFs...")

    # cheeky little parallelisation to make myself feel better while i doomscroll as this runs
    docs = []
    with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = {executor.submit(load_pdf, path): path for path in pdf_paths}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Loading PDFs", unit="file"
        ):
            path = futures[future]
            try:
                docs.extend(future.result())
            except Exception as exc:
                tqdm.write(f"[SKIP] {path}: {exc}")

    print(f"Loaded {len(docs)} pages, splitting into chunks...")

    # recursive text splitting because data is pdfs with non-uniform structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    split = splitter.split_documents(docs)
    # without this, i got very small chunks with book titles and keywords that
    # get very high ranking matches but dont return anything actionable.
    # this is a unique problem that fits my data, not a global tthing
    chunks = [c for c in split if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

    print(f"Dropped {len(split) - len(chunks)} chunks under {MIN_CHUNK_CHARS} chars.")
    print(f"Embedding and inserting {len(chunks)} chunks...")

    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    if batches:
        embed_and_insert_batches(vector_store, batches, EMBED_WORKERS)
        # tag the store with the model that built it, so search_books can
        # pick the matching entry and a future ingest can show which one is
        # live. the collection exists now (first batch created it)
        # only mark the store usable after every batch has been persisted
        write_embedding_model(entry.name)

    print("Done.")


if __name__ == "__main__":
    main()
