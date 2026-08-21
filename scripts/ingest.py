import os
import sys
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

from langchain_core.documents import Document

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_milvus import Milvus, BM25BuiltInFunction

# this is run straight from a shell, so the project root has to go on the path
# before anything under llmtui can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmtui.config import (
    DENSE_INDEX_TYPE,
    DENSE_METRIC_TYPE,
    EMBEDDING_MODEL,
    SPARSE_INDEX_TYPE,
    SPARSE_METRIC_TYPE,
)
from llmtui.config import MILVUS_DEV_DB_PATH as MILVUS_DB

parser = argparse.ArgumentParser(
    description="Ingest a directory of PDFs into the dev vector store."
)
parser.add_argument("pdf_dir", help="directory to search recursively for PDFs")
args = parser.parse_args()

PDF_DIR: str = args.pdf_dir
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150
# art-heavy pages extract to nothing but a running header + page number
# those tiny chunks embed almost purely as the book title, so any query
# naming the book matches them ahead of real content.
# i drop them by filtering by chunk size
# it might be defeating the purpose of BM25 document length normalisation, i dont know
MIN_CHUNK_CHARS: int = 50
BATCH_SIZE: int = 64
CPU_COUNT: int = os.cpu_count()
# threads, not processes: this waits on ollama over http, so the GIL is
# released. serial batches leave the GPU idle during each milvus insert;
# overlapping them saturates it. measured 62 -> 108 chunks/s, and 8 workers
# was no better than 4.
EMBED_WORKERS: int = 4


# checked before the wipe prompt: a bad path should not cost you a database
if not Path(PDF_DIR).is_dir():
    parser.error(f"{PDF_DIR} is not a directory")

# check for protecting previous db
if Path(MILVUS_DB).exists():
    confirm = input(f"{MILVUS_DB} already exists. Wipe it before ingesting to avoid duplicates? (y/N): ")
    if confirm.strip().lower() == "y":
        shutil.rmtree(MILVUS_DB)
        print("Wiped existing vector store.")
    else:
        sys.exit("Aborting.")
        

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

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
        {"metric_type": DENSE_METRIC_TYPE, "index_type": DENSE_INDEX_TYPE, "params": {}},
        {"metric_type": SPARSE_METRIC_TYPE, "index_type": SPARSE_INDEX_TYPE, "params": {}},
    ],
    connection_args={"uri": MILVUS_DB}
)

pdf_paths = list(Path(PDF_DIR).rglob("*.pdf"))
print(f"Detected {CPU_COUNT} CPU cores, loading {len(pdf_paths)} PDFs...")


def load_pdf(path: Path) -> list[Document]:
    return PyMuPDFLoader(str(path)).load()


# cheeky little parallelisation to make myself feel better while i doomscroll as this runs
docs = []
with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
    futures = {executor.submit(load_pdf, path): path for path in pdf_paths}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs", unit="file"):
        path = futures[future]
        try:
            docs.extend(future.result())
        except Exception as exc:
            tqdm.write(f"[SKIP] {path}: {exc}")

print(f"Loaded {len(docs)} pages, splitting into chunks...")

# recursive text splitting because data is pdfs with non-uniform structure
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
split = splitter.split_documents(docs)
# without this, i got very small chunks with book titles and keywords that
# get very high ranking matches but dont return anything actionable.
# this is a unique problem that fits my data, not a global tthing
chunks = [c for c in split if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]

print(f"Dropped {len(split) - len(chunks)} chunks under {MIN_CHUNK_CHARS} chars.")
print(f"Embedding and inserting {len(chunks)} chunks...")

batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

if batches:
    # i frontload and create the collection and its schema.
    # otherwise there is a race condition below raising DataNotMatchException
    vector_store.add_documents(batches[0])
    # cheeky little parallelisation again. god forbid i doom scroll too long
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as executor:
        futures = {executor.submit(vector_store.add_documents, b): n
                   for n, b in enumerate(batches[1:], start=1)}
        for future in tqdm(as_completed(futures), total=len(batches), initial=1,
                           desc="Embedding+inserting", unit="batch"):
            try:
                future.result()
            except Exception as exc:
                tqdm.write(f"[FAILED] batch {futures[future]}: {exc}")

print("Done.")
