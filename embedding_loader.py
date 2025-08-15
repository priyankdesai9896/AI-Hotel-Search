from datetime import timedelta
import time
from typing import List
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions
from couchbase.subdocument import upsert
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    COUCHBASE_CONN_STR,
    COUCHBASE_USERNAME,
    COUCHBASE_PASSWORD,
    BUCKET_NAME,
    SCOPE_NAME,
    COLLECTION_NAME,
)

# --- Config ---

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 200          # how many docs to embed per batch
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 0.5

# --- Init embeddings (normalize for dot_product) ---

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

# --- Connect to Couchbase ---

cluster = Cluster(
    COUCHBASE_CONN_STR,
    ClusterOptions(PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD))
)
cluster.wait_until_ready(timeout=timedelta(seconds=5))
bucket = cluster.bucket(BUCKET_NAME)
collection = bucket.scope(SCOPE_NAME).collection(COLLECTION_NAME)

# --- Fetch docs that need vectors ---

query = f"""
SELECT META().id AS id, description
FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
WHERE type = "hotel"
  AND description IS NOT MISSING
  AND description != ""
  AND description_embedding IS MISSING
"""

rows = list(cluster.query(query, QueryOptions(metrics=False)))
print(f"Found {len(rows)} docs missing embeddings")
def chunks(lst: List[dict], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# --- Process in batches ---

updated = 0
for batch in chunks(rows, BATCH_SIZE):
    ids = [r["id"] for r in batch]
    texts = [r["description"] for r in batch]
    # Embed in one go (faster)
    vecs = embeddings.embed_documents(texts)  # returns List[List[float]]
    for doc_id, vec in zip(ids, vecs):
        # Retry mutate_in a few times if transient error happens
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                collection.mutate_in(doc_id, [upsert("description_embedding", vec)])
                updated += 1
                break
            except Exception as e:
                if attempt == RETRY_ATTEMPTS:
                    print(f"[SKIP] {doc_id} after {RETRY_ATTEMPTS} attempts: {e}")
                else:
                    time.sleep(RETRY_BACKOFF_SEC * attempt)
print(f"Done. Updated {updated} documents.")
