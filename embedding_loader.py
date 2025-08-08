from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions
from couchbase.subdocument import upsert  
from sentence_transformers import SentenceTransformer

from config import (
    COUCHBASE_CONN_STR,
    COUCHBASE_USERNAME,
    COUCHBASE_PASSWORD,
    BUCKET_NAME,
    SCOPE_NAME,
    COLLECTION_NAME
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Couchbase
cluster = Cluster(
    COUCHBASE_CONN_STR,
    ClusterOptions(PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD))
)
cluster.wait_until_ready(timeout=timedelta(seconds=5))
bucket = cluster.bucket(BUCKET_NAME)
collection = bucket.scope(SCOPE_NAME).collection(COLLECTION_NAME)

# Fetch documents that need embeddings
query = f"""
SELECT META().id, description
FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
WHERE description IS NOT MISSING 
  AND description != ""
  AND description_embedding IS MISSING
"""

try:
    rows = cluster.query(query, QueryOptions(metrics=True))
    
    for row in rows:
        doc_id = row["id"]
        desc = row["description"]
        
        # Generate embedding
        embedding = model.encode(desc, normalized_embedding = True).tolist()
        
        # Update document with embedding
        collection.mutate_in(
            doc_id,
            [upsert("description_embedding", embedding)]
        )
        print(f"Updated {doc_id}")

except Exception as e:
    print(f"Error during execution: {e}")
    raise
