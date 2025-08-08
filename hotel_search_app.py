# === Imports ===
import streamlit as st
from sentence_transformers import SentenceTransformer
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions, QueryOptions, SearchOptions
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest, MatchAllQuery
from datetime import timedelta
import re
from config import (
    COUCHBASE_CONN_STR,
    COUCHBASE_USERNAME,
    COUCHBASE_PASSWORD,
    BUCKET_NAME,
    SCOPE_NAME,
    COLLECTION_NAME,
)
st.set_page_config(page_title="Hotel Search", layout="wide")

# === Configuration ===

VECTOR_INDEX_NAME = "vector_idx"
VECTOR_FIELD = "description_embedding"
model = SentenceTransformer("all-MiniLM-L6-v2", device ='cpu')

# === Connect to Couchbase ===

@st.cache_resource
def connect():
    cluster = Cluster(
        COUCHBASE_CONN_STR,
        ClusterOptions(PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD))
    )
    cluster.wait_until_ready(timeout=timedelta(seconds=5))
    return cluster
cluster = connect()
bucket = cluster.bucket(BUCKET_NAME)
scope = bucket.scope(SCOPE_NAME)
collection = scope.collection(COLLECTION_NAME)

# === Dropdown Values ===

@st.cache_data(ttl=3600)
def get_available_values(field: str):
    try:
        sql = f"""
        SELECT DISTINCT `{field}`
        FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
        WHERE `{field}` IS NOT NULL AND type = "hotel"
        ORDER BY `{field}`
        """
        result = cluster.query(sql)
        return [""] + [row[field] for row in result]
    except:
        return [""]
    
# === Parse Query for Semantic ===

def parse_query_components(user_query: str):
    text = user_query.lower()
    keyword_map = {
        "wifi": "free_internet", "internet": "free_internet", "wi-fi": "free_internet",
        "pet friendly": "pets_ok", "pets allowed": "pets_ok", "pet-friendly": "pets_ok",
        "free parking": "free_parking", "parking": "free_parking",
        "breakfast": "free_breakfast", "free breakfast": "free_breakfast"
    }
    filters = {}
    for keyword, field in keyword_map.items():
        if keyword in text:
            filters[field] = True
            text = text.replace(keyword, " ")
    semantic_query = re.sub(r'\b(with|and|the|a|an|in|at|for|to|from)\b', ' ', text)
    semantic_query = re.sub(r'\s+', ' ', semantic_query).strip()
    return semantic_query, filters

# === SQL++ Filtering ===

def get_filtered_hotel_ids(city, state, free_parking, free_breakfast, vacancy, parsed_filters):
    filters = ["type = 'hotel'"]
    params = {}
    if city:
        filters.append("LOWER(city) = LOWER($city)")
        params["city"] = city
    if state:
        filters.append("LOWER(state) = LOWER($state)")
        params["state"] = state
    if free_parking or parsed_filters.get('free_parking'):
        filters.append("free_parking = TRUE")
    if free_breakfast or parsed_filters.get('free_breakfast'):
        filters.append("free_breakfast = TRUE")
    if vacancy:
        filters.append("vacancy = TRUE")
    for amenity in ['free_internet', 'pets_ok']:
        if parsed_filters.get(amenity):
            filters.append(f"{amenity} = TRUE")
    where_clause = " AND ".join(filters)
    sql = f"""
        SELECT META().id AS id, name, city, country, description, price, vacancy,
               free_parking, free_breakfast, free_internet, pets_ok
        FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
        WHERE {where_clause}
        LIMIT 1000
    """
    try:
        result = cluster.query(sql, QueryOptions(named_parameters=params, metrics=False))
        hotels = list(result)
        ids = {h['id'] for h in hotels}
        return ids, hotels
    except Exception:
        return set(), []
    
# === Vector Search ===

def vector_search(semantic_query: str, *, limit: int = 200, num_candidates: int = 500):
    if not semantic_query:
        return []
    try:
        emb = model.encode(semantic_query, normalize_embeddings=True).tolist()
        req = SearchRequest.create(MatchAllQuery()).with_vector_search(
            VectorSearch.from_vector_query(
                VectorQuery(field_name=VECTOR_FIELD, vector=emb, num_candidates=500)
            )
        )
        res = scope.search(VECTOR_INDEX_NAME, req, SearchOptions(limit=200, timeout=timedelta(seconds=4)))
        return [{'id': r.id, 'score': r.score} for r in res.rows()]
    except Exception as e:
        st.error(f"Vector search error: {e}")
        return []
    
# === Hydrate and pull necessary fields from documents to display on the UI ===

def hydrate_docs_by_ids(ids):
    if not ids:
        return {}
    sql = f"""
      SELECT META().id AS id, name, city, state, country, description, price, vacancy,
             free_parking, free_breakfast, free_internet, pets_ok
      FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
      USE KEYS $keys
    """
    rows = cluster.query(
        sql,
        QueryOptions(named_parameters={"keys": list(ids)})
    )
    return {row["id"]: row for row in rows}

# === Hybrid Search ===

def hybrid_search(query, city, state, free_parking, free_breakfast, vacancy, limit=10):
    semantic_query, parsed_filters = parse_query_components(query)
    # 1) Structured filter via SQL++
    filtered_ids, _ = get_filtered_hotel_ids(
        city, state, free_parking, free_breakfast, vacancy, parsed_filters
    )
    if not filtered_ids:
        return []
    # 2) Vector search & intersection
    vec_score = {}
    if semantic_query:
        vec_hits = vector_search(semantic_query, limit=500)  # large pool for intersection
        vec_score = {h['id']: h['score'] for h in vec_hits}
        vec_order = [h['id'] for h in vec_hits]
        # Intersect with structured filter results
        final_ids = [hid for hid in vec_order if hid in filtered_ids][:limit]
        # Fallback to SQL++ if no overlap
        if not final_ids:
            final_ids = list(filtered_ids)[:limit]
    else:
        final_ids = list(filtered_ids)[:limit]
    # 3) Hydrate documents
    docs = hydrate_docs_by_ids(final_ids)
    # 4) Assemble results
    results = []
    for hid in final_ids:
        results.append({
            "id": hid,
            "fields": {k: v for k, v in docs.get(hid, {}).items() if k != "id"},
            "score": float(vec_score.get(hid, 0.0))
        })
    # Vector ordering
    if semantic_query:
        results.sort(key=lambda x: x['score'], reverse=True)
    return results

# === UI ===

st.title(":hotel: Hotel Search (Hybrid SQL++ + Vector Search)")
col1, col2 = st.columns([2, 1])
with col1:
    query = st.text_input("Describe your stay:", placeholder="e.g. luxury hotel with free wifi and pool")
with col2:
    cities = get_available_values("city")
    states = get_available_values("state")
    city = st.selectbox("City", cities)
    state = st.selectbox("State", states)
    col2a, col2b = st.columns(2)
    with col2a:
        free_parking = st.checkbox("Free Parking")
        vacancy = st.checkbox("Vacancy")
    with col2b:
        free_breakfast = st.checkbox("Free Breakfast")
if st.button(":mag: Search"):
    with st.spinner("Searching..."):
        results = hybrid_search(query, city, state, free_parking, free_breakfast, vacancy)
        if results:
            st.success(f":white_check_mark: Found {len(results)} hotels")
            for i, result in enumerate(results, 1):
                f = result["fields"]
                st.markdown(f"### {i}. {f.get('name', 'Unnamed')}")
                st.markdown(f"- :round_pushpin: **Location**: {f.get('city')}, {f.get('country')}")
                st.markdown(f"- :moneybag: **Price**: {f.get('price')}" if f.get('price') else "- :moneybag: **Price**: Not specified")
                st.markdown(f"- :star: **Score**: {result['score']:.3f}")
                st.markdown(f"- :bed: **Vacancy**: {':white_check_mark: Available' if f.get('vacancy') else ':x: Not Available'}")
                amenities = []
                if f.get('free_parking'): amenities.append(":parking: Parking")
                if f.get('free_breakfast'): amenities.append(":croissant: Breakfast")
                if f.get('free_internet'): amenities.append(":signal_strength: WiFi")
                if f.get('pets_ok'): amenities.append(":dog2: Pet Friendly")
                if amenities:
                    st.markdown(f"- :dart: **Amenities**: {', '.join(amenities)}")
                desc = f.get('description', '')
                if desc:
                    st.markdown(f"_{desc}_")
                st.markdown("---")
        else:
            st.info("No results. Try removing some filters or using broader terms.")