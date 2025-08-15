import re
import streamlit as st
from datetime import timedelta
from sentence_transformers import SentenceTransformer
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, QueryOptions, SearchOptions
from couchbase.search import TermQuery, ConjunctionQuery, BooleanFieldQuery, MatchQuery, SearchRequest
from couchbase.vector_search import VectorQuery, VectorSearch
from config import (
    COUCHBASE_CONN_STR,
    COUCHBASE_USERNAME,
    COUCHBASE_PASSWORD,
    BUCKET_NAME,
    SCOPE_NAME,
    COLLECTION_NAME,
    VECTOR_INDEX_NAME,
    VECTOR_FIELD
)
# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(page_title="Hotel Search", layout="wide")
st.title(":hotel: Hotel Search")

# ── Caching Embedding Model and Couchbase Cluster ──────────
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
@st.cache_resource
def connect():
    cluster = Cluster(
        COUCHBASE_CONN_STR,
        ClusterOptions(PasswordAuthenticator(COUCHBASE_USERNAME, COUCHBASE_PASSWORD))
    )
    cluster.wait_until_ready(timeout=timedelta(seconds=10))
    return cluster
embedding_model = load_embedding_model()
cluster = connect()
collection = cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME).collection(COLLECTION_NAME)

# ── Dropdown Options ───────────────────────────────────────
@st.cache_data(ttl=3600)
def get_available_values(field: str):
    sql = f"""
        SELECT DISTINCT `{field}`
        FROM `{BUCKET_NAME}`.`{SCOPE_NAME}`.`{COLLECTION_NAME}`
        WHERE `{field}` IS NOT NULL AND `{field}` != ""
        ORDER BY `{field}`
    """
    result = cluster.query(sql, QueryOptions(metrics=False))
    return [""] + [row[field] for row in result if row[field]]

# ── NLP Query Parsing ──────────────────────────────────────
def parse_query_components(user_query: str):
    text = user_query.lower()
    keyword_map = {
        "wifi": "free_internet", "internet": "free_internet", "wi-fi": "free_internet",
        "pet friendly": "pets_ok", "pets allowed": "pets_ok", "pet-friendly": "pets_ok",
        "free parking": "free_parking", "parking": "free_parking",
        "breakfast": "free_breakfast", "free breakfast": "free_breakfast",
    }
    filters = {}
    cleaned = text
    for kw, field in keyword_map.items():
        if kw in cleaned:
            filters[field] = True
            cleaned = cleaned.replace(kw, " ")
    semantic_query = re.sub(r'\b(with|and|the|a|an|in|at|for|to|from|near)\b', ' ', cleaned)
    semantic_query = re.sub(r'\s+', ' ', semantic_query).strip()
    return semantic_query, filters

# ── FTS Filter Builder ─────────────────────────────────────
def build_fts_filters(city, state, ui_flags, parsed_flags):
    clauses = []
    if city and city.strip():
        clauses.append(TermQuery(city.strip(), field="city"))
    if state and state.strip():
        clauses.append(TermQuery(state.strip(), field="state"))
    def add_bool(field_name, flag):
        if flag:
            clauses.append(BooleanFieldQuery(True, field=field_name))
    add_bool("free_parking", ui_flags.get("free_parking") or parsed_flags.get("free_parking"))
    add_bool("free_breakfast", ui_flags.get("free_breakfast") or parsed_flags.get("free_breakfast"))
    add_bool("vacancy", ui_flags.get("vacancy"))
    add_bool("free_internet", parsed_flags.get("free_internet"))
    add_bool("pets_ok", parsed_flags.get("pets_ok"))
    return ConjunctionQuery(*clauses) if clauses else None

# ── Hybrid Search ──────────────────────────────────────────
def search_couchbase(cluster, index_name, embedding_client, embedding_key, search_text, k=20, fts_filter=None):
    try:
        # 1. Create embedding vector
        vector_embedding = embedding_client.encode(search_text).tolist()

        # 2. Build VectorQuery with optional prefilter
        vector_query = VectorQuery(
            field_name=embedding_key,
            vector=vector_embedding,
            num_candidates=k,
            prefilter=fts_filter
        )
        # 3. Wrap in SearchRequest
        search_req = SearchRequest.create(
            VectorSearch.from_vector_query(vector_query)
        )
        # 4. Run search
        scope = cluster.bucket(BUCKET_NAME).scope(SCOPE_NAME)
        results = scope.search(index_name, search_req, SearchOptions(limit=k, fields=["*"]))
        return [{**(row.fields or {}), "score": row.score} for row in results.rows()]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []
    
# ── UI Inputs ──────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    user_query = st.text_input("Describe your stay:", placeholder="e.g. luxury hotel with wifi", value="luxury hotel")
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

# ── Run Search ─────────────────────────────────────────────
if st.button(":mag: Search"):
    with st.spinner("Searching..."):
        semantic_query, parsed_flags = parse_query_components(user_query)
        fts_filter = build_fts_filters(
            city, state,
            ui_flags={"free_parking": free_parking, "free_breakfast": free_breakfast, "vacancy": vacancy},
            parsed_flags=parsed_flags
        )
        hits = search_couchbase(
            cluster=cluster,
            index_name=VECTOR_INDEX_NAME,
            embedding_client=embedding_model,
            embedding_key=VECTOR_FIELD,
            search_text=semantic_query,
            k=20,
            fts_filter=fts_filter
        )
        # Fallback: if filters too strict
        if not hits and fts_filter is not None:
            st.write(":arrows_counterclockwise: **Trying without filters...**")
            hits = search_couchbase(
                cluster=cluster,
                index_name=VECTOR_INDEX_NAME,
                embedding_client=embedding_model,
                embedding_key=VECTOR_FIELD,
                search_text=semantic_query,
                k=20,
                fts_filter=None
            )
        # ── Display Results ────────────────────────────────
        if hits:
            st.success(f":white_check_mark: Found {len(hits)} hotels")
            for i, row in enumerate(hits, 1):
                st.markdown(f"### {i}. {row.get('name', 'Unnamed')}")
                st.markdown(f"_{row.get('description', '')}_")
                st.markdown(f"- :round_pushpin: **Location**: {row.get('city', '—')}, {row.get('country', '—')}")
                st.markdown(f"- :moneybag: **Price**: {row.get('price', 'Not specified')}")
                st.markdown(f"- :star: **Score**: {row.get('score', 0.0):.3f}")
                vacancy_text = ":white_check_mark: Available" if row.get("vacancy") else ":x: Not Available"
                st.markdown(f"- :bed: **Vacancy**: {vacancy_text}")
                amenities = []
                if row.get("free_parking"): amenities.append(":parking: Parking")
                if row.get("free_breakfast"): amenities.append(":croissant: Breakfast")
                if row.get("free_internet"): amenities.append(":signal_strength: WiFi")
                if row.get("pets_ok"): amenities.append(":dog2: Pet Friendly")
                if amenities:
                    st.markdown(f"- :dart: **Amenities**: {', '.join(amenities)}")
        else:
            st.info("No results. Try removing filters or using broader terms.")