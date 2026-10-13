import html
import os

import requests
import streamlit as st

# --- Config ---
# 127.0.0.1, not "localhost": on Windows, "localhost" tries IPv6 first and uvicorn only
# listens on IPv4, which adds a ~2 s timeout to every request
API_URL = os.getenv("SEARCH_API_URL", "http://127.0.0.1:8002/search")
EXAMPLE_QUERIES = [
    "quiet bathroom exhaust fan",
    "wireless noise cancelling headphones",
    "gift for a coffee lover",
    "waterproof hiking boots",
]

st.set_page_config(
    page_title="AI Product Search",
    page_icon="🔍",
    layout="centered",
)

# --- Styles ---
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 3rem; max-width: 820px;}

    .hero {text-align: center; margin-bottom: 2rem;}
    .badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
        border-radius: 999px;
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        margin: 0 0 0.75rem 0;
        color: #0F172A;
    }
    .hero h1 span {
        background: linear-gradient(90deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {color: #475569; font-size: 1.05rem; margin: 0 auto; max-width: 600px;}

    div[data-testid="stForm"] {
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        background: #FFFFFF;
        padding: 1rem 1.1rem 0.4rem 1.1rem;
        box-shadow: 0 4px 24px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stForm"] input {font-size: 1.05rem;}

    .section-label {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.5rem 0 0.5rem 0;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }
    .card:hover {border-color: #C7D2FE; box-shadow: 0 6px 20px rgba(79, 70, 229, 0.08);}
    .card-top {display: flex; justify-content: space-between; align-items: center; gap: 1rem;}
    .card-title {font-weight: 650; color: #0F172A; font-size: 1.02rem; margin: 0;}
    .card-desc {color: #475569; font-size: 0.92rem; margin: 0.45rem 0 0.6rem 0; line-height: 1.5;}
    .card-meta {color: #94A3B8; font-size: 0.8rem; font-family: ui-monospace, monospace;}
    .source {
        display: inline-block;
        margin-left: 0.4rem;
        padding: 0.05rem 0.5rem;
        border-radius: 999px;
        background: #F1F5F9;
        color: #475569;
        font-family: inherit;
        font-size: 0.72rem;
    }
    .score {
        flex-shrink: 0;
        background: #ECFDF5;
        color: #047857;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .features {display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem; margin-top: 2.5rem;}
    .feature {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem;
    }
    .feature .icon {font-size: 1.4rem;}
    .feature h4 {margin: 0.4rem 0 0.25rem 0; font-size: 0.95rem; color: #0F172A;}
    .feature p {margin: 0; color: #64748B; font-size: 0.85rem; line-height: 1.45;}
    @media (max-width: 640px) {
        .features {grid-template-columns: 1fr;}
        .hero h1 {font-size: 2rem;}
    }

    .footer {text-align: center; color: #94A3B8; font-size: 0.8rem; margin-top: 3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Backend call ---
def search_products(query: str, top_k: int) -> dict:
    """Query the search API: {"results": [{product_id, description, score, ...}], "took_ms"}."""
    response = requests.get(API_URL, params={"query": query, "top_k": top_k}, timeout=15)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {"results": data}


def split_description(text: str) -> tuple[str, str]:
    """Product descriptions are title + bullet points; use the first line as the title."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "Untitled product", ""
    title = lines[0] if len(lines[0]) <= 120 else lines[0][:117] + "..."
    body = " · ".join(lines[1:])
    if len(body) > 260:
        body = body[:257] + "..."
    return title, body


def render_result(item: dict) -> None:
    title, body = split_description(item.get("description", ""))
    score = item.get("score")
    score_html = f'<span class="score">{float(score) * 100:.0f}% match</span>' if score is not None else ""
    # Which first-stage retriever(s) found this product
    sources = [label for key, label in (("vector_score", "semantic"), ("lexical_score", "keyword")) if key in item]
    sources_html = "".join(f'<span class="source">{label}</span>' for label in sources)
    st.markdown(
        f"""
        <div class="card">
            <div class="card-top">
                <p class="card-title">{html.escape(title)}</p>
                {score_html}
            </div>
            <p class="card-desc">{html.escape(body)}</p>
            <span class="card-meta">ID · {html.escape(str(item.get("product_id", "—")))}{sources_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Hero ---
st.markdown(
    """
    <div class="hero">
        <div class="badge">✨ AI-Based Search System for E-Commerce</div>
        <h1>Find products by <span>meaning</span>,<br/>not just keywords</h1>
        <p>Describe what you're looking for in plain language. Our semantic search engine
        understands intent and returns the most relevant products from the catalog.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Search bar ---
if "query" not in st.session_state:
    st.session_state.query = ""

# An example button was clicked on the previous run: fill the search bar and run it
run_example = "pending_query" in st.session_state
if run_example:
    st.session_state.query = st.session_state.pop("pending_query")

with st.form("search_form", clear_on_submit=False):
    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    with col_input:
        query = st.text_input(
            "Search",
            key="query",
            placeholder="e.g. lightweight laptop for travel and video editing",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)
    top_k = st.slider("Number of results", 5, 50, 10, step=5)

# --- Example queries ---
st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
example_cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(example_cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state.pending_query = example
        st.rerun()

submitted = submitted or run_example

# --- Results ---
if submitted and query.strip():
    with st.spinner("Searching with AI..."):
        try:
            response = search_products(query.strip(), top_k)
            results = response.get("results", [])
        except requests.exceptions.ConnectionError:
            st.warning(
                f"Can't reach the search API at `{API_URL}`. Start the backend, "
                "or set the `SEARCH_API_URL` environment variable."
            )
            results = None
        except Exception as error:
            st.error(f"Search failed: {error}")
            results = None

    if results is not None:
        took = f" · {response['took_ms']} ms" if "took_ms" in response else ""
        st.markdown(
            f'<div class="section-label">{len(results)} results for “{html.escape(query.strip())}”{took}</div>',
            unsafe_allow_html=True,
        )
        if not results:
            st.info("No matching products found. Try describing it differently.")
        for item in results:
            render_result(item)
elif submitted:
    st.info("Type something to search.")
else:
    # --- How it works ---
    st.markdown(
        """
        <div class="features">
            <div class="feature">
                <div class="icon">🧠</div>
                <h4>Semantic understanding</h4>
                <p>Queries are encoded with a Sentence-Transformer model (MiniLM) to capture meaning.</p>
            </div>
            <div class="feature">
                <div class="icon">⚡</div>
                <h4>Hybrid retrieval</h4>
                <p>Vector search (pgvector) and keyword search run together and are fused into one ranked list.</p>
            </div>
            <div class="feature">
                <div class="icon">🛍️</div>
                <h4>Cross-encoder reranking</h4>
                <p>A second model (MS MARCO MiniLM) reads query and product together to verify and shortlist the best matches.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">AI-Based Search System for E-Commerce Platforms · '
    "Powered by Sentence Transformers, cross-encoder reranking &amp; pgvector</div>",
    unsafe_allow_html=True,
)
