import json
import time
from pathlib import Path
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS — refined dark-finance aesthetic
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main { background-color: #0d1117; color: #e6edf3; }

    .stTextInput > div > div > input {
        background: #161b22;
        border: 1px solid #30363d;
        color: #e6edf3;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stButton > button {
        background: linear-gradient(135deg, #238636, #1f7a2e);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }

    .answer-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #238636;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }

    .metric-pill {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #58a6ff;
    }

    .confidence-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }

    .thought-step {
        background: #0d1117;
        border-left: 3px solid #30363d;
        padding: 6px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #8b949e;
    }

    .ragas-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.4rem;
        font-weight: 600;
    }

    .source-ref {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #8b949e;
    }

    h1 { font-weight: 600; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 400; color: #c9d1d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📊 Agent Intelligence Panel")
    st.markdown("---")

    # ── Ragas Evaluation Scores ──────────────────────────────
    st.markdown("### 🧪 Ragas Eval Scores")
    ragas_path = Path("./ragas_results.json")
    if ragas_path.exists():
        ragas_data = json.loads(ragas_path.read_text())
        f_score = ragas_data.get("faithfulness", 0.0)
        r_score = ragas_data.get("answer_relevancy", 0.0)

        col1, col2 = st.columns(2)
        with col1:
            color = "#238636" if f_score >= 0.8 else "#f0883e" if f_score >= 0.6 else "#f85149"
            st.markdown(
                f"""<div style='text-align:center'>
                <div class='ragas-score' style='color:{color}'>{f_score:.2f}</div>
                <div style='font-size:0.75rem;color:#8b949e'>Faithfulness</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            color = "#238636" if r_score >= 0.8 else "#f0883e" if r_score >= 0.6 else "#f85149"
            st.markdown(
                f"""<div style='text-align:center'>
                <div class='ragas-score' style='color:{color}'>{r_score:.2f}</div>
                <div style='font-size:0.75rem;color:#8b949e'>Answer Rel.</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.caption("Scores from latest offline evaluation run.")
    else:
        st.info("Run `python app.py --evaluate` to generate Ragas scores.")

    st.markdown("---")

    # ── Thought Log (populated after a query runs) ───────────
    st.markdown("### 🧠 Agent Thought Process")
    thought_placeholder = st.empty()

    if "thought_log" in st.session_state and st.session_state.thought_log:
        with thought_placeholder.container():
            for step in st.session_state.thought_log:
                st.markdown(
                    f"<div class='thought-step'>{step}</div>",
                    unsafe_allow_html=True,
                )
    else:
        thought_placeholder.caption("Thought log will appear here after a query.")

    st.markdown("---")
    st.markdown("### ⚙️ Config")
    st.caption("Index: `financial-rag`")
    st.caption("Model: `gpt-4o-mini`")
    st.caption("Embeddings: `text-embedding-3-small`")
    st.caption("Max self-correction retries: `2`")


# ─────────────────────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────────────────────

st.markdown("# 📈 Financial Analyst Agent")
st.markdown(
    "Enterprise-grade **Reliable RAG** system — LangGraph · Pinecone · Ragas"
)
st.markdown("---")

# ── Sample questions ─────────────────────────────────────────
SAMPLES = [
    "What was Apple's revenue and EPS in Q3 2024?",
    "How did Microsoft Azure grow in Q4 FY2024?",
    "What is Nvidia's current stock price and recent earnings guidance?",
    "Compare Amazon AWS revenue growth across the last two quarters.",
]

st.markdown("**Try a sample question:**")
cols = st.columns(len(SAMPLES))
for i, sample in enumerate(SAMPLES):
    if cols[i].button(sample[:40] + "…", key=f"sample_{i}"):
        st.session_state.prefill_query = sample

# ── Query Input ───────────────────────────────────────────────
prefill = st.session_state.get("prefill_query", "")
query = st.text_input(
    "Ask a financial question",
    value=prefill,
    placeholder="e.g. What was Apple's EPS in Q3 2024?",
    key="query_input",
)

run_btn = st.button("🔍 Run Analysis", type="primary")

# ── Response Area ─────────────────────────────────────────────
if run_btn and query.strip():
    # Clear previous prefill
    if "prefill_query" in st.session_state:
        del st.session_state["prefill_query"]

    with st.spinner("Routing query → Retrieving context → Grading → Generating…"):
        try:
            from graph_logic import run_query

            state = run_query(query.strip())
            answer = state.get("final_answer")

            # Persist thought log to sidebar
            st.session_state.thought_log = state.get("thought_log", [])
            # Force sidebar refresh
            st.rerun()

        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()

    if answer:
        # ── Confidence Badge ──────────────────────────────────
        conf = answer.confidence_score
        badge_color = (
            "#238636" if conf >= 0.75
            else "#f0883e" if conf >= 0.5
            else "#f85149"
        )
        badge_label = (
            "HIGH CONFIDENCE" if conf >= 0.75
            else "MEDIUM CONFIDENCE" if conf >= 0.5
            else "LOW CONFIDENCE"
        )

        st.markdown(
            f"""<span class='confidence-badge' style='background:{badge_color}22;
            border:1px solid {badge_color};color:{badge_color}'>
            {badge_label} {conf:.0%}</span>""",
            unsafe_allow_html=True,
        )

        # ── Answer Card ───────────────────────────────────────
        st.markdown(
            f"<div class='answer-card'>{answer.answer}</div>",
            unsafe_allow_html=True,
        )

        # ── Key Metrics ───────────────────────────────────────
        if answer.key_metrics:
            st.markdown("#### 📈 Extracted Key Metrics")
            metrics_html = "".join(
                f"<span class='metric-pill'>{k}: {v}</span>"
                for k, v in answer.key_metrics.items()
            )
            st.markdown(metrics_html, unsafe_allow_html=True)

        # ── Sources ───────────────────────────────────────────
        if answer.source_documents_used:
            st.markdown("#### 📎 Sources Used")
            for src in answer.source_documents_used:
                st.markdown(
                    f"<div class='source-ref'>↳ {src}</div>",
                    unsafe_allow_html=True,
                )

        # ── Raw JSON (expandable) ─────────────────────────────
        with st.expander("View raw structured JSON output"):
            st.json(answer.model_dump())

elif run_btn and not query.strip():
    st.warning("Please enter a question first.")

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built with LangGraph · LangChain · Pinecone · Ragas | "
    "Portfolio project — Enterprise Reliable RAG Financial Agent"
)
