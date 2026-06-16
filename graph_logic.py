import os
import json
import logging
from typing import Annotated, Literal, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 1.  PYDANTIC SCHEMAS  (strict JSON output contract)
# ─────────────────────────────────────────────────────────────

class FinancialAnswer(BaseModel):
    """
    Strict JSON contract for every answer produced by the agent.
    The PydanticOutputParser enforces this at generation time.
    """
    answer: str = Field(
        description="The complete, factual answer to the user's financial question."
    )
    source_documents_used: list[str] = Field(
        description=(
            "List of source references used (e.g. document titles, "
            "URLs, or chunk IDs)."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "A self-assessed confidence score between 0.0 and 1.0 "
            "reflecting how well the retrieved context supports the answer."
        ),
    )
    key_metrics: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional dictionary of extracted financial KPIs, "
            "e.g. {'Revenue': '$94.9B', 'EPS': '$1.53'}."
        ),
    )


class RouteDecision(BaseModel):
    """Decision produced by the Multi-Query Router node."""
    query_variations: list[str] = Field(
        description="Exactly 3 alternative phrasings of the original query."
    )
    route: Literal["vector", "web", "both"] = Field(
        description=(
            "'vector' for historical earnings data, "
            "'web' for real-time prices/news, "
            "'both' when both data sources are needed."
        )
    )


class GradeDecision(BaseModel):
    """Binary grade produced by the Grade Gate node."""
    grade: Literal["relevant", "irrelevant"] = Field(
        description=(
            "'relevant' if the retrieved context sufficiently answers "
            "the query; 'irrelevant' otherwise."
        )
    )
    reason: str = Field(description="One-sentence justification for the grade.")


# ─────────────────────────────────────────────────────────────
# 2.  AGENT STATE  (typed dict that flows through the graph)
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Mutable state passed between every node.
    `add_messages` is a LangGraph reducer — it appends rather than
    overwrites the messages list, enabling a full conversation trace.
    """
    # Conversation history (user + assistant turns)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Original user query
    original_query: str

    # 3 alternative query variations produced by the router
    query_variations: list[str]

    # Tool routing decision: "vector" | "web" | "both"
    route: str

    # Raw retrieved context chunks (combined from all tools)
    retrieved_context: list[str]

    # Source references (filenames, URLs, chunk IDs)
    source_references: list[str]

    # Number of self-correction retry attempts (max = 2)
    retry_count: int

    # Final structured answer (populated in the Generation node)
    final_answer: FinancialAnswer | None

    # Internal thought-process log for the Streamlit sidebar
    thought_log: list[str]


# ─────────────────────────────────────────────────────────────
# 3.  LLM & TOOL INITIALISATION
# ─────────────────────────────────────────────────────────────

def build_llm(temperature: float = 0.0) -> ChatGroq:
    """Return a deterministic GPT-4o-mini instance."""
    return ChatGroq(
        model="gpt-4o-mini",
        temperature=temperature,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


def build_embeddings() -> HuggingFaceEmbeddings:
    """text-embedding-3-small — cost-effective, high-quality."""
    return HuggingFaceEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )


def build_pinecone_retriever(
    index_name: str = "financial-rag",
    top_k: int = 6,
) -> PineconeVectorStore:
    """
    Connect to an existing Pinecone index and return a retriever.
    Metadata filters can be applied at query time via `search_kwargs`.
    """
    embeddings = build_embeddings()
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def build_web_search_tool(max_results: int = 4) -> TavilySearchResults:
    """
    Tavily search — real-time financial news and stock prices.
    Falls back gracefully if TAVILY_API_KEY is not set (returns mock).
    """
    return TavilySearchResults(
        max_results=max_results,
        tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
    )


# ─────────────────────────────────────────────────────────────
# 4.  INDIVIDUAL GRAPH NODES
# ─────────────────────────────────────────────────────────────

# ── 4a. Multi-Query Router ────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a financial query analyst. Given a user question:
1. Generate EXACTLY 3 alternative phrasings that improve vector-search recall
   (e.g. use synonyms, rephrase as a fact, include ticker symbols).
2. Decide the best data source:
   - "vector"  → the question is about historical earnings, filings, or fundamentals
   - "web"     → the question requires real-time stock price or very recent news
   - "both"    → the question needs historical context AND live data

Respond ONLY with valid JSON matching this schema:
{format_instructions}""",
        ),
        ("human", "{query}"),
    ]
)


def node_multi_query_router(state: AgentState) -> dict:
    """
    Node 1 — Multi-Query Router
    ───────────────────────────
    Expands the user query into 3 variations and selects the
    appropriate retrieval route.
    """
    logger.info("NODE │ multi_query_router")
    llm = build_llm()
    parser = PydanticOutputParser(pydantic_object=RouteDecision)

    chain = ROUTER_PROMPT | llm | parser
    decision: RouteDecision = chain.invoke(
        {
            "query": state["original_query"],
            "format_instructions": parser.get_format_instructions(),
        }
    )

    log_entry = (
        f"[Router] Route='{decision.route}' | "
        f"Variations: {decision.query_variations}"
    )
    logger.info(log_entry)

    return {
        "query_variations": decision.query_variations,
        "route": decision.route,
        "thought_log": state.get("thought_log", []) + [log_entry],
        "retry_count": 0,
    }


# ── 4b. Retrieval ─────────────────────────────────────────────

def node_retrieval(state: AgentState) -> dict:
    """
    Node 2 — Retrieval
    ──────────────────
    Executes retrieval across the selected tools using all 3 query
    variations. Results are deduplicated by page-content hash.
    """
    logger.info("NODE │ retrieval (retry=%d)", state.get("retry_count", 0))
    route = state["route"]
    queries = state["query_variations"]
    context_chunks: list[str] = []
    source_refs: list[str] = []

    # ── Vector Store retrieval ──
    if route in ("vector", "both"):
        retriever = build_pinecone_retriever()
        seen: set[str] = set()
        for q in queries:
            docs = retriever.invoke(q)
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    context_chunks.append(doc.page_content)
                    meta = doc.metadata
                    source_refs.append(
                        f"{meta.get('source', 'unknown')} "
                        f"[{meta.get('company', '')} {meta.get('quarter', '')}]"
                    )

    # ── Web / Tavily retrieval ──
    if route in ("web", "both"):
        search_tool = build_web_search_tool()
        for q in queries[:1]:  # limit to primary query for web search
            results = search_tool.invoke(q)
            for r in results:
                snippet = r.get("content", "")
                url = r.get("url", "web")
                if snippet and snippet not in context_chunks:
                    context_chunks.append(snippet)
                    source_refs.append(url)

    log_entry = (
        f"[Retrieval] Retrieved {len(context_chunks)} unique chunks "
        f"via route='{route}'"
    )
    logger.info(log_entry)

    return {
        "retrieved_context": context_chunks,
        "source_references": source_refs,
        "thought_log": state.get("thought_log", []) + [log_entry],
    }


# ── 4c. Grade Gate (Self-Correction) ─────────────────────────

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict relevance grader for a financial RAG system.
Given the user question and the retrieved context passages, assess whether
the context is SUFFICIENT to produce an accurate, grounded answer.

Grade as "relevant" only if:
  • The context contains specific data points that directly address the query.
  • There is no clear factual gap that would force the LLM to hallucinate.

Grade as "irrelevant" if the context is off-topic, too generic, or missing
the specific metrics needed to answer the question.

Respond ONLY with valid JSON:
{format_instructions}""",
        ),
        (
            "human",
            "Question: {question}\n\nContext:\n{context}",
        ),
    ]
)


def node_grade_gate(state: AgentState) -> dict:
    """
    Node 3 — Grade Gate
    ───────────────────
    LLM-based grader. Populates 'grade' key in state; the conditional
    edge function reads this to route to Generation or Rewriter.
    """
    logger.info("NODE │ grade_gate")
    llm = build_llm()
    parser = PydanticOutputParser(pydantic_object=GradeDecision)

    context_str = "\n\n---\n\n".join(state["retrieved_context"][:6])

    chain = GRADE_PROMPT | llm | parser
    decision: GradeDecision = chain.invoke(
        {
            "question": state["original_query"],
            "context": context_str,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    log_entry = (
        f"[GradeGate] grade='{decision.grade}' | "
        f"reason='{decision.reason}' | retry={state.get('retry_count', 0)}"
    )
    logger.info(log_entry)

    return {
        "grade": decision.grade,
        "thought_log": state.get("thought_log", []) + [log_entry],
    }


def edge_grade_router(state: AgentState) -> str:
    """
    Conditional edge after Grade Gate.
    ────────────────────────────────────
    Returns the name of the next node:
      • "generate"  → context is good, proceed
      • "rewrite"   → context is bad and retries remain
      • "generate"  → context is bad but max retries (2) exhausted;
                       generate a best-effort / low-confidence answer
    """
    grade = state.get("grade", "irrelevant")
    retries = state.get("retry_count", 0)
    max_retries = 2

    if grade == "relevant":
        logger.info("EDGE │ grade_router → generate")
        return "generate"
    if retries < max_retries:
        logger.info("EDGE │ grade_router → rewrite (retry %d)", retries + 1)
        return "rewrite"
    # Max retries exhausted — generate with low confidence
    logger.warning("EDGE │ grade_router → generate (max retries exhausted)")
    return "generate"


# ── 4d. Query Rewriter ───────────────────────────────────────

REWRITE_PROMPT = PromptTemplate.from_template(
    """The previous retrieval attempt for the question below returned
irrelevant context. Rewrite the query to be MORE SPECIFIC and targeted
so that a vector similarity search will return better results.

Original question: {question}
Previous query variations used: {previous_variations}

Return ONLY the improved query as a plain string — no JSON, no quotes."""
)


def node_query_rewriter(state: AgentState) -> dict:
    """
    Query Rewriter Node
    ───────────────────
    Produces a new primary query, regenerates 3 variations via the router
    prompt, and increments the retry counter. The graph then loops back
    to node_retrieval.
    """
    logger.info("NODE │ query_rewriter")
    llm = build_llm(temperature=0.3)  # slight creativity for rewriting

    rewritten_q: str = (REWRITE_PROMPT | llm).invoke(
        {
            "question": state["original_query"],
            "previous_variations": state["query_variations"],
        }
    ).content.strip()

    # Re-run the router logic inline to produce 3 new variations
    router_parser = PydanticOutputParser(pydantic_object=RouteDecision)
    router_chain = ROUTER_PROMPT | llm | router_parser
    new_decision: RouteDecision = router_chain.invoke(
        {
            "query": rewritten_q,
            "format_instructions": router_parser.get_format_instructions(),
        }
    )

    new_retry = state.get("retry_count", 0) + 1
    log_entry = (
        f"[Rewriter] retry={new_retry} | rewritten='{rewritten_q}' | "
        f"new variations={new_decision.query_variations}"
    )
    logger.info(log_entry)

    return {
        "original_query": rewritten_q,          # update primary query
        "query_variations": new_decision.query_variations,
        "route": new_decision.route,
        "retry_count": new_retry,
        "thought_log": state.get("thought_log", []) + [log_entry],
    }


# ── 4e. Generation & Structured Validation ───────────────────

GENERATION_SYSTEM = """You are an elite financial analyst AI assistant.
Using ONLY the context provided below, answer the user's question with
precision and cite specific figures where available.

IMPORTANT FORMATTING RULES:
- You MUST respond with a single JSON object matching this exact schema:
{format_instructions}
- Do NOT wrap the JSON in markdown code fences.
- Do NOT include any text before or after the JSON.
- If the context does not fully support the answer, set confidence_score
  below 0.5 and acknowledge the gap in the 'answer' field."""

GENERATION_HUMAN = """Context:
{context}

Question: {question}"""


def node_generation(state: AgentState) -> dict:
    """
    Node 4 — Generation & Structured Validation
    ─────────────────────────────────────────────
    Generates the final answer with PydanticOutputParser enforcement.
    The parser will raise ValidationError if the LLM deviates from
    the FinancialAnswer schema — we handle this with a fallback.
    """
    logger.info("NODE │ generation")
    llm = build_llm()
    parser = PydanticOutputParser(pydantic_object=FinancialAnswer)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATION_SYSTEM),
            ("human", GENERATION_HUMAN),
        ]
    )

    context_str = "\n\n---\n\n".join(state["retrieved_context"][:8])
    sources = state.get("source_references", [])

    chain = prompt | llm | parser

    try:
        answer: FinancialAnswer = chain.invoke(
            {
                "context": context_str,
                "question": state["original_query"],
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception as exc:
        # Graceful degradation: return a structured error response
        logger.error("Generation validation failed: %s", exc)
        answer = FinancialAnswer(
            answer=(
                "I was unable to generate a fully structured answer. "
                "Please try rephrasing your question."
            ),
            source_documents_used=sources[:3],
            confidence_score=0.1,
        )

    log_entry = (
        f"[Generation] confidence={answer.confidence_score:.2f} | "
        f"sources={len(answer.source_documents_used)}"
    )
    logger.info(log_entry)

    # Append assistant reply to message history
    assistant_msg = AIMessage(content=answer.answer)

    return {
        "final_answer": answer,
        "messages": [assistant_msg],
        "thought_log": state.get("thought_log", []) + [log_entry],
    }


# ─────────────────────────────────────────────────────────────
# 5.  GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and compile the LangGraph StateGraph.

    Node registry
    ─────────────
    multi_query_router  →  Node 1
    retrieval           →  Node 2
    grade_gate          →  Node 3
    rewrite             →  Query Rewriter (loops back to retrieval)
    generate            →  Node 4

    Conditional edges
    ─────────────────
    grade_gate  ──[relevant]──►  generate
                ──[irrelevant, retries < 2]──►  rewrite
                ──[irrelevant, retries >= 2]──►  generate (forced)
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("multi_query_router", node_multi_query_router)
    graph.add_node("retrieval", node_retrieval)
    graph.add_node("grade_gate", node_grade_gate)
    graph.add_node("rewrite", node_query_rewriter)
    graph.add_node("generate", node_generation)

    # Entry point
    graph.add_edge(START, "multi_query_router")

    # Linear edges
    graph.add_edge("multi_query_router", "retrieval")
    graph.add_edge("retrieval", "grade_gate")

    # Conditional branch at Grade Gate
    graph.add_conditional_edges(
        "grade_gate",
        edge_grade_router,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        },
    )

    # Rewriter loops back to retrieval
    graph.add_edge("rewrite", "retrieval")

    # Generation is the terminal node
    graph.add_edge("generate", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# 6.  PUBLIC HELPER: run a single query end-to-end
# ─────────────────────────────────────────────────────────────

def run_query(query: str) -> AgentState:
    """
    Convenience wrapper — builds and invokes the full graph.

    Returns the final AgentState so callers can access:
      state["final_answer"]   → FinancialAnswer pydantic object
      state["thought_log"]    → list of internal reasoning steps
    """
    app = build_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "query_variations": [],
        "route": "vector",
        "retrieved_context": [],
        "source_references": [],
        "retry_count": 0,
        "final_answer": None,
        "thought_log": [],
    }
    return app.invoke(initial_state)
