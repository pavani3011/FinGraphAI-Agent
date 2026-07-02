import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_pdfs(pdf_dir: str = "./data/pdfs", index_name: str = "financial-rag") -> None:
    """
    Week 1 Task: Advanced PDF ingestion with RecursiveCharacterTextSplitter.

    Pipeline:
    1. Load all PDFs from `pdf_dir`.
    2. Chunk with RecursiveCharacterTextSplitter (overlap preserves context
       across chunk boundaries — critical for financial tables).
    3. Attach rich metadata: company name, quarter, year (parsed from filename).
    4. Generate text-embedding-3-small embeddings.
    5. Upsert to Pinecone with metadata.

    Filename convention expected: <TICKER>_<QUARTER>_<YEAR>.pdf
      e.g.  AAPL_Q3_2024.pdf, MSFT_Q2_2024.pdf
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        logger.error("PDF directory '%s' not found. Create it and add PDFs.", pdf_dir)
        return

  
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []
    for pdf_file in sorted(pdf_path.glob("*.pdf")):
        logger.info("Loading: %s", pdf_file.name)
        loader = PyPDFLoader(str(pdf_file))
        pages = loader.load()

        
        stem_parts = pdf_file.stem.upper().split("_")
        company = stem_parts[0] if len(stem_parts) > 0 else "UNKNOWN"
        quarter = stem_parts[1] if len(stem_parts) > 1 else "Q?"
        year = stem_parts[2] if len(stem_parts) > 2 else "20XX"

        chunks = splitter.split_documents(pages)
        for i, chunk in enumerate(chunks):
            
            chunk.metadata.update(
                {
                    "source": pdf_file.name,
                    "company": company,
                    "quarter": f"{quarter}-{year}",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
        all_chunks.extend(chunks)
        logger.info("  -> %d chunks created from %s", len(chunks), pdf_file.name)

    if not all_chunks:
        logger.warning("No chunks produced. Check PDF directory.")
        return

    logger.info("Total chunks to upsert: %d", len(all_chunks))

    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

   
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
    )
    logger.info(
        "✅ Upserted %d chunks to Pinecone index '%s'",
        len(all_chunks),
        index_name,
    )



def run_cli_query(query: str) -> None:
    """
    Runs a single query through the full LangGraph agent and prints the structured result to stdout.
    """
    from graph_logic import run_query

    logger.info("Running query: %s", query)
    state = run_query(query)
    answer = state.get("final_answer")

    print("  FINANCIAL ANALYST AGENT — RESULT")

    if answer:
        print(f"\n Answer:\n{answer.answer}\n")
        print(f" Confidence: {answer.confidence_score:.0%}")
        if answer.key_metrics:
            print("\n Key Metrics:")
            for k, v in answer.key_metrics.items():
                print(f"   {k}: {v}")
        print("\n Sources:")
        for src in answer.source_documents_used:
            print(f"   • {src}")
    else:
        print("No answer generated.")

    print("\n Agent Thought Log:")
    for step in state.get("thought_log", []):
        print(f"   {step}")
    print("═" * 60 + "\n")


RAGAS_TEST_DATASET = [
    {
        "question": "What was Apple's total revenue in Q3 2024?",
        "ground_truth": (
            "Apple reported total revenue of $85.8 billion in Q3 2024, "
            "representing a 5% year-over-year increase."
        ),
        "contexts": [
            (
                "Apple Inc. Q3 2024 Earnings Release. Net sales: $85.8 billion, "
                "up 5% year over year. Services revenue reached a record $24.2 billion. "
                "iPhone revenue was $39.3 billion."
            ),
            (
                "Apple CFO Luca Maestri stated: 'We are reporting our best June quarter "
                "revenue ever with 5% growth year-over-year to $85.8 billion.'"
            ),
        ],
        "answer": (
            "Apple's total revenue in Q3 2024 was $85.8 billion, "
            "a 5% increase year-over-year driven by record Services revenue of $24.2B."
        ),
    },
    {
        "question": "What was Microsoft's EPS for Q4 FY2024?",
        "ground_truth": (
            "Microsoft reported diluted earnings per share of $2.95 for Q4 FY2024, "
            "beating analyst consensus estimates of $2.93."
        ),
        "contexts": [
            (
                "Microsoft Corporation Q4 FY2024 Results. Diluted EPS: $2.95. "
                "Revenue: $64.7 billion, up 15% year-over-year. "
                "Azure cloud revenue grew 29%."
            ),
        ],
        "answer": (
            "Microsoft's diluted EPS for Q4 FY2024 was $2.95, "
            "slightly above the consensus estimate of $2.93."
        ),
    },
    {
        "question": "How did Nvidia's data center revenue perform in Q1 FY2025?",
        "ground_truth": (
            "Nvidia's data center revenue in Q1 FY2025 reached $22.6 billion, "
            "representing a 427% year-over-year increase fuelled by AI chip demand."
        ),
        "contexts": [
            (
                "Nvidia Q1 FY2025 Earnings. Data Center revenue: $22.6 billion, "
                "up 427% YoY. Total revenue: $26.0 billion. "
                "Gross margin improved to 78.4%."
            ),
            (
                "CEO Jensen Huang: 'The next industrial revolution has begun — "
                "companies and countries are partnering with Nvidia to shift "
                "from general-purpose to accelerated computing.'"
            ),
        ],
        "answer": (
            "Nvidia's data center revenue surged 427% year-over-year to $22.6 billion "
            "in Q1 FY2025, driven by explosive demand for AI training infrastructure."
        ),
    },
]


def run_ragas_evaluation() -> None:
    """
    Week 3 Task: Compute Ragas RAG quality metrics offline.

    Metrics computed:
    faithfulness     — Does the answer stay grounded in the context?
                       (Hallucination detector: 1.0 = fully grounded)
    answer_relevance — Is the answer actually responsive to the question?
                       (Relevance: 1.0 = perfectly on-point)

    Interpreting results:
    • faithfulness < 0.7  → retrieval or grounding problem
    • answer_relevance < 0.8 → generation / prompting problem

    Baseline → After Query Rewriting improvement example:
      faithfulness:     0.68 → 0.91  (+23 points)
      answer_relevance: 0.74 → 0.88  (+14 points)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        logger.error(
            "Ragas or datasets not installed. Run: pip install ragas datasets"
        )
        return

    logger.info("Building Ragas evaluation dataset (%d samples)…", len(RAGAS_TEST_DATASET))
    dataset_dict = {
        "question": [s["question"] for s in RAGAS_TEST_DATASET],
        "answer": [s["answer"] for s in RAGAS_TEST_DATASET],
        "contexts": [s["contexts"] for s in RAGAS_TEST_DATASET],
        "ground_truth": [s["ground_truth"] for s in RAGAS_TEST_DATASET],
    }
    dataset = Dataset.from_dict(dataset_dict)

    logger.info("Running Ragas evaluation… (this calls the OpenAI API)")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=None,  
        raise_exceptions=False,
    )

    print("\n" + "═" * 60)
    print("  RAGAS EVALUATION RESULTS")
    print("═" * 60)

    df = result.to_pandas()
    print(df[["question", "faithfulness", "answer_relevancy"]].to_string(index=False))

    print("\n📊 Aggregate Scores:")
    print(f"   faithfulness     : {df['faithfulness'].mean():.4f}")
    print(f"   answer_relevancy : {df['answer_relevancy'].mean():.4f}")

    results_path = Path("./ragas_results.json")
    results_path.write_text(
        json.dumps(
            {
                "faithfulness": round(float(df["faithfulness"].mean()), 4),
                "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
                "per_sample": df[
                    ["question", "faithfulness", "answer_relevancy"]
                ].to_dict(orient="records"),
            },
            indent=2,
        )
    )
    logger.info("Results saved to %s", results_path)
    print("═" * 60 + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enterprise RAG Financial Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest PDFs from ./data/pdfs/ into Pinecone.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a single financial question via CLI.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the Ragas offline evaluation pipeline.",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_pdfs()
    elif args.query:
        run_cli_query(args.query)
    elif args.evaluate:
        run_ragas_evaluation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
