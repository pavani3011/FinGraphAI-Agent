import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def clean_page_text(text:str)->str:
    lines = text.split('\n')
    cleaned = [line for line in lines if len(line.strip())>20]
    return '\n'.join(cleaned)

def explore(pdf_path:str):
    path = Path(pdf_path).resolve()
    print(f"File: {pdf_path}")

    if not path.is_file():
        print(f"❌ Error: The file does not exist at {path}")
        sys.exit(1)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    for page in pages:
        page.page_content = clean_page_text(page.page_content)

    print(f"Total pages: {len(pages)}")
    print(f"Total characters: {sum(len(p.page_content) for p in pages):,}")
    print(f"Avg chars per page: {sum(len(p.page_content) for p in pages) // len(pages):,}")

    print(f"\n---Page1---")
    print(pages[0].page_content[:800])

    print(f"\n---Page2---")
    if len(pages)>1:
        print(pages[1].page_content[:800])

    # look for key financial markers
    full_text = " ".join(p.page_content for p in pages)
    markers = ["revenue","earnings per share", "EPS", "net income",
               "operating income", "guidance", "fiscal","quarter"]
    print(f"\n--- Keyword presence ---")
    for marker in markers:
        count = full_text.lower().count(marker.lower())
        status = "✅" if count> 0 else "❌"
        print(f" {status} '{marker}': {count} occurrences")

    # check for table noise 
    tab_chars = full_text.count('\t')
    pipe_chars = full_text.count('|')
    dollar_chars = full_text.count('$')
    print(f"Tab chars: {tab_chars}")
    print(f"pipe chars: {pipe_chars}")
    print(f"dollar chars: {dollar_chars}")

    # check for page header/footer noise
    for i,page in enumerate(pages[:5]):
        preview = page.page_content[:100].replace('\n',' ')
        print(f"Page {i+1}: {preview}")

    #quick estimate
    total_chars = sum(len(p.page_content) for p in pages)
    chunk_Size = 1000
    overlap = 200
    effective_chunk = chunk_Size-overlap
    estimated_chunk = total_chars // effective_chunk
    print(f"Estimated chunks: ~{estimated_chunk}")

if __name__ == "__main__":
    default_path = r"C:\Users\vivaa\Desktop\FinGraphAI_Agent\data\apple\a411a029-368f-4479-b416-25c404acca3d.pdf"
    pdf = sys.argv[1] if len(sys.argv) > 1 else default_path
    explore(pdf)


    