from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

pdf_path = r"data\NVIDIA-Q1-26.pdf"
pages = PyPDFLoader(pdf_path).load()

print(f"Pages loaded: {len(pages)}")
for i, page in enumerate(pages):
    print(f"Page {i+1}: {len(page.page_content):,} chars")

total = sum(len(p.page_content) for p in pages)
print(f"\nTotal chars: {total:,}")
print(f"Average per page: {total // len(pages):,}")


