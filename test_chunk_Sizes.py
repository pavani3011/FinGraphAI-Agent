from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pages = PyPDFLoader(r"data\NVIDIA-Q1-26.pdf").load()

configs = [
    {"chunk_size":500, "chunk_overlap":100},
    {"chunk_size":1000, "chunk_overlap":200},
    {"chunk_size":1500, "chunk_overlap":300},
]

for config in configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = config["chunk_size"],
        chunk_overlap= config["chunk_overlap"],
        separators=["\n\n","\n",". ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    avg_len= sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"\nchunk_size = {config['chunk_size']}, overlap={config['chunk_overlap']}")
    print(f" total chunks: {len(chunks)}")
    print(f" avg chunk len: {avg_len} chars")
    print(f" sample chunk:")
    aws_chunks = [c for c in chunks if "AWS" in c.page_content]
    if aws_chunks:
        print(f" ---")
        print(f" {aws_chunks[0].page_content[:400]}")
        print(f" ---")