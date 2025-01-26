from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from pathlib import Path
from tempfile import mkdtemp
import os
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.vector_stores.milvus import MilvusVectorStore

load_dotenv(override=True)
EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
MILVUS_URI = str(Path(mkdtemp()) / "docling.db")
GEN_MODEL = HuggingFaceInferenceAPI(
    token=os.getenv("HF_TOKEN"),
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
)
SOURCE = "put/your/file/directory"  # Docling Technical Report
QUERY = "Which is this paper about?"

embed_dim = len(EMBED_MODEL.get_text_embedding("hi"))


reader = DoclingReader()
node_parser = MarkdownNodeParser()
directory = Path(SOURCE)

# Get all files in the directory (non-recursive)
files = [file for file in directory.iterdir() if file.is_file()]

vector_store = MilvusVectorStore(
    uri=str(Path(mkdtemp()) / "docling.db"),  # or set as needed
    dim=embed_dim,
    overwrite=True,
)
index = VectorStoreIndex.from_documents(
    documents=reader.load_data(files),
    transformations=[node_parser],
    storage_context=StorageContext.from_defaults(vector_store=vector_store),
    embed_model=EMBED_MODEL,
)
result = index.as_query_engine(llm=GEN_MODEL).query(QUERY)
print(f"Q: {QUERY}\nA: {result.response.strip()}\n\nSources:")
print([(n.text, n.metadata) for n in result.source_nodes])
