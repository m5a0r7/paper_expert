from llama_index.core import SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


loader = SimpleDirectoryReader(
    input_dir="./pdf_files", required_exts=[".pdf"], recursive=True
)
docs = loader.load_data()
embed_model = HuggingFaceEmbedding(
    model_name="Snowflake/snowflake-arctic-embed-m", trust_remote_code=True
)


# ====== Create vector store and upload indexed data ======
Settings.embed_model = embed_model  # we specify the embedding model to be used
index = VectorStoreIndex.from_documents(docs)


# setting up the llm
llm = Ollama(model="deepseek-r1:7b", request_timeout=60.0)

# ====== Setup a query engine on the index previously created ======
Settings.llm = llm  # specifying the llm to be used
query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
