import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformer_wrapper import SentenceTransformerEmbeddings

load_dotenv()
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))

loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"],
    bs_kwargs={"parse_only": bs4_strainer},
    requests_kwargs={"timeout": 30}
)

docs = loader.load()

print(f"total characters: {len(docs[0].page_content)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

all_splits = text_splitter.split_documents(docs)
print(f"total splits: {len(all_splits)}")

embedding_function = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=embedding_function,
    persist_directory="./chroma_db"
)

ids = vector_store.add_documents(all_splits)

print(f"Persisted {len(ids)} documents to disk.")
