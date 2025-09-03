from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from sentence_transformer_wrapper import SentenceTransformerEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
model = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')

vector_store = Chroma(
    collection_name="prompt_engineering",
    embedding_function=model,
    persist_directory="./chroma_db"
)

prompt = ChatPromptTemplate.from_template(
    '''You are a helpful assistant that can answer question abaut the blog post on prompt engineering.
    Use the following pieces of retrieved context to answer the question. If you don't know answer, just say "I don't know"
    Question: {question}
    Context: {context}
    Answer:''')

llm_model = os.getenv("MODEL")
base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")

llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key)

question = "What is Ferrari?"

retrieved_docs = vector_store.similarity_search(question, k=3)
docs_content = "\n".join([doc.page_content for doc in retrieved_docs])

message = prompt.invoke({"question": question, "context": docs_content})

answer = llm.invoke(message)

print(answer.content)