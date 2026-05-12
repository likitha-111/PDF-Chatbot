from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import chromadb
from groq import Groq
import uuid
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_store"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_rag"
)

def process_pdf(pdf_path):

    global collection

    try:
        chroma_client.delete_collection("pdf_collection")
    except:
        pass

    collection = chroma_client.create_collection(
        name="pdf_collection"
    )

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in chunks]

    embeddings = embedding_model.encode(texts).tolist()

    ids = [str(uuid.uuid4()) for _ in texts]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings
    )

    print("PDF processed successfully")


def retrieve_chunks(query, k=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results["documents"][0]


def ask_question(query):

    chunks = retrieve_chunks(query)

    context = "\n\n".join(chunks)
    print("Context for question:", context)

    prompt = f"""
    You are a precise PDF question-answering assistant.

    <RULES>
    1. Answer ONLY using the provided context.
    2. Do NOT add external knowledge.
    3. Avoid unnecessary introductions or conclusions
    4. Be concise and direct.
    5. Do not repeat information.
    6. If answer is missing, reply EXACTLY: Answer not found in the document.
    </RULES>


    <CONTEXT>
    {context}
    </CONTEXT>

    <QUESTION>
    {query}
    </QUESTION>

    <OUTPUT_FORMAT>
    - Output MUST be valid markdown
    - Use markdown tables for comparisons only when necessary
    - Use bullet points for lists
    - Do NOT use titles unless necessary
    - Do NOT generate sections like Overview, Summary, Conclusion
    - Do NOT wrap output in quotes
    - Output ONLY the answer
    </OUTPUT_FORMAT>
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1
    )

    return response.choices[0].message.content