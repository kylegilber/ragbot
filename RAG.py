
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tkinter import filedialog

'''
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import ollama
import time
'''

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    ""
]


def main():

    # prompt user to select a knowledge base
    file = filedialog.askopenfilename(
        title= "Select a file to use as the knowledge base",
        filetypes= [("Text Files", "*.pdf;")])

    # load knowledge base
    loader = PyMuPDFLoader(file_path= file)
    documents = loader.load()

    # load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code= True)
    
    # reduce sequence length
    model.max_seq_length = 8192

    # chunk documents
    chunks = split(512, documents)


def split(size, documents):
    """
    Split documents into smaller chunks

    :param size: integer specifying chunk size
    :param documents: list of documents
    """

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
        chunk_size= size,
        chunk_overlap= int(size / 10),
        add_start_index= True,
        strip_whitespace= True,
        separators= SEPARATORS
    )

    chunks = splitter.split_documents(documents= documents)
    return chunks

'''


model = "sentence-transformers/all-mpnet-base-v2"
hf = HuggingFaceEmbeddings(model_name= model)

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
collection = "bulletin"

vector_store = PGVector(
    embeddings= hf,
    collection_name= collection,
    connection= connection,
    use_jsonb= True,
)
'''
'''
# loads a PDF file as Documents and splits the docs into smaller chunks
def load(file):
    loader = PyMuPDFLoader(file_path= file)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 450, chunk_overlap = 0)
    chunks = splitter.split_documents(documents= docs)
    return chunks
    
# embeds the data and stores the resulting vectors locally
def embed(chunks):
    # establish access to a embedding model through Ollama
    embeddings = OllamaEmbeddings(model= "mxbai-embed-large")   # 7th ranked MTEB model

    # specify a storage folder for embeddings
    store = LocalFileStore("Embeddings")

    # initialize a cache-backed embedder
    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings= embeddings,  # the embedder to use for embedding
        document_embedding_cache= store,    # ByteStore for caching embeddings
        namespace= embeddings.model         # sets namespace to embedding model name
    )
    
    # embed chunks and store resulting embedding vectors in a vector store
    vectorstore = FAISS.from_documents(documents= chunks, embedding= embedder)
    vectorstore.save_local("Indexes")    # saves vectorstore in local folder
    return vectorstore

def rag(file, query):
    # load existing or create new vector store as retriever
    try:
        retriever = FAISS.load_local(
            folder_path= "Indexes",
            embeddings= OllamaEmbeddings(model= "mxbai-embed-large"),
            allow_dangerous_deserialization= True)
    except:
        chunks = load(file) 
        retriever = embed(chunks)

    # embed the user's query
    vector = OllamaEmbeddings(model= "mxbai-embed-large").embed_query(query)

    # similarity search
    docs = retriever.similarity_search_with_score_by_vector(vector, k= 2,)

    # maximal marginal relevance (MMR) search
    #docs = retriever.max_marginal_relevance_search_by_vector(vector, k= 2, fetch_k= 10)

    # give the search's top two results as context
    if docs[0][1] < 250:    # verify the results are similar by checking their score
        context = docs[0][0].page_content + docs[1][0].page_content
    else:
        return f"The query {query} appears unrelated to the provided file." 
    
    # define set of instructions for model behavior
    SYSTEM_PROMPT = """You are a helpful assistant who answers questions based on snippets 
        of text provided in context. Keep your answers grounded in the context and be as concise
        as possible."""
    
    # format and combine user query with file context
    prompt = f"Question: {query}\n\nContext: {context}"
    
    # provide model specifications and generate response
    response = ollama.chat(
        model= "llama3",    # context size: 8K
        messages= [
            {"role" : "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content" : prompt}
        ]   
    )
    return response["message"]["content"]

'''
main()