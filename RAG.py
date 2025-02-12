from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from transformers import AutoTokenizer
from tkinter import filedialog

EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
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

def split(size, documents):
    """
    Split documents into smaller chunks

    :param size: integer specifying chunk size
    :param documents: list of documents
    :returns: list of chunks
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


def embed(chunks, model):
    """
    Embed chunks and locally store the resulting vectors

    :param chunks: list of document data
    :param model: embedding model
    :returns: a vector store 
    """

    store = LocalFileStore("embeddings")

    embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings= model,
        document_embedding_cache= store,
        namespace= model.model_name
    )

    vectorstore = FAISS.from_documents(documents= chunks, embedding= embedder)
    vectorstore.save_local("indexes")
    return vectorstore


def load(model):
    """
    Load a locally-saved vector store

    :param model: embedding model
    :returns: a vector store
    """

    vectorstore = FAISS.load_local(
        folder_path= "indexes",
        embeddings= model,
        allow_dangerous_deserialization= True
    )

    return vectorstore


def main():

    # prompt user to select a knowledge base
    file = filedialog.askopenfilename(
        title= "Select a file to use as the knowledge base",
        filetypes= [("Text Files", "*.pdf;")])

    # load knowledge base
    loader = PyMuPDFLoader(file_path= file)
    documents = loader.load()

    # chunk documents
    chunks = split(512, documents)

    # load embedding model
    hf = HuggingFaceEmbeddings(
        model_name= EMBEDDING_MODEL,
        model_kwargs= {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": False},
        multi_process= True
    )

    # get vector store
    try: store = load(hf)
    except: store = embed(chunks, hf)


'''

def rag(file, query):
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
if __name__ == '__main__':
    main()