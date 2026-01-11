import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma 
from dotenv import load_dotenv

load_dotenv()

def load_files(docs_path = "docs"):

    print("Loading Files.....")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists.")
    
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        loader_kwargs={'encoding': 'utf-8'}  
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No files in the directory {docs_path}")
    
    print("File Loaded Successfully!")
    
    return documents

def split_documents(documents, chunk_size=800, chunk_overlap = 0):
    # here chunksize 800 is tokens
    print('Splitting Documents into chunks!!!')

    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks, persists_directory = 'db/chroma_db'):

    print("Creating Embeddings and storing it in the Chroma DB!!!!")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("---- Creating Vector Store ------")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persists_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("---- Finished Creating Vector Store")

    return vectorstore

def main():

    # Loading the files
    documents = load_files()
    
    # Chunking
    chunks = split_documents(documents)

    # Embedding & Loading it in the vector db
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()  