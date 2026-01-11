from langchain import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessge

load_dotenv()

persistent_directory = 'db/chroma_db'

#Loading embeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persistent_directory=persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space":"cosine"}
)

query = "which island does spacex lease for its launches in the pacific"

# This method creates and returns a NEW object of type VectorStoreRetriever

retriever = db.as_retriever(search_kwargs = {"k":3}) 

relevant_docs = retriever.invoke(query)

# Combining the query and the relevant documnet contents

combined_input = f"""Based on the following documents, please answer this question {query}

    Documents:
    {chr(10).join([f" - {doc.page_content}" for doc in relevant_docs])}

    Please, provide a clear helpful answer using only the information using these documents. If you can't find
    answer in the documents, say "I don't have enough information to answer the question based on the provided documnets"

    """

model = ChatOpenAI(model = "gpt-4o")

messages = [
    HumanMessage(content = combined_input)
]

result = model.invoke(messages)

print(result.content)