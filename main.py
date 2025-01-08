import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

if __name__ == "__main__":
    #load file and split data
    file_path = "/home/prajodh/local-vector-db/2210.03629v3.pdf"
    loader  = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 30, separator = "\n")
    splitDocuments  = text_splitter.split_documents(documents)
    
    #LOADF EMBEDDINGS MODEL AND SAVE IN LOCAL VECTOR STORE
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splitDocuments, embeddings)
    vectorstore.save_local("local_pdf_index")

    new_vector_store = FAISS.load_local("local_pdf_index", embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    chain_docs = create_stuff_documents_chain(llm, retrieval_prompt)
    retrival_chain = create_retrieval_chain(new_vector_store.as_retriever(), chain_docs)
    res = retrival_chain.invoke(input = {"input":"give the gist of REACT in 3 lines"})
    print(res["answer"])