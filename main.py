import boto3
import streamlit as st
from langchain.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import PyPDFLoader # Changed to PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os 

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def get_documents():
    # Define the PDF file path
    pdf_file_path = "icd_10_codes.pdf"
    
    # --- Debugging additions (adjusted for single file) ---
    st.info(f"Attempting to load PDF from: {os.path.abspath(pdf_file_path)}")
    if not os.path.exists(pdf_file_path):
        st.error(f"Error: PDF file '{pdf_file_path}' not found at {os.path.abspath('.')}. Please ensure it exists.")
        return []
    else:
        st.info(f"Found PDF file: {pdf_file_path}")
    # --- End debugging additions ---

    loader = PyPDFLoader(pdf_file_path) # Changed to PyPDFLoader for a single file
    
    documents = []
    try:
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}. Ensure the PDF is not corrupted and has read permissions.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        st.error("No text content could be extracted from the PDF file. It might be empty or unreadable.")
        return []
    return docs

def get_vector_store(docs):
    if not docs:
        st.warning("Cannot create vector store: No documents to process.")
        return

    vector_store_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vector_store_faiss.save_local("faiss_index")
    st.success("Vector Store Created/Updated Successfully!")

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>
Question: {question}
Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_llm():
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "maxTokenCount": 500
        }
    )
    return llm

def get_response_llm(llm, vector_store_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("RAG Demo")
    st.header("End to end RAG Application")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    
    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Create/Update Vector Store"):
            with st.spinner("Processing documents and creating vector store..."):
                docs = get_documents()
                if docs:
                    get_vector_store(docs)

        if st.button("Get Answer"):
            if user_question:
                with st.spinner("Getting response..."):
                    try:
                        import os
                        if not os.path.exists("faiss_index"):
                            st.error("Vector store not found. Please click 'Create/Update Vector Store' first.")
                            return

                        faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                        llm = get_llm()
                        response_text = get_response_llm(llm, faiss_index, user_question)
                        st.write(response_text)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        st.info("Please ensure the vector store is created and the 'interview_questions.pdf' file is in the same directory as your script.")
            else:
                st.warning("Please enter a question to get an answer.")

if __name__ == "__main__":
    main()
