import boto3
import streamlit as st
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os 
import shutil 
import hashlib 

# Define the directory where uploaded PDFs will be stored
UPLOAD_DIR = "uploaded_pdfs"
FAISS_INDEX_DIR = "faiss_index"

# Define the maximum allowed file size in bytes (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 # 10 MB in bytes

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

def get_documents(pdf_file_path):
    st.info(f"Attempting to load PDF from: {os.path.abspath(pdf_file_path)}")
    if not os.path.exists(pdf_file_path):
        st.error(f"Error: PDF file '{pdf_file_path}' not found. Please ensure it exists.")
        return []
    else:
        st.info(f"Found PDF file: {os.path.basename(pdf_file_path)}")

    loader = PyPDFLoader(pdf_file_path)
    
    documents = []
    try:
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}. Ensure the PDF is not corrupted and has read permissions.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=700)
    
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        st.error("No text content could be extracted from the PDF file. It might be empty or unreadable.")
        return []
    return docs

def get_vector_store(docs, progress_bar=None):
    """
    Creates a FAISS vector store from the provided documents and embeddings,
    and saves it locally. If an existing index exists, it will be overwritten.
    Includes progress updates.
    """
    if not docs:
        st.warning("Cannot create vector store: No documents to process.")
        return None

    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)
        print(f"Removed existing FAISS index directory: {FAISS_INDEX_DIR}")

    # Initialize FAISS index with the first few documents
    # This prevents an error if docs is empty or very small
    if not docs:
        return None
        
    # Process documents in batches to update progress bar
    batch_size = 50 # Adjust batch size based on your document size and API limits
    total_docs = len(docs)
    
    # Create the initial FAISS index with the first batch
    vector_store_faiss = FAISS.from_documents(
        docs[:batch_size],
        bedrock_embeddings
    )
    if progress_bar:
        progress_bar.progress(min(10, int((batch_size / total_docs) * 100)), text="Building knowledge base...")

    # Add remaining documents in batches
    for i in range(batch_size, total_docs, batch_size):
        vector_store_faiss.add_documents(docs[i:i + batch_size])
        if progress_bar:
            progress = min(100, int(((i + batch_size) / total_docs) * 100))
            progress_bar.progress(progress, text=f"Building knowledge base... {progress}% complete")

    vector_store_faiss.save_local(FAISS_INDEX_DIR)
    print("Vector Store Created/Updated Successfully!")
    if progress_bar:
        progress_bar.progress(100, text="Knowledge base built!")
    st.success("Knowledge Base updated successfully! You can now ask questions.")
    return vector_store_faiss # Return the created vector store

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
    """
    Instantiates and returns the Anthropic Claude 3 Sonnet LLM from Bedrock.
    Uses BedrockChat for Claude models.
    """
    llm = BedrockChat(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        client=bedrock,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 500
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
    st.set_page_config("RAG Chatbot Demo", layout="centered")
    st.title("📄 RAG Chatbot with Bedrock")
    st.markdown("Upload a PDF to build the knowledge base, then ask questions about its content.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False
    if "last_processed_file_id" not in st.session_state:
        st.session_state.last_processed_file_id = None
    if "faiss_index_obj" not in st.session_state:
        st.session_state.faiss_index_obj = None
    
    def process_uploaded_file(uploaded_file_obj):
        if uploaded_file_obj.size > MAX_FILE_SIZE_BYTES:
            st.error(f"File size exceeds the limit of {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB. Please upload a smaller file.")
            st.session_state.vector_store_initialized = False
            st.session_state.last_processed_file_id = None
            st.session_state.faiss_index_obj = None
            return

        file_content = uploaded_file_obj.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        uploaded_file_obj.seek(0) 
        
        file_id = f"{uploaded_file_obj.name}-{uploaded_file_obj.size}-{file_hash}" 
        
        if st.session_state.last_processed_file_id != file_id:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file_obj.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file_obj.getbuffer())
            st.success(f"File '{uploaded_file_obj.name}' uploaded successfully!")

            # --- Progress bar for document processing ---
            progress_text = "Processing document... Please wait."
            my_bar = st.progress(0, text=progress_text)
            
            with st.spinner("Preparing documents..."):
                docs = get_documents(file_path)
            
            if docs:
                # Pass the progress bar to get_vector_store
                faiss_index = get_vector_store(docs, progress_bar=my_bar) 
                if faiss_index:
                    st.session_state.faiss_index_obj = faiss_index
                    st.session_state.vector_store_initialized = True
                    st.session_state.last_processed_file_id = file_id
                    st.session_state.messages = []
                    my_bar.empty() # Clear the progress bar after completion
                else:
                    st.error("Failed to create vector store. Please try another file.")
                    st.session_state.vector_store_initialized = False
                    st.session_state.last_processed_file_id = None
                    st.session_state.faiss_index_obj = None
                    my_bar.empty()
            else:
                st.error("Failed to process the uploaded PDF. Please try another file.")
                st.session_state.vector_store_initialized = False
                st.session_state.last_processed_file_id = None
                st.session_state.faiss_index_obj = None
                my_bar.empty()
        else:
            st.info("This file has already been processed. Knowledge base is ready.")
            st.session_state.vector_store_initialized = True

    # --- Check for existing FAISS index on initial load (only if not already initialized) ---
    if not st.session_state.vector_store_initialized and os.path.exists(FAISS_INDEX_DIR):
        with st.spinner("Loading existing knowledge base..."):
            try:
                st.session_state.faiss_index_obj = FAISS.load_local(FAISS_INDEX_DIR, bedrock_embeddings, allow_dangerous_deserialization=True)
                st.session_state.vector_store_initialized = True
                st.info("Existing knowledge base found. You can start asking questions.")
                if st.session_state.last_processed_file_id is None:
                    st.session_state.last_processed_file_id = "existing_index_loaded" 
            except Exception as e:
                st.error(f"Error loading existing FAISS index: {e}. Please try re-uploading a PDF.")
                st.session_state.vector_store_initialized = False
                st.session_state.last_processed_file_id = None
                st.session_state.faiss_index_obj = None


    with st.sidebar:
        st.title("Upload Document")
        with st.form("pdf_upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
            st.info(f"File size limit: {MAX_FILE_SIZE_BYTES / (1024 * 1024):.0f} MB")
            submit_button = st.form_submit_button("Process Document")

            if submit_button:
                if uploaded_file is not None:
                    process_uploaded_file(uploaded_file)
                else:
                    st.warning("Please upload a PDF file before clicking 'Process Document'.")
        
        if not st.session_state.vector_store_initialized:
            st.info("Upload a PDF and click 'Process Document' to get started.")
        elif st.session_state.vector_store_initialized and st.session_state.last_processed_file_id:
            display_name = st.session_state.last_processed_file_id.split('-')[0] if st.session_state.last_processed_file_id != "existing_index_loaded" else "from local storage"
            st.info(f"Knowledge base loaded. You can ask questions! (Current KB: `{display_name}`)")


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Type your question here...", 
        disabled=not st.session_state.vector_store_initialized
    )

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Getting response..."):
            try:
                if not st.session_state.vector_store_initialized or st.session_state.faiss_index_obj is None:
                    st.error("Knowledge base not initialized or loaded. Please upload a PDF first.")
                    st.session_state.messages.append({"role": "assistant", "content": "Knowledge base not ready. Please upload a PDF first."})
                    st.rerun()
                    return

                faiss_index = st.session_state.faiss_index_obj
                llm = get_llm()
                response_text = get_response_llm(llm, faiss_index, user_question)
                
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.info("Please ensure your AWS credentials are configured and the uploaded PDF is valid.")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}. Please check the console for details or try again."})
                st.rerun()

if __name__ == "__main__":
    main()
