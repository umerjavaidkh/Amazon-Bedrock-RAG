import boto3
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Bedrock # Correct import for text generation models

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1" # Ensure this region is where Bedrock models are enabled for your account
)

# Set the model ID for Amazon Titan Text Express v1
model_id = "amazon.titan-text-express-v1"

# Instantiate the Bedrock LLM for text generation
llm = Bedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={
        "temperature": 0.5,
        "maxTokenCount": 500 # <--- CHANGE THIS: Use maxTokenCount instead of max_tokens_to_sample
    }
)

# Chatbot function
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=['language', 'user_text'],
        template="You are a helpful assistant. You are speaking in {language}.\n\n{user_text}"
    )
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'user_text': user_text})
    return response

# Streamlit UI
st.title("Amazon Bedrock LLM Demo with Titan Text Express")

language = st.sidebar.selectbox("Language", ["english", "hindi", "spanish"])
user_text = st.sidebar.text_area(label="What is your question?", max_chars=100)

if user_text:
    response = my_chatbot(language, user_text)
    st.write(response["text"])
