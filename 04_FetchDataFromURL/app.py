import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_ollama.llms import OllamaLLM

# Streamlit app configuration
st.set_page_config(page_title="LangChain: Summarize Text From Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text website")
st.subheader('Summarize URL')

# Sidebar for choosing LLM and providing Groq API key
with st.sidebar:
    llm_choice = st.selectbox("Choose LLM Model", ["Ollama", "Groq API"])
    
    groq_api_key = None
    if llm_choice == "Groq API":
        groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# Set the LLM model based on the choice
if llm_choice == "Groq API":
    if not groq_api_key:
        st.error("Please provide your Groq API Key to use Groq.")
    else:
        llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
elif llm_choice == "Ollama":
    llm = OllamaLLM(model="llama3.2")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Button to trigger summarization
if st.button("Summarize the Content from YT or Website"):
    if not generic_url.strip():
        st.error("Please provide the URL to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading content..."):
                # Load content from the URL
                if "youtube111.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

            with st.spinner("Generating summary..."):
                # Run the summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run({"input_documents": docs})

            st.success("Summary generated successfully!")
            st.write(output_summary)

        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            st.write(f"Debug Info: {e}")
