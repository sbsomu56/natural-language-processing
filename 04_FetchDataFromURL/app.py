import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain_ollama.llms import OllamaLLM
import nltk
# nltk.download('averaged_perceptron_tagger_eng')

## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')



## Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
# llm =ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
llm = OllamaLLM(model="llama3.2")

prompt_template="""
Provide a summary of the following content in 300 words:
Content: {text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Loading content..."):
                # Load content from the URL
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                docs = loader.load()

                # Debug the loaded documents
                # st.write("Debug: Loaded Docs", docs)

            with st.spinner("Generating summary..."):
                # Run the summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run({"input_documents": docs})

            st.success("Summary generated successfully!")
            st.write(output_summary)

        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            st.write(f"Debug Info: {e}")
