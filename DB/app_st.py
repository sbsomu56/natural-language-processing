import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import pandasql as ps
import os
from langchain_groq import ChatGroq
import re
from ollama import Client

groq_api_key = st.secrets["groq"]["api_key"]

try:
    client = Client(
    host='http://localhost:11434',
    headers={'x-some-header': 'some-value'})
    model_list = client.list()
    OLLAMA_MODEL_LIST = [model['model'] for model in model_list['models']]
except:
    OLLAMA_MODEL_LIST = ['llama3.2:latest']

def remove_special_characters(input_string):
    # Use regex to remove non-alphanumeric characters and spaces
    result = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    return result

# List of Groq models (from the screenshot)
GROQ_MODELS = [
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "deepseek-r1-distill-llama-70b",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3-70b-8192",
    "llama-3-8b-8192",
]

# Streamlit App Title
st.title("SQL Query Generator and Executor")
st.write("Upload a CSV file, describe your query in plain English, and get the results!")

# Sidebar for LLM and Model Selection
st.sidebar.title("LLM Configuration")

# LLM Provider Dropdown
llm_provider = st.sidebar.selectbox("Select LLM Provider", ["Groq","Ollama"])

if llm_provider == "Groq":
    # Add a note for users about the Groq API key
    st.sidebar.info(
        """
        **Note**: To use Groq models, you need to provide your Groq API key.
        You can obtain your API key from the Groq platform dashboard. 
        Visit [Groq Platform](https://groq.com) to log in and generate your API key.
        """
    )

    # Input API Key
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password",value=groq_api_key)
    # Model Dropdown
    selected_model = st.sidebar.selectbox("Select a Groq Model", GROQ_MODELS)
else:
    st.sidebar.info(
        """
        **Note**: If you are running this in cloud then this opion will not be accessible.
        """
    )
    # selected_model = st.sidebar.text_input("Enter Ollama Model Name", value="llama3.2")
    selected_model = st.sidebar.selectbox("Enter Ollama Model Name", OLLAMA_MODEL_LIST)

# File Uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Load DataFrame
        transaction_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("### Original File Preview:")
        st.write(transaction_df.head())

        # Clean column names
        original_columns = transaction_df.columns.tolist()
        transaction_df.columns = [remove_special_characters(col.replace(" ", "_")) for col in transaction_df.columns]
        col_names = ','.join(transaction_df.columns)

        # Notify about column changes
        if original_columns != transaction_df.columns.tolist():
            st.warning(
                "Note: Column names have been modified to replace spaces with underscores for consistency. "
                "The updated column names are reflected below."
            )

        # Show updated data preview
        st.write("### Updated File Preview:")
        st.write(transaction_df.head())

        # Display data types
        st.write("### Column Data Types:")
        st.write(transaction_df.dtypes.astype(str).to_frame(name="Data Type").T)


        # Provide option to download updated dataset
        csv_data = transaction_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Updated CSV",
            data=csv_data,
            file_name="updated_transaction.csv",
            mime="text/csv",
        )

        # SQL Query Input
        user_request = st.text_input(
            "Describe your query (e.g., 'Frequency distribution of the number of transactions by account number'):",
            value="",
        )

        # Generate SQL Query and Execute
        if user_request:
            if llm_provider == "Groq" and not api_key:
                st.error("Please enter a valid Groq API key.")
            else:
                if llm_provider == "Ollama":
                    llm = OllamaLLM(model=selected_model)
                else:
                    # Replace this placeholder with the actual Groq LLM initialization
                    llm = ChatGroq(model=selected_model,api_key=api_key)  # Replace with `GroqLLM(model=selected_model, api_key=api_key)` if implemented

                from langchain_core.prompts import ChatPromptTemplate

                generic_template = """
                Role: You are a SQL expert.

                Context: A SQL table has the following columns: {col_names}.
                Output: Write a valid SQL query based on the user's request. Ensure the query is compatible with the given columns.
                Assume the table name is 'df'. Do not include extra formatting or comments.

                User Request: {text}
                """

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", generic_template),
                        ("user", "{text}"),
                    ]
                )
                parser = StrOutputParser()
                chain = prompt | llm | parser

                # Generate SQL Query
                if "generated_query" not in st.session_state:
                    st.session_state.generated_query = None

                if st.button("Generate Query"):
                    try:
                        query = chain.invoke({
                            "col_names": col_names,
                            "text": user_request,
                        })
                        st.session_state.generated_query = query
                        st.success("SQL query generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating SQL query: {e}")
                # Editable Query Text Area
                if st.session_state.generated_query:
                    st.write("### Generated SQL Query:")
                    edited_query = st.text_area(
                        "Edit the generated SQL query if needed:",
                        value=st.session_state.generated_query,
                        key="edited_query",
                        height=150,
                    )

                    # Execute Edited Query
                    if st.button("Execute Query"):
                        try:
                            result = ps.sqldf(edited_query, {"df": transaction_df})
                            st.write("### Query Results:")
                            st.dataframe(result)
                        except Exception as e:
                            st.error(f"Error executing SQL query: {e}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
