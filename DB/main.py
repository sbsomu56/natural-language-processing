from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
import pandasql as ps
import pandas as pd
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize LLM
llm = OllamaLLM(model="llama3.2")

# Check if file exists
file_path = 'DB/transaction.csv'
if not os.path.exists(file_path):
    logging.error(f"File not found: {file_path}")
    raise FileNotFoundError(f"File not found: {file_path}")

# Load DataFrame
try:
    transaction_df = pd.read_csv(file_path)
    logging.info("CSV file loaded successfully.")
except Exception as e:
    logging.error(f"Error reading CSV file: {e}")
    raise

# Clean column names
transaction_df.columns = [col.replace(" ", "") for col in transaction_df.columns]
col_names = ','.join(transaction_df.columns)

# Refine prompt template
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
        ("user", "{text}")
    ]
)
parser = StrOutputParser()

# Create chain
chain = prompt | llm | parser

# Generate query
user_request = "Frequency distribution of the number of transactions by account number"
query = chain.invoke({
    "col_names": col_names,
    "text": user_request
})

# Log the generated query
logging.info(f"Generated SQL query: {query}")

# Execute the query
try:
    df = transaction_df
    result = ps.sqldf(query, locals())
    logging.info("Query executed successfully.")
    print(result)
except Exception as e:
    logging.error(f"Error executing SQL query: {e}")
    raise
