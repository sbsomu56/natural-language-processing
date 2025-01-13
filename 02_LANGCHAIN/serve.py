from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

# API_KEY:
GROQ_API_KEY="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"

model=ChatGroq(model="Gemma2-9b-It",groq_api_key=GROQ_API_KEY)
generic_template = "Translate the following into {language}"
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),("user","{text}")
    ]
)
parser = StrOutputParser()

chain=prompt|model|parser
# chain.invoke({
#     "language":"french",
#     "text":"Hello"
# })

## App definition
app=FastAPI(title="Langchain server",version="1.0",description="A simple LCEL example using langchain and groq api")

## app definition:
add_routes(
    app,
    chain,
    path="/chain"
    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="localhost",port=8000)