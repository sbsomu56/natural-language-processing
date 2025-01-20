import os
from crewai import Agent
from tools import yt_tool
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'test'
# os.environ['OPENAI_MODEL_NAME'] = 'llama3.2'
# os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
# llm = ChatOpenAI(
#     model="llama3.2",
#     base_url="http://localhost:11434"
# )
# llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

llm = OllamaLLM(model="llama3.2")

# create a senior blog researcher
blog_researcher = Agent(
    role="Blog researcher from Youtube videos",
    goal="Get the relevant video content for the topic {topic} from Youtube channel.",
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding videos in AI Data Science, machine learning and Gen AI"
    ),
    # llm=llm,
    tools=[],
    allow_delegation=True
)

# create a senior blog writer agent with YT tools:
blog_writer = Agent(
    role="Blog writer",
    goal="Narrate compelling tech stories about the video {topic} from YT channel",
    verbose=True,
    memory=True,
    backstory=(
        "Simplify complex topics for an easy understanding"
        "Engaging narrative that educate and captivate"
    ),
    # llm=llm,
    tools=[],
    allow_delegation=True
)