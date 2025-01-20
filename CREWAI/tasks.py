import os
from crewai import Task
from tools import yt_tool
from agents import blog_researcher,blog_writer
os.environ['OPENAI_API_KEY'] = 'sk-dsdsdds'
os.environ['OPENAI_MODEL_NAME'] = 'llama3.2'
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'

# research task
research_task = Task(
    description=(
        "Identify the video {topic}."
        "Get detailed information about the video from the channel."
    ),
    expected_output="A comprehensive 3 paragraphs long report based on the {topic} of the video",
    tools=[yt_tool],
    agent=blog_researcher
)

# write task
write_task = Task(
    description=(
        "get the information from the youtube channel on the topic {topic}."
    ),
    expected_output="Summarize the information from youtube channel video on the topic {topic} and create a content for the blog.",
    tools=[yt_tool],
    agent=blog_writer,
    async_execution=False,
    output_file="new-blog-post.md"
)