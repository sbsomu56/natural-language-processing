import os
from crewai import Crew,Process
from agents import blog_researcher,blog_writer
from tasks import research_task,write_task
os.environ['OPENAI_API_KEY'] = 'sk-dsdsdds'
os.environ['OPENAI_MODEL_NAME'] = 'llama3.2'
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'

crew = Crew(
    agents=[blog_researcher,blog_writer],
    tasks=[research_task,write_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

result=crew.kickoff(
    inputs={"topic":"AI vs MS vs Data Science"}
)
print(result)