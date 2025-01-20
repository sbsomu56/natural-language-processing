import os
from crewai_tools import YoutubeChannelSearchTool
os.environ['OPENAI_API_KEY'] = 'sk-dsdsdds'
os.environ['OPENAI_MODEL_NAME'] = 'llama3.2'
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'

yt_tool=YoutubeChannelSearchTool(
    youtube_channel_handle="@krishnaik06",
    )
