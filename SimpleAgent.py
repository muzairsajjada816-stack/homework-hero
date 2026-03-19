from crewai import Agent, Task, Crew , LLM
import os
from dotenv import load_dotenv
load_dotenv()

llm = LLM(
    api_key=os.getenv("GROQ_API_KEY"),
)

story_agent = Agent(
    role="spy",
    goal = "tell a friendship and adventure type story for kids.",
    backstory = "a creative story for kids about friendship and adventure.",
    llm=llm,
)
story_task = Task(
    description="Create a 10 line story which should be fun and engaging story for kids about friendship.",
    expected_output="A short story for kids showing friendship adventure between two kids Names: Uzair , Bilal.",
    agent=story_agent,
)
crew = Crew(
    agents=[story_agent],
    tasks=[story_task],
)

result = crew.kickoff()
print(result)