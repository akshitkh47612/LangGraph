import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

class AgentState(TypedDict):
      """State for the agent"""
      messages: List[HumanMessage]

llm = ChatOpenAI(
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
      )

def process(state: AgentState) -> AgentState:
      """Process the agent"""
      messages = state['messages']
      response = llm.invoke(messages)
      print(response.content)
      return state

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

user_input = input("Enter your message: ")
while user_input != "exit":
      app.invoke({"messages": [HumanMessage(content=user_input)]})
      user_input = input("Enter your message: ")
print("Exiting...")