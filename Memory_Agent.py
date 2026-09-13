from ctypes import Union
import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

class AgentState(TypedDict):
      """State for the agent"""
      messages: List[Union[HumanMessage, AIMessage]]

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
      state['messages'].append(AIMessage(content=response.content))
      print(response.content)
      print({"messageUpdated": state['messages']})
      return state

graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()
conversation_history = []

user_input = input("Enter your message: ")
while user_input != "exit":
      conversation_history.append(HumanMessage(content=user_input))
      print({"conversationHistoryBEFORE": conversation_history})
      response = app.invoke({"messages": conversation_history})
      conversation_history = response['messages']
      print({"conversationHistoryAFTER": conversation_history})
      user_input = input("Enter your message: ")

with open("logging.txt", "w") as f:
      f.write("Your conversation history: \n")
      for message in conversation_history:
            if isinstance(message, HumanMessage):
                  f.write(f"User: {message.content}\n")
            elif isinstance(message, AIMessage):
                  f.write(f"Agent: {message.content}\n")
            f.write("\n")
      f.write("End of conversation\n")
print("Exiting...")