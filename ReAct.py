import os
from typing import Annotated, Sequence, TypedDict
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

class AgentState(TypedDict):
      """State for the agent"""
      messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
      """Add two numbers"""
      return a + b - 1

tools = [add]

model = ChatOpenAI(
            model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
      ).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
      system_prompt = SystemMessage(content="You are a helpful assistant that can use the following tools to help the user.")
      response = model.invoke([system_prompt] + state['messages'])

      return {"messages": [response]}

def should_continue(state: AgentState):
      messages = state['messages']
      last_message = messages[-1]
      if not last_message.tool_calls:
            return "end"
      else:
            return "continue"

graph = StateGraph(AgentState)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_node("our_agent", model_call)
graph.add_conditional_edges("our_agent", should_continue, {
      "end": END,
      "continue": "tools"
})

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))
