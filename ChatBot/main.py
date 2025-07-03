import os
from dotenv import load_dotenv

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()

# Get the OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

# Initialize the model using ChatOpenAI with OpenRouter settings
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  
)


# Define a new state schema with messages and language
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Define a prompt template with a system message
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define LangGraph workflow
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Define thread ID
config = {"configurable": {"thread_id": "chat-thread-001"}}
language = "Spanish"

# First message
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
output = app.invoke({"messages": input_messages, "language": language}, config)
print(output["messages"][-1].content)

# Follow-up message
followup = [HumanMessage(content="What's my name?")]
output = app.invoke({"messages": followup, "language": language}, config)
print(output["messages"][-1].content)





