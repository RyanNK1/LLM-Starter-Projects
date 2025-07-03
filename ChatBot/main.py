import os
from dotenv import load_dotenv

from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, trim_messages
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

# --- Original trimmer (commented out due to token counting issue) ---
# trimmer = trim_messages(
#     max_tokens=100,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )

# --- Custom token counter using transformers ---
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def count_tokens(messages: Sequence[BaseMessage]) -> int:
    text = "\n".join([msg.content for msg in messages])
    return len(tokenizer.encode(text))

# --- New trimmer using custom token counter ---
trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define LangGraph workflow
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages, "language": state["language"]})
    
    # Stream response from model
    stream = model.stream(prompt)
    collected_chunks = []
    print("Assistant:", end=" ", flush=True)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
        collected_chunks.append(chunk)
    print()  # Newline after streaming

    return {"messages": [collected_chunks[-1]]}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Define thread ID
config = {"configurable": {"thread_id": "chat-thread-001"}}
language = "English"

# First message
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
for step in app.stream({"messages": input_messages, "language": language}, config):
    pass  # Streaming handled in call_model

# Follow-up message
followup = [HumanMessage(content="What's my name?")]
for step in app.stream({"messages": followup, "language": language}, config):
    pass  # Streaming handled in call_model






