import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
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

# Define a prompt template with a system message
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Define thread ID
config = {"configurable": {"thread_id": "chat-thread-001"}}

# First message
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
output = app.invoke({"messages": input_messages}, config)
print(output["messages"][-1].content)

# Follow-up message
followup = [HumanMessage(content="What's my name?")]
output = app.invoke({"messages": followup}, config)
print(output["messages"][-1].content)





