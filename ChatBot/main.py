import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  

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
messages = [
    HumanMessage(content="Hi! I'm Ryan."),
    AIMessage(content=" Hi Ryan! It's nice to meet you. How can I assist you today? Is there something specific you would like to talk about or ask for help with? I'm here to help with answers, guidance, and helpful information on a wide range of topics. Let me know how I can help, and we can get started!"),
    #Yes, this is the mistral response after the previous commit lol
    HumanMessage(content="What's my name?"),
]
# Send a simple message to the model
response = model.invoke(messages)

# Print the model's response
print(response.content)




