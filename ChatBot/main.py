import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
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

# Send a simple message to the model
response = model.invoke([HumanMessage(content="Hi! I'm Ryan")])

# Print the model's response
print(response.content)




