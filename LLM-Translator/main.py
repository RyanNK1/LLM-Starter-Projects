#w/oLangsmith Integration Tracing and Debugging and Monitoring
'''
import os
from dotenv import load_dotenv
from openai import OpenAI
from httpx import URL


# Load environment variables
load_dotenv()

# Set OpenRouter endpoint and API key
OpenAI.api_key = os.getenv("OPENROUTER_API_KEY")
#OpenAI.base_url = "https://openrouter.ai/api/v1"

# Prompt template
PROMPT_TEMPLATE = """
Translate the following text from {input_language} into {target_language}:
{text}
"""

# Step 1: Create prompt
def create_prompt(input_language, target_language, text):
    return PROMPT_TEMPLATE.format(input_language=input_language,target_language=target_language, text=text)

# Step 2: Translate
def translate_text(input_language, target_language, text):
    prompt = create_prompt(input_language, target_language, text)

    client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=URL("https://openrouter.ai/api/v1")
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# Main execution
if __name__ == "__main__":
    print("Welcome to the English Language Translator")
    target_language = input('Enter target language:').strip()
    input_language = input('Enter the input language:').strip()
    text = input(f"Enter {input_language} text to translate: ").strip()

    #Error Handling

    try:
        translation = translate_text(input_language, target_language, text)
        print("\nTranslation:\n", translation)
    except Exception as e:
        print(f"\n Translation Failed: {e}")

'''
import os
from dotenv import load_dotenv
#from langchain.chat_models import ChatOpenAI
#from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Prompt template
PROMPT_TEMPLATE = """
Translate the following text from {input_language} into {target_language}:
{text}
"""

# Step 1: Create prompt
def create_prompt(input_language, target_language, text):
    return PROMPT_TEMPLATE.format(
        input_language=input_language,
        target_language=target_language,
        text=text
    )

# Step 2: Translate using LangChain + LangSmith
def translate_text(input_language, target_language, text):
    prompt = create_prompt(input_language, target_language, text)

    chat = ChatOpenAI(
        model="mistralai/mistral-7b-instruct",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY")
    )

    messages = [
        SystemMessage(content="You are a helpful translator."),
        HumanMessage(content=prompt)
    ]

    response = chat.invoke(messages)
    return response.content

# Main execution
if __name__ == "__main__":
    print("Welcome to the English Language Translator")
    target_language = input('Enter target language:').strip()
    input_language = input('Enter the input language:').strip()
    text = input(f"Enter {input_language} text to translate: ").strip()

    try:
        translation = translate_text(input_language, target_language, text)
        print("\nTranslation:\n", translation)
    except Exception as e:
        print(f"\nTranslation Failed: {e}")
