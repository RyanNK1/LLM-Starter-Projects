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
Translate the following text from English into {language}:
{text}
"""

# Step 1: Create prompt
def create_prompt(language, text):
    return PROMPT_TEMPLATE.format(language=language, text=text)

# Step 2: Translate
def translate_text(language, text):
    prompt = create_prompt(language, text)

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
    language = input('Enter target language:').strip()
    text = input("Enter English text to translate: ").strip()

    #Error Handling

    try:
        translation = translate_text(language, text)
        print("\nTranslation:\n", translation)
    except Exception as e:
        print(f"\n Translation Failed: {e}")