## Overview

This is a simple Python-based translator application that uses **OpenRouter** to access LLMs (Large Language Models) for translating text between languages. It uses the `mistralai/mistral-7b-instruct` model via OpenRouter's API.

---

## 📂 File: `main.py`

### 🔧 Dependencies

```python
import os
from dotenv import load_dotenv
from openai import OpenAI
from httpx import URL
```

- `os`: Used to access environment variables.
- `dotenv.load_dotenv()`: Loads environment variables from a `.env` file.
- `OpenAI`: The OpenAI client, compatible with OpenRouter's API.
- `httpx.URL`: Used to define the base URL for the OpenRouter API.

---

### 🔐 Environment Setup

```python
load_dotenv()
OpenAI.api_key = os.getenv("OPENROUTER_API_KEY")
```

- Loads the `.env` file to securely access your API key.
- Retrieves the `OPENROUTER_API_KEY` from the environment and sets it for the OpenAI client.

---

### 🧠 Prompt Template

```python
PROMPT_TEMPLATE = """
Translate the following text from {input_language} into {target_language}:
{text}
"""
```

- A formatted string that serves as the prompt for the LLM.
- `{input_language}`, `{target_language}`, and `{text}` are placeholders to be filled dynamically.

---

### 🏗️ Step 1: Create Prompt

```python
def create_prompt(input_language, target_language, text):
    return PROMPT_TEMPLATE.format(input_language=input_language, target_language=target_language, text=text)
```

- This function fills in the prompt template with the user-provided languages and text.

---

### 🌐 Step 2: Translate Text

```python
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
```

- Creates a prompt using the earlier function.
- Initializes the OpenAI client with OpenRouter’s endpoint.
- Sends a chat completion request to the model with:
  - A **system message** to set the model’s behavior.
  - A **user message** containing the translation prompt.
- Returns the translated text from the model's response.

---

### 🚀 Main Execution Block

```python
if __name__ == "__main__":
    print("Welcome to the English Language Translator")
    target_language = input('Enter target language:').strip()
    input_language = input('Enter the input language:').strip()
    text = input(f"Enter {input_language} text to translate: ").strip()

    try:
        translation = translate_text(input_language, target_language, text)
        print("\nTranslation:\n", translation)
    except Exception as e:
        print(f"\n Translation Failed: {e}")
```

- Runs when the script is executed directly.
- Prompts the user for:
  - Target language
  - Input language
  - Text to translate
- Calls the `translate_text` function and prints the result.
- Includes error handling to catch and display any issues during translation.

---

## 📁 .env File

Make sure `.env` file contains:

```
OPENROUTER_API_KEY=api_key
```

This keeps your API key secure and out of the source code.

---

## ✅ Requirements

Install dependencies using:

```bash
pip install python-dotenv openai httpx
```

---
