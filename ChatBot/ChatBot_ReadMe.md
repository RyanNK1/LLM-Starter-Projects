## 🧩 Part 1: Basic Chat Model Invocation

### 🧠 Concept Overview

This part introduces the **simplest possible interaction** with a language model using LangChain and OpenRouter. The goal is to:

- Load the API key securely.
- Initialize a chat model.
- Send a message to the model.
- Print the response.

This forms the foundation for all future chatbot functionality.

---

### 📚 Key Terms Explained

- **LangChain**: A framework for building applications powered by language models. It abstracts away model calls, prompt formatting, memory, and more.
- **OpenRouter**: A gateway that lets one access multiple LLMs (like Mistral, Claude, etc.) using a unified API.
- **ChatOpenAI**: A LangChain class that wraps chat models compatible with OpenAI’s API format.
- **Environment Variables**: Secure way to store sensitive data like API keys outside your code.
- **HumanMessage**: A LangChain message class representing user input in a chat.

---

### 📄 Code Breakdown (Line-by-Line)

```python
import os
from dotenv import load_dotenv
```
- `os`: Python’s built-in module for interacting with the operating system.
- `load_dotenv`: Loads environment variables from a `.env` file into the Python environment.

---

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  
```
- `HumanMessage`: Represents a message from the user in a chat-style interaction.
- `ChatOpenAI`: LangChain’s wrapper for chat models that follow OpenAI’s API format. You’ll use this to send and receive messages.

---

```python
# Load environment variables from .env file
load_dotenv()
```
- This reads the `.env` file and makes variables like `OPENROUTER_API_KEY` available via `os.getenv`.

---

```python
# Get the OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
```
- Retrieves the API key from the environment.
- If the key is missing, it raises an error to prevent unauthorized or failed API calls.

---

```python
# Initialize the model using ChatOpenAI with OpenRouter settings
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  
)
```
- Initializes the chat model.
- `base_url`: Points to OpenRouter’s API.
- `api_key`: Authenticates the requests.
- `model`: Specifies which model to use—in this case, Mistral 7B Instruct.

---

```python
# Send a simple message to the model
response = model.invoke([HumanMessage(content="Hi! I'm Ryan")])
```
- Sends a message to the model.
- `invoke`: Executes the model call.
- `[HumanMessage(...)]`: Wraps the input in a format the model understands.

---

```python
# Print the model's response
print(response.content)
```
- Displays the model’s reply in the terminal.

---

### ✅ Outcome

By the end of Part 1:
- Successfully connected to OpenRouter.
- Sent a message to a chat model.
- Printed the model’s response.

This confirms the environment is working and sets the stage for building a full chatbot.

---

Awesome! Let’s document **Part 2: Adding Message History** of your ChatBot project.

---

## 🧩 Part 2: Adding Message History

### 🧠 Concept Overview

In this part, **message history**—a critical feature for building a conversational chatbot- is introduced. Instead of sending a single message to the model, I now send a **sequence of messages** that simulate a back-and-forth dialogue.

This allows the model to:
- **Maintain context** across turns.
- **Remember previous user inputs**.
- **Respond more naturally**, as if it were part of an ongoing conversation.

---

### 📚 Key Terms Explained

- **Message History**: A list of past interactions (user and assistant messages) that are sent to the model to provide context.
- **HumanMessage**: Represents a message from the user.
- **AIMessage**: Represents a message from the assistant (model).
- **Context Window**: The maximum number of tokens a model can consider at once. Older messages may be trimmed when this limit is reached.

---

### 🔄 Changes from Part 1

Feature -  Part 1 -  Part 2    
Message Format - Single message - List of messages (history) 
Model Invocation - One-shot - Context-aware
Response Behavior- Stateless - Conversational   

---

### 📄 Code Breakdown (Line-by-Line)

```python
import os
from dotenv import load_dotenv
```
- Same as Part 1: used to load environment variables securely.

---

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  
```
- `AIMessage` is newly imported to represent the assistant’s previous responses.
- This allows you to simulate a full conversation history.

---

```python
# Load environment variables from .env file
load_dotenv()
```
- Loads your `.env` file to can access the API key.

---

```python
# Get the OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
```
- Retrieves the API key and raises an error if it’s missing.

---

```python
# Initialize the model using ChatOpenAI with OpenRouter settings
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  
)
```
- Initializes the chat model using OpenRouter and specifies the Mistral 7B Instruct model.

---

```python
messages = [
    HumanMessage(content="Hi! I'm Ryan."),
    AIMessage(content="Hi Ryan! It's nice to meet you. How can I assist you today? Is there something specific you would like to talk about or ask for help with? I'm here to help with answers, guidance, and helpful information on a wide range of topics. Let me know how I can help, and we can get started!"),
    HumanMessage(content="What's my name?"),
]
```
- This is the **message history**.
- It includes:
  - A user greeting.
  - The assistant’s response (manually added for now).
  - A follow-up question from the user.
- This simulates a real conversation and gives the model context to answer the last question accurately.

---

```python
# Send a simple message to the model
response = model.invoke(messages)
```
- Sends the entire message history to the model.
- The model uses all previous messages to generate a context-aware response.

---

```python
# Print the model's response
print(response.content)
```
- Displays the assistant’s reply to the last user message.

---

### ✅ Outcome

By the end of Part 2, the chatbot:
- Maintains context across multiple turns.
- Can answer follow-up questions more intelligently.
- Is one step closer to a real conversational agent.

---

Absolutely! Here's the documentation for **Part 3: Adding Persistence with LangGraph**, written in the **first person** as requested.

---

## 🧩 Part 3: Adding Persistence with LangGraph

### 🧠 Concept Overview

In this part of the project, I wanted my chatbot to **remember previous messages** across multiple interactions. Instead of manually passing the entire message history every time, I used **LangGraph** to manage the flow and **MemorySaver** to persist the conversation state.

This means the chatbot can now maintain context automatically, just like a real conversation. I don’t need to keep track of the history myself—LangGraph does it for me using a **thread ID**.

---

### 📚 Key Terms Explained

- **LangGraph**: A framework that lets me build workflows as graphs. Each node represents a step in the logic, and edges define how data flows between them.
- **StateGraph**: A LangGraph class used to define the structure of the graph, including its nodes and transitions.
- **MessagesState**: A built-in schema that holds a list of messages (user and assistant).
- **MemorySaver**: A checkpointing tool that stores and retrieves conversation history using a thread ID.
- **Thread ID**: A unique identifier for a conversation. LangGraph uses this to know which messages belong to which session.
- **START**: A special node that marks the beginning of the graph.

---

### 📄 Code Breakdown (Line-by-Line)

```python
import os
from dotenv import load_dotenv
```
I imported modules to load environment variables. This helps me keep sensitive data like API keys out of the code.

---

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  
```
I imported `HumanMessage` to represent user input and `ChatOpenAI` to interact with the model via OpenRouter.

---

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
```
Here I brought in the core LangGraph components:
- `MemorySaver` to persist message history.
- `START` to mark the beginning of the graph.
- `MessagesState` to define the structure of the state.
- `StateGraph` to build the workflow.

---

```python
# Load environment variables from .env file
load_dotenv()
```
This loads my `.env` file so I can access the API key securely.

---

```python
# Get the OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
```
I retrieved the API key and added a check to make sure it exists. If it’s missing, the script stops with an error.

---

```python
# Initialize the model using ChatOpenAI with OpenRouter settings
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  
)
```
I initialized the chat model using OpenRouter and specified the Mistral 7B Instruct model.

---

```python
# Define LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)
```
I created a LangGraph workflow using `MessagesState`, which will hold the conversation messages.

---

```python
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}
```
This function is the core of the graph. It takes the current state (message history), sends it to the model, and returns the updated state with the model’s response.

---

```python
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
```
I added a node named `"model"` to the graph and connected it to the `START` node. This means the graph starts by calling the model.

---

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
I compiled the graph and attached a memory checkpoint using `MemorySaver`. This allows the chatbot to remember previous messages using a thread ID.

---

```python
# Define thread ID
config = {"configurable": {"thread_id": "chat-thread-001"}}
```
I defined a thread ID to uniquely identify the conversation. This is how memory persistence works—LangGraph uses this ID to store and retrieve message history.

---

```python
# First message
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
output = app.invoke({"messages": input_messages}, config)
print(output["messages"][-1].content)
```
I sent the first message to the chatbot. The response is printed, and the message is saved in memory under the thread ID.

---

```python
# Follow-up message
followup = [HumanMessage(content="What's my name?")]
output = app.invoke({"messages": followup}, config)
print(output["messages"][-1].content)
```
I sent a follow-up message. Thanks to memory persistence, the model remembers the previous message and responds accordingly.

---

### ✅ Outcome

By the end of Part 3, I had:
- A LangGraph-powered chatbot.
- Automatic memory persistence using thread IDs.
- A scalable foundation for multi-turn conversations.

---

## 🧩 Part 4: Prompt Templates

### 🧠 Concept Overview

In this part of the project, I introduced **prompt templates** to make my chatbot more structured and flexible. Instead of manually building the prompt every time I send a message to the model, I used LangChain’s `ChatPromptTemplate` to define a reusable format.

This template includes:
- A **system message** that sets the assistant’s behavior.
- A **placeholder** for the conversation history.

This approach helps me:
- Maintain a consistent tone and personality for the assistant.
- Dynamically inject message history into the prompt.
- Separate prompt logic from model invocation logic.

---

### 📚 Key Terms Explained

- **Prompt Template**: A reusable structure for generating prompts. It combines static instructions with dynamic content.
- **System Message**: A special message that sets the assistant’s role or behavior (e.g., “You are a helpful assistant.”).
- **MessagesPlaceholder**: A LangChain utility that dynamically inserts a list of messages into the prompt.
- **invoke()**: A method that executes the prompt and returns the model’s response.

---

### 📄 Code Breakdown (Line-by-Line)

```python
import os
from dotenv import load_dotenv
```
I imported modules to load environment variables securely. This keeps sensitive data like API keys out of the code.

---

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  
```
I imported `HumanMessage` to represent user input and `ChatOpenAI` to interact with the model via OpenRouter.

---

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
```
These are the LangGraph components I used to define the workflow and enable memory persistence:
- `MemorySaver` stores conversation history.
- `START` marks the beginning of the graph.
- `MessagesState` defines the structure of the state.
- `StateGraph` builds the workflow.

---

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
```
This is new in Part 4. I imported:
- `ChatPromptTemplate` to define the prompt structure.
- `MessagesPlaceholder` to dynamically insert message history.

---

```python
# Load environment variables from .env file
load_dotenv()
```
Loads my `.env` file so I can access the API key securely.

---

```python
# Get the OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
```
Retrieves the API key and raises an error if it’s missing.

---

```python
# Initialize the model using ChatOpenAI with OpenRouter settings
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",  
    api_key=api_key,
    model="mistralai/mistral-7b-instruct",  
)
```
Initializes the chat model using OpenRouter and specifies the Mistral 7B Instruct model.

---

```python
# Define a prompt template with a system message
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```
Here I created a prompt template:
- The system message sets the assistant’s behavior.
- The `MessagesPlaceholder` dynamically inserts the conversation history (`messages`) into the prompt.

---

```python
# Define LangGraph workflow
workflow = StateGraph(state_schema=MessagesState)
```
I defined the LangGraph workflow using `MessagesState` to manage message history.

---

```python
def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}
```
This function now uses the prompt template:
- `prompt_template.invoke(state)` generates the full prompt using the system message and message history.
- The model is then called with this structured prompt.

---

```python
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
```
I added the model node to the graph and connected it to the `START` node.

---

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
I compiled the graph and attached a memory checkpoint using `MemorySaver`.

---

```python
# Define thread ID
config = {"configurable": {"thread_id": "chat-thread-001"}}
```
I defined a thread ID to uniquely identify the conversation.

---

```python
# First message
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
output = app.invoke({"messages": input_messages}, config)
print(output["messages"][-1].content)
```
I sent the first message. The prompt template ensures the assistant responds in a helpful tone.

---

```python
# Follow-up message
followup = [HumanMessage(content="What's my name?")]
output = app.invoke({"messages": followup}, config)
print(output["messages"][-1].content)
```
I sent a follow-up message. Thanks to memory and the prompt template, the assistant responds with context and consistency.

---

### ✅ Outcome

By the end of Part 4, I had:
- A structured prompt system using templates.
- A consistent assistant persona.
- Dynamic message injection for flexible conversations.

---
Great! Let's continue with your documentation for **Part 5: Dynamic Inputs in Prompt Templates**. Here's a structured write-up that matches the style and clarity of your previous parts:

---

## 🧩 Part 5: Dynamic Inputs in Prompt Templates

### 🧠 Concept Overview

In this part, I extended the prompt template functionality by introducing **dynamic inputs**, specifically the ability to change the **language** of the assistant's responses. This makes the chatbot more versatile and user-friendly for multilingual interactions.

Instead of hardcoding the assistant’s behavior, I used a `{language}` placeholder in the system message. This allows me to control the assistant’s response language dynamically at runtime.

---

### 📚 Key Terms Explained

- **Dynamic Prompt Input**: A variable (like `{language}`) that can be replaced with user-defined values when the prompt is invoked.
- **TypedDict**: A Python typing construct used to define structured state objects.
- **StateGraph**: A LangGraph component that defines the flow of operations based on a state schema.
- **MessagesPlaceholder**: Inserts a list of messages into the prompt dynamically.
- **invoke()**: Executes the prompt with the current state and returns the model’s response.

---

### 📄 Code Breakdown (Line-by-Line)

```python
from typing import Sequence
from typing_extensions import Annotated, TypedDict
```
I imported typing utilities to define a structured state object that includes both messages and a language field.

---

```python
from langchain_core.messages import HumanMessage, BaseMessage
```
These classes represent user and assistant messages in the conversation.

---

```python
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
```
I defined a custom state schema with:
- `messages`: the conversation history.
- `language`: the target language for the assistant’s responses.

---

```python
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
```
This prompt template includes a dynamic `{language}` placeholder in the system message, allowing runtime control over the assistant’s response language.

---

```python
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}
```
This function:
- Builds the prompt using the current state (including language).
- Sends it to the model and returns the response.

---

```python
workflow = StateGraph(state_schema=State)
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
```
I created a LangGraph workflow using the new `State` schema and added the model node.

---

```python
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
```
I compiled the graph and attached a memory checkpoint to persist conversation history.

---

```python
config = {"configurable": {"thread_id": "chat-thread-001"}}
language = "Spanish"
```
I set the thread ID and chose Spanish as the response language.

---

```python
input_messages = [HumanMessage(content="Hi! I'm Ryan.")]
output = app.invoke({"messages": input_messages, "language": language}, config)
print(output["messages"][-1].content)
```
I sent the first message. The assistant responded in Spanish, thanks to the dynamic language input.

---

```python
followup = [HumanMessage(content="What's my name?")]
output = app.invoke({"messages": followup, "language": language}, config)
print(output["messages"][-1].content)
```
I sent a follow-up message. The assistant remembered my name and responded in Spanish, demonstrating both memory and dynamic prompt behavior.

---

### ✅ Outcome

By the end of Part 5, I had:
- A multilingual chatbot that responds in the user’s preferred language.
- A flexible prompt system with dynamic inputs.
- A structured state schema that supports both messages and metadata.

---

Perfect! Here's the documentation for **Part 6: Managing Conversation History** and **Part 7: Streaming Responses**, continuing in the same style as your previous entries:

---

## 🧩 Part 6: Managing Conversation History

### 🧠 Concept Overview

In this part, I focused on **trimming and managing conversation history** to ensure the chatbot remains efficient and responsive, especially when dealing with long conversations. Since language models have token limits, it's important to control how much of the message history is passed to the model.

I used LangChain’s `trim_messages` utility to selectively reduce the message history based on token count. This ensures that the most relevant parts of the conversation are retained while staying within model limits.

---

### 📚 Key Terms Explained

- **Token Limit**: The maximum number of tokens (words, punctuation, etc.) a model can process in a single prompt.
- **trim_messages**: A LangChain utility that trims message history based on token count and strategy.
- **Custom Token Counter**: A function that estimates token usage using a tokenizer from Hugging Face.

---

### 📄 Code Breakdown (Line-by-Line)

```python
from langchain_core.messages import trim_messages
```
I imported the `trim_messages` utility to manage message history.

---

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
```
I used a Hugging Face tokenizer to estimate token usage since the default token counter had issues with OpenRouter.

---

```python
def count_tokens(messages: Sequence[BaseMessage]) -> int:
    text = "\n".join([msg.content for msg in messages])
    return len(tokenizer.encode(text))
```
This function counts tokens by encoding the concatenated message contents.

---

```python
trimmer = trim_messages(
    max_tokens=100,
    strategy="last",
    token_counter=count_tokens,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
```
I configured the trimmer to:
- Keep the last messages.
- Include system messages.
- Start trimming from the last human message.
- Avoid partial messages.

---

```python
trimmed_messages = trimmer.invoke(state["messages"])
```
Before sending messages to the model, I trimmed them to stay within the token limit.

---

### ✅ Outcome

By the end of Part 6, I had:
- A reliable way to manage long conversations.
- A custom token counter for OpenRouter compatibility.
- A trimmed message history that preserves context without exceeding limits.

---

## 🧩 Part 7: Streaming Responses

### 🧠 Concept Overview

In this part, I implemented **streaming responses** to make the chatbot feel more interactive and real-time. Instead of waiting for the entire response to be generated, the assistant now streams its reply chunk-by-chunk as it's being generated.

This improves user experience by reducing perceived latency and mimicking natural conversation flow.

---

### 📚 Key Terms Explained

- **Streaming**: Receiving and displaying model output incrementally as it's generated.
- **stream()**: A method that returns an iterator over response chunks from the model.
- **flush=True**: Ensures immediate printing of each chunk without buffering.

---

### 📄 Code Breakdown (Line-by-Line)

```python
stream = model.stream(prompt)
collected_chunks = []
print("Assistant:", end=" ", flush=True)
for chunk in stream:
    print(chunk.content, end="", flush=True)
    collected_chunks.append(chunk)
print()
```
This block:
- Starts streaming the model response.
- Prints each chunk as it arrives.
- Collects the final chunk to store in memory.

---

```python
for step in app.stream({"messages": input_messages, "language": language}, config):
    pass  # Streaming handled in call_model
```
I used `app.stream()` to invoke the graph and handle streaming inside the `call_model` function.

---

### ✅ Outcome

By the end of Part 7, I had:
- A chatbot that streams responses in real-time.
- A smoother and more engaging user experience.
- A flexible architecture that supports both streaming and memory.

---
Here's the documentation for **Part 8: CLI Interface**, along with an explanation of why the original trimmer was commented out:

---

## 🧩 Part 8: CLI Interface

### 🧠 Concept Overview

In this part, I built a **Command-Line Interface (CLI)** for the chatbot, allowing users to interact with it directly from the terminal. This makes the chatbot more accessible and practical for testing or lightweight usage without needing a web interface.

The CLI supports:
- Real-time input from the user.
- Streaming responses from the assistant.
- Persistent conversation history across turns.
- Exit command to gracefully end the session.

---

### 📚 Key Terms Explained

- **CLI (Command-Line Interface)**: A text-based interface where users type commands and receive output directly in the terminal.
- **input()**: A built-in Python function that reads user input from the terminal.
- **Streaming**: The assistant responds incrementally, improving responsiveness.
- **Conversation Loop**: A `while` loop that continuously accepts user input until an exit condition is met.

---

### 📄 Code Breakdown (Line-by-Line)

```python
message_history = []
```
I initialized an empty list to store the conversation history.

---

```python
print("\n--- ChatBot (type 'exit' to quit) ---\n")
```
This prints a welcome message and instructions for exiting the chatbot.

---

```python
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Conversation ended.")
        break
```
This loop:
- Continuously prompts the user for input.
- Ends the session if the user types `"exit"` or `"quit"`.

---

```python
message_history.append(HumanMessage(content=user_input))
for step in app.stream({"messages": message_history, "language": language}, config):
    pass  # Streaming handled in call_model
```
Each user message is added to the history and passed to the LangGraph app. The assistant's response is streamed and printed in real-time.

---

### 🧩 Why the Original Trimmer Was Commented Out

```python
# trimmer = trim_messages(
#     max_tokens=100,
#     strategy="last",
#     token_counter=model,
#     include_system=True,
#     allow_partial=False,
#     start_on="human",
# )
```

The original trimmer was commented out because it relied on the model itself (`token_counter=model`) to count tokens. This caused issues with OpenRouter, which doesn't expose a compatible token counting method for LangChain's internal utilities.

Instead, I implemented a **custom token counter** using Hugging Face's `AutoTokenizer`:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
```

This approach ensures accurate token counting and avoids compatibility issues with OpenRouter.

---

### ✅ Outcome

By the end of Part 8, I had:
- A fully functional CLI chatbot.
- Real-time streaming responses.
- Persistent conversation history.
- A user-friendly interface for testing and interaction.

---

