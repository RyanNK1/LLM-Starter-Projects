import os
from dotenv import load_dotenv
from typing import Sequence, Dict, Any
from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI  
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
from langsmith import Client
from langchain.callbacks.manager import collect_runs

# Load environment variables
load_dotenv()

# --- Initialize LangSmith ---
client = Client()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph-ChatBot"

# --- Streamlit App Setup ---
st.title("🤖 LangGraph ChatBot")
st.markdown("A conversational AI with conversation history")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_count" not in st.session_state:
    st.session_state.conversation_count = 1
if "saved_conversations" not in st.session_state:
    st.session_state.saved_conversations = {}
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = "chat_1"

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model Settings expander
    with st.expander("⚙️ Model Settings", expanded=True):
        st.markdown("""
        **Temperature**  
        Controls randomness:  
        - Lower = more predictable  
        - Higher = more creative  
        """)
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=2.0, 
            value=0.7,
            key="temp_slider",
            label_visibility="collapsed"
        )
        
        st.markdown("""
        **Max Tokens**  
        Limits response length:  
        - Too low = cut-off answers  
        - Too high = verbose responses  
        """)
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2000,
            value=1000,
            key="tokens_slider",
            label_visibility="collapsed"
        )
    
    # Language and model selection
    language = st.selectbox(
        "Response Language",
        ["English", "Spanish", "French", "German"],
        key="lang_select"
    )
    
    model_name = st.selectbox(
        "Model",
        ["mistralai/mistral-7b-instruct", "anthropic/claude-2", "openai/gpt-3.5-turbo"],
        index=0,
        key="model_select"
    )
    
    # Context management
    st.header("🔍 Context Management")
    context_length = st.slider(
        "Conversation Memory",
        min_value=0,
        max_value=10,
        value=4,
        key="context_slider",
        help="Number of previous exchanges to remember (0 for no context)"
    )
    
    # Conversation controls
    st.header("📝 Conversation")
    
    # New Chat button
    if st.button("🔄 Start New Chat", key="new_chat"):
        # Save current conversation
        conv_data = {
            "messages": st.session_state.messages.copy(),
            "config": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "language": language,
                "model": model_name
            }
        }
        st.session_state.saved_conversations[st.session_state.current_conversation] = conv_data
        
        # Create new conversation
        st.session_state.conversation_count += 1
        new_conv_id = f"chat_{st.session_state.conversation_count}"
        st.session_state.current_conversation = new_conv_id
        st.session_state.messages = []
        st.rerun()
    
    # Conversation history dropdown
    if st.session_state.saved_conversations:
        conv_options = [f"Chat {cid.split('_')[1]}" for cid in st.session_state.saved_conversations.keys()]
        selected_conv = st.selectbox(
            "Previous Chats",
            options=conv_options,
            index=conv_options.index(f"Chat {st.session_state.current_conversation.split('_')[1]}")
            if st.session_state.current_conversation in st.session_state.saved_conversations 
            else 0
        )
        
        # Switch conversation if different one selected
        selected_conv_id = f"chat_{selected_conv.split(' ')[1]}"
        if selected_conv_id != st.session_state.current_conversation:
            if selected_conv_id in st.session_state.saved_conversations:
                st.session_state.current_conversation = selected_conv_id
                st.session_state.messages = st.session_state.saved_conversations[selected_conv_id]["messages"].copy()
                st.rerun()
    
    st.caption(f"Current: Chat {st.session_state.current_conversation.split('_')[1]}")

# Initialize model
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=model_name,
    temperature=temperature,
    max_tokens=max_tokens
)

# State schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond in {language}."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# LangGraph workflow
workflow = StateGraph(state_schema=State)

def call_model(state: State):
    messages = state["messages"][-max(1, context_length)*2:] if context_length > 0 else []
    prompt = prompt_template.invoke({
        "messages": messages,
        "language": state["language"]
    })
    
    with collect_runs() as cb:
        response = model.invoke(
            prompt,
            config={
                "metadata": {
                    "conversation_id": st.session_state.current_conversation,
                    "temperature": temperature,
                    "model": model_name,
                    "context_length": context_length
                },
                "tags": ["chatbot", f"conv-{st.session_state.current_conversation}"]
            }
        )
        run_id = cb.traced_runs[0].id if cb.traced_runs else None
    
    return {"messages": [response], "run_id": run_id}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Chat interface
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare input
    message_history = [HumanMessage(content=msg["content"]) for msg in st.session_state.messages if msg["role"] == "user"]
    
    # Get response
    with st.spinner("Thinking..."):
        result = None
        for step in app.stream({
            "messages": message_history,
            "language": language
        }, config={"configurable": {"thread_id": st.session_state.current_conversation}}):
            result = step
            if "model" in step:
                response = step["model"]["messages"][0].content
                run_id = step["model"].get("run_id")
                if run_id:
                    st.sidebar.markdown(f"🔗 [View in LangSmith](https://smith.langchain.com/runs/{run_id})")
    
    # Add response
    if result and "model" in result:
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Auto-save the updated conversation
        conv_data = {
            "messages": st.session_state.messages.copy(),
            "config": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "language": language,
                "model": model_name
            }
        }
        st.session_state.saved_conversations[st.session_state.current_conversation] = conv_data