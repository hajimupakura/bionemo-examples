import streamlit as st
import anthropic
import time

# Set up Streamlit app
st.title("Drug Discovery Playground 🤖🧪")
st.caption("Powered by Anthropic's Claude - Ask about drugs, genes, or clinical trials!")

# Initialize Anthropic client
# Use environment variable or configuration file for API key
import os

# Get API key from environment variable or use a placeholder
api_key = os.environ.get("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
client = anthropic.Anthropic(api_key=api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm Claude, your drug discovery assistant. Ask me anything!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def get_claude_response(prompt):
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.3,
            system="You are an expert drug discovery scientist. Provide detailed, accurate answers about pharmaceuticals, genes, and clinical trials.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

# User input
if prompt := st.chat_input("Ask a drug discovery question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Claude is thinking..."):
            full_response = get_claude_response(prompt)
            time.sleep(0.05)  # Add a small delay for smoother streaming appearance
        
        st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a sidebar with information
with st.sidebar:
    st.markdown("""
    ### About this app
    This drug discovery assistant is powered by Anthropic's Claude AI. 
    You can ask questions about:
    - Drug mechanisms and interactions
    - Gene functions and pathways
    - Clinical trials and research
    - Pharmaceutical development
    - Molecular biology
    
    ### Note
    This is for informational purposes only. Always consult healthcare professionals for medical advice.
    """)