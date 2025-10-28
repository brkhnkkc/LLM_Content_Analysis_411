import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="AI Agents Chat", layout="wide")

# Title
st.title("LLMs for Content Analysis")
st.markdown("Two Scientists (AI) Have a Conversation")

# Read API key from file
try:
    with open('secretKey.txt', 'r') as f:
        api_key = f.read().strip()
    st.sidebar.success("API Key ✅")
except FileNotFoundError:
    st.error("❌ Could not find the API key")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Simple chatbot class
class ChatAgent:
    def __init__(self, name, persona):
        self.name = name
        self.persona = persona
    
    def respond(self, message, conversation_history=""):
        """Generate a response based on the message and persona"""
        prompt = f"""You are {self.name}. {self.persona}
        
Previous conversation:
{conversation_history}

Respond to this message: {message}

Keep your response brief (2-3 sentences)."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Sidebar for configuration
st.sidebar.header("Agent Configuration")

# Agent 1 persona
agent1_persona = st.sidebar.text_area(
    "Agent 1 Persona",
    value="You are a curious and optimistic researcher who loves asking questions.",
    height=100,
    key="persona1"
)

# Agent 2 persona  
agent2_persona = st.sidebar.text_area(
    "Agent 2 Persona",
    value="You are a thoughtful and analytical researcher who provides detailed explanations.",
    height=100,
    key="persona2"
)

# Number of conversation turns
num_turns = st.sidebar.slider("Number of conversation turns", 1, 10, 3)

# Starting topic
starting_topic = st.sidebar.text_input(
    "Starting topic",
    value="What makes a good content analysis codebook?",
    key="topic"
)

st.sidebar.divider()
st.sidebar.info("Click the button to start the conversation!")

# Main area - always visible
st.divider()
st.write("Configure the agents in the sidebar, then click the button to start!")

if st.button("🚀 Start Conversation", type="primary", use_container_width=True):
    # Create agents
    agent1 = ChatAgent("Agent 1", agent1_persona)
    agent2 = ChatAgent("Agent 2", agent2_persona)
    
    # Initialize conversation
    conversation_history = []
    current_message = starting_topic
    
    st.subheader("💬 Conversation:")
    
    # Display starting topic
    st.info(f"**Starting Topic:** {starting_topic}")
    st.divider()
    
    # Have the conversation
    for turn in range(num_turns):
        # Agent 1 responds
        with st.container():
            st.markdown(f"### 🔵 Agent 1 (Turn {turn + 1})")
            with st.spinner("Agent 1 is thinking..."):
                response1 = agent1.respond(current_message, "\n".join(conversation_history))
            st.write(response1)
            conversation_history.append(f"Agent 1: {response1}")
        
        st.divider()
        
        # Agent 2 responds
        with st.container():
            st.markdown(f"### 🟢 Agent 2 (Turn {turn + 1})")
            with st.spinner("Agent 2 is thinking..."):
                response2 = agent2.respond(response1, "\n".join(conversation_history))
            st.write(response2)
            conversation_history.append(f"Agent 2: {response2}")
        
        st.divider()
        
        # Update current message for next turn
        current_message = response2
    
    st.success("✅ Conversation complete!")
    
    # Show full conversation log
    with st.expander("📝 View Full Conversation Log"):
        for i, message in enumerate(conversation_history):
            st.text(f"{i+1}. {message}")

st.divider()
st.caption("Powered by OpenAI GPT-4o-mini")