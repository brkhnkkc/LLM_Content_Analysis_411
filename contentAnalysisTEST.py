import streamlit as st
from openai import OpenAI
import pandas as pd
import json

# Page config
st.set_page_config(page_title="AI Agents Chat", layout="wide")

# Title
st.title("LLMs for Content Analysis")
st.markdown("Two Scientists (AI) Annotate Breast Cancer Narratives")

# Read API key from file
try:
    with open('secretKey.txt', 'r') as f:
        api_key = f.read().strip()
    st.sidebar.success("API Key Accepted")
except FileNotFoundError:
    st.error("Could not find the API key")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Codebook text
CODEBOOK = """
**Codebook for Coders**

The text was posted by breast cancer nonprofit organizations on Facebook
and contained narratives/stories related to breast cancer.

**Narrative Event(s) related to breast cancer "NE" [choose all that apply]**

Definition: A narrative event involves specific events or actions
experienced by a character or narrator in the post. Please code the
occurrences of all events in the post, such as "3", "2,4", or "2,4,5".

1. Prevention
2. Detection, diagnosis
3. Treatment 
   - Receiving treatment (e.g., getting the IV chemo, lying in the hospital bed)
   - Treatment effects (e.g., bald head, flat chest, wearing a head wrap)
   - Treatment milestone or completion (e.g., ringing the chemo bell, showing radiation therapy completion certificate).
4. Survivorship
   - Complete remission/cancer free; recurrence; a second cancer; and death.
   - Fundraising, any prosocial or philanthropic activities.

**Narrator perspective "NP" [choose one]**

Definition: Narrator is the person telling the story. When coding,
prioritize a perspective that is NOT the breast cancer organization.

1. Breast cancer survivor 
2. Breast cancer survivor's family or friends 
3. Mixed (i.e., survivor + family or friends)
4. Journalists/news media 
5. Breast cancer organization
"""

# Chatbot class
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
    
    def annotate(self, text_entry, codebook):
        """Independently annotate a text entry based on the codebook"""
        prompt = f"""You are {self.name}, a social science researcher with the following characteristics:
{self.persona}

Your task is to annotate the following Facebook post according to the provided codebook.

CODEBOOK:
{codebook}

FACEBOOK POST TO ANNOTATE:
{text_entry}

Please provide your annotation in the following JSON format:
{{
    "NE": "your answer for Narrative Events (e.g., '2,3' or '1' or '2,4,5')",
    "NP": "your answer for Narrator Perspective (single number 1-5)",
    "reasoning": "brief explanation of your choices"
}}

Be precise and follow the codebook strictly. For NE, list all that apply separated by commas. For NP, choose only one number."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except:
            # Fallback parsing if JSON isn't perfect
            content = response.choices[0].message.content
            # Try to extract NE, NP, and reasoning
            return {
                "NE": "error",
                "NP": "error",
                "reasoning": content
            }

# Sidebar for configuration
st.sidebar.header("Agent Configuration")

# Agent 1 persona
agent1_persona = st.sidebar.text_area(
    "Emily Carter",
    value="You are Dr. Emily Carter, a 45-year-old Caucasian female social scientist with a Ph.D. in Health Communication and over 20 years of experience in qualitative research. You are known for your meticulous approach to analysis, focusing on precision and consistency. As you analyze the data, ensure that each element is carefully examined and categorized. Pay close attention to the details, and make decisions based on thorough reasoning. Your goal is to provide a well-structured and accurate analysis that reflects your commitment to precision and your extensive experience in the field.",
    height=100,
    key="persona1"
)

# Agent 2 persona  
agent2_persona = st.sidebar.text_area(
    "Michael Rodriguez",
    value="You are Dr. Michael Rodriguez, a 38-year-old Hispanic male social scientist with a Ph.D. in Sociology and 15 years of experience in analyzing social dynamics and health narratives. You are known for your intuitive and empathetic approach to research, focusing on the emotional tone and social context. As you analyze the data, consider the broader implications and the underlying human experiences. Your goal is to capture the nuances and emotional depth of the data, reflecting your understanding of the social dynamics and your commitment to empathy and insight.",
    height=100,
    key="persona2"
)

st.sidebar.divider()

# Round selector
training_round = st.sidebar.selectbox(
    "Training Round",
    options=[1, 2, 3],
    help="Select which round of training (10 posts each)"
)

st.sidebar.divider()

# File upload for dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset (Excel or CSV)",
    type=['xlsx', 'csv']
)

# Main area - Create tabs
tab1, tab2, tab3 = st.tabs(["📚 Codebook", "🔍 Annotation", "💬 Discussion"])

# TAB 1: Codebook
with tab1:
    st.header("Codebook")
    st.markdown(CODEBOOK)
    st.info("This codebook will be used by both agents to annotate the Facebook posts.")

# TAB 2: Annotation
with tab2:
    st.header(f"Phase II: Independent Annotation - Round {training_round}")
    
    if uploaded_file is None:
        st.warning("⚠️ Please upload the dataset_30examples.xlsx file in the sidebar to begin annotation.")
        st.stop()
    
    # Load the Excel or CSV file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success(f"✅ Dataset loaded: {len(df)} total posts")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
    
    # Get posts for current round (10 posts per round)
    start_idx = (training_round - 1) * 10
    end_idx = start_idx + 10
    round_posts = df.iloc[start_idx:end_idx]
    
    st.info(f"Annotating posts {start_idx + 1} to {end_idx} (Round {training_round})")
    
    # Show preview of posts
    with st.expander("📄 Preview Posts to Annotate"):
        for i, (idx, row) in enumerate(round_posts.iterrows()):
            st.write(f"**Post {i + 1} (Message ID: {row['Message Id']}):**")
            st.write(f"{row['content']}")
            st.caption(f"Ground Truth - NE: {row['NE']}, NP: {row['NP']}")
            st.divider()
    
    st.divider()
    
    # Annotation button
    if st.button("🚀 Start Independent Annotation", type="primary", use_container_width=True):
        # Create agents
        agent1 = ChatAgent("Emily Carter", agent1_persona)
        agent2 = ChatAgent("Michael Rodriguez", agent2_persona)
        
        st.subheader("Annotation in Progress...")
        
        # Store annotations
        annotations_emily = []
        annotations_michael = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Annotate each post
        for i, (idx, row) in enumerate(round_posts.iterrows()):
            status_text.text(f"Annotating post {i + 1}/10...")
            
            # Emily annotates
            ann_emily = agent1.annotate(row['content'], CODEBOOK)
            annotations_emily.append(ann_emily)
            
            # Michael annotates
            ann_michael = agent2.annotate(row['content'], CODEBOOK)
            annotations_michael.append(ann_michael)
            
            progress_bar.progress((i + 1) / 10)
        
        status_text.text("✅ Annotation complete!")
        progress_bar.empty()
        
        # Store in session state
        st.session_state.annotations_emily = annotations_emily
        st.session_state.annotations_michael = annotations_michael
        st.session_state.round_posts = round_posts
        st.session_state.current_round = training_round
        
        st.success("✅ Both agents have completed their annotations!")
        st.rerun()
    
    # Display results if annotations exist
    if 'annotations_emily' in st.session_state and st.session_state.current_round == training_round:
        st.divider()
        st.subheader("Annotation Results")
        
        # Create results dataframe
        results_data = []
        agreements_ne = 0
        agreements_np = 0
        
        for i, (idx, row) in enumerate(st.session_state.round_posts.iterrows()):
            emily_ann = st.session_state.annotations_emily[i]
            michael_ann = st.session_state.annotations_michael[i]
            
            # Check agreement
            agree_ne = emily_ann['NE'] == michael_ann['NE']
            agree_np = emily_ann['NP'] == michael_ann['NP']
            
            if agree_ne:
                agreements_ne += 1
            if agree_np:
                agreements_np += 1
            
            results_data.append({
                'Post #': i + 1,
                'Content Preview': row['content'][:50] + "...",
                'Emily NE': emily_ann['NE'],
                'Michael NE': michael_ann['NE'],
                'Ground Truth NE': row['NE'],
                'Agree NE': '✅' if agree_ne else '❌',
                'Emily NP': emily_ann['NP'],
                'Michael NP': michael_ann['NP'],
                'Ground Truth NP': row['NP'],
                'Agree NP': '✅' if agree_np else '❌'
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agreement Rate (NE)", f"{agreements_ne}/10")
        col2.metric("Agreement Rate (NP)", f"{agreements_np}/10")
        
        # Calculate accuracy
        emily_correct_ne = sum(1 for i, row in enumerate(st.session_state.round_posts.iterrows()) 
                               if str(st.session_state.annotations_emily[i]['NE']) == str(row[1]['NE']))
        michael_correct_ne = sum(1 for i, row in enumerate(st.session_state.round_posts.iterrows()) 
                                 if str(st.session_state.annotations_michael[i]['NE']) == str(row[1]['NE']))
        emily_correct_np = sum(1 for i, row in enumerate(st.session_state.round_posts.iterrows()) 
                               if str(st.session_state.annotations_emily[i]['NP']) == str(row[1]['NP']))
        michael_correct_np = sum(1 for i, row in enumerate(st.session_state.round_posts.iterrows()) 
                                 if str(st.session_state.annotations_michael[i]['NP']) == str(row[1]['NP']))
        
        col3.metric("Emily Accuracy", f"{(emily_correct_ne + emily_correct_np)}/20")
        col4.metric("Michael Accuracy", f"{(michael_correct_ne + michael_correct_np)}/20")
        
        st.divider()
        
        # Show results table
        st.dataframe(results_df, use_container_width=True)
        
        # Show detailed annotations with reasoning
        with st.expander("📝 View Detailed Annotations with Reasoning"):
            for i, (idx, row) in enumerate(st.session_state.round_posts.iterrows()):
                st.write(f"**Post {i + 1}:** {row['content']}")
                st.write(f"**Ground Truth:** NE={row['NE']}, NP={row['NP']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Emily's Annotation:**")
                    st.json(st.session_state.annotations_emily[i])
                with col2:
                    st.write("**Michael's Annotation:**")
                    st.json(st.session_state.annotations_michael[i])
                
                st.divider()

# TAB 3: Discussion (placeholder for next phase)
with tab3:
    st.header("Phase III: Discussion of Disagreements")
    st.info("This phase will be implemented next. Here, agents will discuss posts where they disagreed.")
    
    if 'annotations_emily' in st.session_state:
        # Find disagreements
        disagreements = []
        for i, (idx, row) in enumerate(st.session_state.round_posts.iterrows()):
            emily_ann = st.session_state.annotations_emily[i]
            michael_ann = st.session_state.annotations_michael[i]
            
            if emily_ann['NE'] != michael_ann['NE'] or emily_ann['NP'] != michael_ann['NP']:
                disagreements.append(i + 1)
        
        if disagreements:
            st.warning(f"Found {len(disagreements)} posts with disagreements: {disagreements}")
        else:
            st.success("No disagreements found! Both agents agreed on all annotations.")

st.divider()