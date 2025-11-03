import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import io

# Page config
st.set_page_config(page_title="AI Agents Chat", layout="wide")
st.title("LLMs for Content Analysis")
st.markdown("Two Scientists (AI) Annotate Breast Cancer Narratives")

# Read API key
try:
    with open('secretKey.txt', 'r') as f:
        api_key = f.read().strip()
    st.sidebar.success("API Key Accepted")
except FileNotFoundError:
    st.error("Could not find the API key")
    st.stop()

client = OpenAI(api_key=api_key)

# Codebook
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

# Initialize session state
if 'api_costs' not in st.session_state:
    st.session_state.api_costs = {'total_tokens': 0, 'total_cost': 0.0}
if 'current_codebook' not in st.session_state:
    st.session_state.current_codebook = CODEBOOK
    st.session_state.codebook_history = [{"round": 0, "codebook": CODEBOOK}]

# Chatbot class
class ChatAgent:
    def __init__(self, name, persona):
        self.name = name
        self.persona = persona
    
    def call_api(self, prompt, temp=0.3):
        """Unified API call with cost tracking"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        usage = response.usage
        st.session_state.api_costs['total_tokens'] += usage.total_tokens
        st.session_state.api_costs['total_cost'] += (usage.prompt_tokens/1e6)*0.15 + (usage.completion_tokens/1e6)*0.60
        return response.choices[0].message.content
    
    def parse_json(self, content):
        """Parse JSON with fallbacks"""
        try:
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            import re
            return {
                "NE": re.findall(r'"NE"\s*:\s*"([^"]+)"', content)[-1] if re.findall(r'"NE"\s*:\s*"([^"]+)"', content) else "parse_error",
                "NP": re.findall(r'"NP"\s*:\s*"([^"]+)"', content)[-1] if re.findall(r'"NP"\s*:\s*"([^"]+)"', content) else "parse_error",
                "reasoning": re.findall(r'"reasoning"\s*:\s*"([^"]+)"', content)[-1] if re.findall(r'"reasoning"\s*:\s*"([^"]+)"', content) else content[:200]
            }
    
    def annotate(self, text, codebook):
        prompt = f"""You are {self.name}, a social science researcher with the following characteristics:
{self.persona}

Your task is to annotate the following Facebook post according to the provided codebook.

CODEBOOK:
{codebook}

FACEBOOK POST TO ANNOTATE:
{text}

Please provide your annotation in the following JSON format:
{{"NE": "your answer for Narrative Events (e.g., '2,3' or '1' or '2,4,5')", "NP": "your answer for Narrator Perspective (single number 1-5)", "reasoning": "brief explanation of your choices"}}

Be precise and follow the codebook strictly. For NE, list all that apply separated by commas. For NP, choose only one number."""
        return self.parse_json(self.call_api(prompt, 0.3))
    
    def discuss(self, text, own_ann, other_ann, other_name, codebook, history):
        history_text = "\n".join(history) if history else "This is the first round of discussion."
        prompt = f"""You are {self.name}, a social science researcher with the following characteristics:
{self.persona}

You are discussing a Facebook post where you and {other_name} had different annotations.

CODEBOOK: {codebook}
FACEBOOK POST: {text}

YOUR ANNOTATION: NE={own_ann['NE']}, NP={own_ann['NP']}, Reasoning={own_ann['reasoning']}
{other_name}'S ANNOTATION: NE={other_ann['NE']}, NP={other_ann['NP']}, Reasoning={other_ann['reasoning']}

PREVIOUS DISCUSSION: {history_text}

Please provide your perspective on this disagreement in 3-4 sentences."""
        return self.call_api(prompt, 0.6)
    
    def final_decision(self, text, own_ann, other_ann, history, codebook):
        prompt = f"""You are {self.name}. Based on the discussion below, provide your FINAL annotation decision.

CODEBOOK: {codebook}
FACEBOOK POST: {text}
YOUR INITIAL: NE={own_ann['NE']}, NP={own_ann['NP']}
OTHER INITIAL: NE={other_ann['NE']}, NP={other_ann['NP']}
DISCUSSION: {chr(10).join(history)}

Provide in JSON format: {{"NE": "final NE", "NP": "final NP", "reasoning": "brief explanation"}}"""
        return self.parse_json(self.call_api(prompt, 0.3))

# Sidebar
st.sidebar.header("Agent Configuration")
agent1_persona = st.sidebar.text_area("Emily Carter", 
    value="You are Dr. Emily Carter, a 45-year-old Caucasian female social scientist with a Ph.D. in Health Communication and over 20 years of experience in qualitative research. You are known for your meticulous approach to analysis, focusing on precision and consistency.", 
    height=80, key="p1")
agent2_persona = st.sidebar.text_area("Michael Rodriguez",
    value="You are Dr. Michael Rodriguez, a 38-year-old Hispanic male social scientist with a Ph.D. in Sociology and 15 years of experience in analyzing social dynamics and health narratives. You are known for your intuitive and empathetic approach to research, focusing on the emotional tone and social context.",
    height=80, key="p2")

st.sidebar.divider()
st.sidebar.subheader("💰 API Usage")
st.sidebar.metric("Total Cost", f"${st.session_state.api_costs['total_cost']:.4f}")
st.sidebar.metric("Total Tokens", f"{st.session_state.api_costs['total_tokens']:,}")
remaining = 10.0 - st.session_state.api_costs['total_cost']
if remaining < 2.0:
    st.sidebar.warning(f"⚠️ Low budget: ${remaining:.2f}")

st.sidebar.divider()
training_round = st.sidebar.selectbox("Training Round", [1, 2, 3], help="Select which round of training (10 posts each)")
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload dataset (Excel or CSV)", type=['xlsx', 'csv'])

# Helper functions
def load_data(file):
    return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)

def calc_metrics(posts, ann_e, ann_m):
    agree_ne = sum(1 for i in range(len(posts)) if ann_e[i]['NE'] == ann_m[i]['NE'])
    agree_np = sum(1 for i in range(len(posts)) if ann_e[i]['NP'] == ann_m[i]['NP'])
    acc_e_ne = sum(1 for i, (_, row) in enumerate(posts.iterrows()) if str(ann_e[i]['NE']) == str(row['NE']))
    acc_e_np = sum(1 for i, (_, row) in enumerate(posts.iterrows()) if str(ann_e[i]['NP']) == str(row['NP']))
    acc_m_ne = sum(1 for i, (_, row) in enumerate(posts.iterrows()) if str(ann_m[i]['NE']) == str(row['NE']))
    acc_m_np = sum(1 for i, (_, row) in enumerate(posts.iterrows()) if str(ann_m[i]['NP']) == str(row['NP']))
    return agree_ne, agree_np, acc_e_ne, acc_e_np, acc_m_ne, acc_m_np

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📚 Codebook", "🔍 Annotation", "💬 Discussion", "📝 Update Codebook", "📊 Metrics"])

# TAB 1: Codebook
with tab1:
    st.header("Codebook")
    st.info(f"**Currently viewing:** Round {st.session_state.get('completed_rounds', 0)} Codebook")
    st.markdown(st.session_state.current_codebook)
    
    if len(st.session_state.codebook_history) > 1:
        with st.expander("📜 View Codebook History"):
            for i, entry in enumerate(st.session_state.codebook_history):
                st.write(f"**Round {entry['round']} Codebook:**")
                st.text_area(f"Round {entry['round']}", entry['codebook'], height=200, disabled=True, key=f"h{i}")
                st.divider()

# TAB 2: Annotation
with tab2:
    st.header(f"Phase II: Independent Annotation - Round {training_round}")
    
    if not uploaded_file:
        st.warning("⚠️ Please upload dataset")
        st.stop()
    
    df = load_data(uploaded_file)
    st.success(f"✅ Dataset loaded: {len(df)} posts")
    
    start_idx = (training_round - 1) * 10
    round_posts = df.iloc[start_idx:start_idx + 10]
    st.info(f"Annotating posts {start_idx + 1} to {start_idx + 10}")
    
    with st.expander("📄 Preview Posts"):
        for i, (_, row) in enumerate(round_posts.iterrows()):
            st.write(f"**Post {i+1}:** {row['content']}")
            st.caption(f"Ground Truth - NE: {row['NE']}, NP: {row['NP']}")
            st.divider()
    
    if st.button("🚀 Start Independent Annotation", type="primary", use_container_width=True):
        agent1 = ChatAgent("Emily Carter", agent1_persona)
        agent2 = ChatAgent("Michael Rodriguez", agent2_persona)
        
        progress = st.progress(0)
        status = st.empty()
        ann_e, ann_m = [], []
        
        for i, (_, row) in enumerate(round_posts.iterrows()):
            status.text(f"Annotating post {i + 1}/10...")
            ann_e.append(agent1.annotate(row['content'], st.session_state.current_codebook))
            ann_m.append(agent2.annotate(row['content'], st.session_state.current_codebook))
            progress.progress((i + 1) / 10)
        
        status.text("✅ Complete!")
        progress.empty()
        
        st.session_state.annotations_emily = ann_e
        st.session_state.annotations_michael = ann_m
        st.session_state.round_posts = round_posts
        st.session_state.current_round = training_round
        st.rerun()
    
    if 'annotations_emily' in st.session_state and st.session_state.current_round == training_round:
        st.divider()
        st.subheader("Annotation Results")
        
        agree_ne, agree_np, acc_e_ne, acc_e_np, acc_m_ne, acc_m_np = calc_metrics(
            st.session_state.round_posts, 
            st.session_state.annotations_emily,
            st.session_state.annotations_michael
        )
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Agreement (NE)", f"{agree_ne}/10")
        col2.metric("Agreement (NP)", f"{agree_np}/10")
        col3.metric("Emily Accuracy", f"{acc_e_ne + acc_e_np}/20")
        col4.metric("Michael Accuracy", f"{acc_m_ne + acc_m_np}/20")
        
        results = []
        for i, (_, row) in enumerate(st.session_state.round_posts.iterrows()):
            e, m = st.session_state.annotations_emily[i], st.session_state.annotations_michael[i]
            results.append({
                'Post #': i + 1, 'Message ID': row['Message Id'], 'Content': row['content'][:80] + "...",
                'Emily NE': e['NE'], 'Michael NE': m['NE'], 'Truth NE': row['NE'], 'Agree NE': '✅' if e['NE']==m['NE'] else '❌',
                'Emily NP': e['NP'], 'Michael NP': m['NP'], 'Truth NP': row['NP'], 'Agree NP': '✅' if e['NP']==m['NP'] else '❌'
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        st.download_button("📥 Download CSV", output.getvalue(), f"round_{training_round}_annotations.csv", "text/csv")

# TAB 3: Discussion
with tab3:
    st.header("Phase III: Discussion of Disagreements")
    
    if 'annotations_emily' not in st.session_state:
        st.warning("⚠️ Complete Phase II first")
        st.stop()
    
    disagreements = []
    for i, (_, row) in enumerate(st.session_state.round_posts.iterrows()):
        e, m = st.session_state.annotations_emily[i], st.session_state.annotations_michael[i]
        if e['NE'] != m['NE'] or e['NP'] != m['NP']:
            disagreements.append({
                'index': i, 'post_num': i + 1, 'content': row['content'],
                'emily_ann': e, 'michael_ann': m, 'ground_truth': {'NE': row['NE'], 'NP': row['NP']}
            })
    
    if not disagreements:
        st.success("🎉 No disagreements!")
        st.stop()
    
    st.info(f"Found **{len(disagreements)}** disagreements")
    
    if st.button("💬 Start Discussion (3 rounds)", type="primary", use_container_width=True):
        agent_e = ChatAgent("Emily Carter", agent1_persona)
        agent_m = ChatAgent("Michael Rodriguez", agent2_persona)
        
        discussion_results = []
        progress = st.progress(0)
        
        for d_idx, d in enumerate(disagreements):
            st.markdown(f"### 📌 Post {d['post_num']}")
            with st.expander("View Content", expanded=True):
                st.write(d['content'])
            
            history = []
            for r in range(3):
                st.write(f"**💬 Round {r + 1}**")
                e_resp = agent_e.discuss(d['content'], d['emily_ann'], d['michael_ann'], "Michael", st.session_state.current_codebook, history)
                history.append(f"Emily: {e_resp}")
                st.write(f"🔵 **Emily:** {e_resp}")
                
                m_resp = agent_m.discuss(d['content'], d['michael_ann'], d['emily_ann'], "Emily", st.session_state.current_codebook, history)
                history.append(f"Michael: {m_resp}")
                st.write(f"🟢 **Michael:** {m_resp}")
            
            st.write("**🎯 Final Decisions:**")
            e_final = agent_e.final_decision(d['content'], d['emily_ann'], d['michael_ann'], history, st.session_state.current_codebook)
            m_final = agent_m.final_decision(d['content'], d['michael_ann'], d['emily_ann'], history, st.session_state.current_codebook)
            
            col1, col2, col3 = st.columns(3)
            col1.write(f"🔵 **Emily Final:** NE={e_final['NE']}, NP={e_final['NP']}")
            col2.write(f"🟢 **Michael Final:** NE={m_final['NE']}, NP={m_final['NP']}")
            col3.write(f"✅ **Truth:** NE={d['ground_truth']['NE']}, NP={d['ground_truth']['NP']}")
            
            discussion_results.append({
                'post_num': d['post_num'], 'index': d['index'], 'discussion': history,
                'emily_initial': d['emily_ann'], 'michael_initial': d['michael_ann'],
                'emily_final': e_final, 'michael_final': m_final, 'ground_truth': d['ground_truth']
            })
            
            st.markdown("---")
            progress.progress((d_idx + 1) / len(disagreements))
        
        progress.empty()
        st.session_state.discussion_results = discussion_results
        st.success("✅ Discussions complete!")

# TAB 4: Update Codebook
with tab4:
    st.header("Phase IV: Update Codebook")
    
    if 'discussion_results' not in st.session_state:
        st.warning("⚠️ Complete Phase III first")
        st.stop()
    
    st.info(f"**Round {training_round}:** {len(st.session_state.discussion_results)} disagreements discussed")
    
    if st.button("🔄 Generate Updated Codebook", type="primary", use_container_width=True):
        with st.spinner("Updating codebook..."):
            summary = []
            for d in st.session_state.discussion_results:
                post = st.session_state.round_posts.iloc[d['index']]['content']
                summary.append(f"""POST: {post[:200]}...
DISAGREEMENT: Emily NE={d['emily_initial']['NE']}, NP={d['emily_initial']['NP']} | Michael NE={d['michael_initial']['NE']}, NP={d['michael_initial']['NP']}
DISCUSSION: {chr(10).join(d['discussion'][:4])}
FINAL: Emily NE={d['emily_final']['NE']}, NP={d['emily_final']['NP']} | Michael NE={d['michael_final']['NE']}, NP={d['michael_final']['NP']}
TRUTH: NE={d['ground_truth']['NE']}, NP={d['ground_truth']['NP']}""")
            
            prompt = f"""You are a senior social science researcher refining a codebook for content analysis.

CURRENT CODEBOOK:
{st.session_state.current_codebook}

DISAGREEMENTS FROM ROUND {training_round}:
{chr(10).join(summary)}

Update the codebook to make it clearer. Focus on: 1) Adding clarifications where annotators disagreed 2) Providing specific examples for ambiguous cases 3) Refining confusing definitions 4) Keeping same structure but improving clarity

Provide the complete updated codebook:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            usage = response.usage
            st.session_state.api_costs['total_tokens'] += usage.total_tokens
            st.session_state.api_costs['total_cost'] += (usage.prompt_tokens/1e6)*0.15 + (usage.completion_tokens/1e6)*0.60
            
            st.session_state.updated_codebook = response.choices[0].message.content
            
        st.success("✅ Updated!")
        st.rerun()
    
    if 'updated_codebook' in st.session_state:
        st.subheader("Updated Codebook")
        col1, col2 = st.columns(2)
        col1.text_area("Previous", st.session_state.current_codebook, height=400, disabled=True, key="prev")
        col2.text_area("Updated", st.session_state.updated_codebook, height=400, disabled=True, key="new")
        
        st.divider()
        st.subheader("💾 Apply Updated Codebook")
        
        if training_round < 3:
            if st.button(f"📥 Save & Prepare for Round {training_round + 1}", type="primary", use_container_width=True):
                if 'round_metrics' not in st.session_state:
                    st.session_state.round_metrics = []
                
                agree_ne, agree_np, acc_e_ne, acc_e_np, acc_m_ne, acc_m_np = calc_metrics(
                    st.session_state.round_posts,
                    st.session_state.annotations_emily,
                    st.session_state.annotations_michael
                )
                
                final_agree_ne = sum(1 for d in st.session_state.discussion_results if d['emily_final']['NE']==d['michael_final']['NE'])
                final_agree_np = sum(1 for d in st.session_state.discussion_results if d['emily_final']['NP']==d['michael_final']['NP'])
                
                st.session_state.round_metrics.append({
                    'round': training_round,
                    'initial_agreement_ne': agree_ne/10, 'initial_agreement_np': agree_np/10,
                    'emily_accuracy_ne': acc_e_ne/10, 'emily_accuracy_np': acc_e_np/10,
                    'michael_accuracy_ne': acc_m_ne/10, 'michael_accuracy_np': acc_m_np/10,
                    'num_disagreements': len(st.session_state.discussion_results),
                    'final_agreement_ne': final_agree_ne/len(st.session_state.discussion_results) if st.session_state.discussion_results else 1.0,
                    'final_agreement_np': final_agree_np/len(st.session_state.discussion_results) if st.session_state.discussion_results else 1.0
                })
                
                st.session_state.current_codebook = st.session_state.updated_codebook
                st.session_state.codebook_history.append({"round": training_round, "codebook": st.session_state.updated_codebook})
                st.session_state.completed_rounds = training_round
                
                for key in ['annotations_emily', 'annotations_michael', 'round_posts', 'discussion_results', 'updated_codebook']:
                    st.session_state.pop(key, None)
                
                st.success(f"✅ Round {training_round} complete!")
                st.balloons()
        else:
            if st.button("✅ Finalize Codebook", type="primary", use_container_width=True):
                if 'round_metrics' not in st.session_state:
                    st.session_state.round_metrics = []
                
                agree_ne, agree_np, acc_e_ne, acc_e_np, acc_m_ne, acc_m_np = calc_metrics(
                    st.session_state.round_posts,
                    st.session_state.annotations_emily,
                    st.session_state.annotations_michael
                )
                
                final_agree_ne = sum(1 for d in st.session_state.discussion_results if d['emily_final']['NE']==d['michael_final']['NE'])
                final_agree_np = sum(1 for d in st.session_state.discussion_results if d['emily_final']['NP']==d['michael_final']['NP'])
                
                st.session_state.round_metrics.append({
                    'round': training_round,
                    'initial_agreement_ne': agree_ne/10, 'initial_agreement_np': agree_np/10,
                    'emily_accuracy_ne': acc_e_ne/10, 'emily_accuracy_np': acc_e_np/10,
                    'michael_accuracy_ne': acc_m_ne/10, 'michael_accuracy_np': acc_m_np/10,
                    'num_disagreements': len(st.session_state.discussion_results),
                    'final_agreement_ne': final_agree_ne/len(st.session_state.discussion_results) if st.session_state.discussion_results else 1.0,
                    'final_agreement_np': final_agree_np/len(st.session_state.discussion_results) if st.session_state.discussion_results else 1.0
                })
                
                st.session_state.finalized_codebook = st.session_state.updated_codebook
                st.session_state.completed_rounds = 3
                
                for key in ['annotations_emily', 'annotations_michael', 'round_posts', 'discussion_results', 'updated_codebook']:
                    st.session_state.pop(key, None)
                
                st.success("🎉 Codebook finalized!")
                st.info("📊 View the Metrics tab to see performance across all rounds")
                st.balloons()

# TAB 5: Metrics
with tab5:
    st.header("Phase V: Performance Metrics")
    
    if 'round_metrics' in st.session_state and st.session_state.round_metrics:
        st.subheader("📊 Training Phase (Rounds 1-3)")
        
        import matplotlib.pyplot as plt
        metrics_df = pd.DataFrame(st.session_state.round_metrics)
        
        # Table
        display_df = metrics_df.copy()
        for col in ['initial_agreement_ne', 'initial_agreement_np', 'emily_accuracy_ne', 'emily_accuracy_np', 
                    'michael_accuracy_ne', 'michael_accuracy_np', 'final_agreement_ne', 'final_agreement_np']:
            display_df[col] = (display_df[col] * 100).round(1).astype(str) + '%'
        st.dataframe(display_df, use_container_width=True)
        
        st.divider()
        
        # Agreement chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_df['round'], metrics_df['initial_agreement_ne']*100, 'o-', label='Initial NE', color='#FF6B6B', linewidth=2, markersize=8)
        ax.plot(metrics_df['round'], metrics_df['initial_agreement_np']*100, 'o-', label='Initial NP', color='#4ECDC4', linewidth=2, markersize=8)
        ax.plot(metrics_df['round'], metrics_df['final_agreement_ne']*100, 'o--', label='Final NE', color='#95E1D3', linewidth=2, markersize=8)
        ax.plot(metrics_df['round'], metrics_df['final_agreement_np']*100, 'o--', label='Final NP', color='#F38181', linewidth=2, markersize=8)
        ax.set_xlabel('Round', fontsize=12)
        ax.set_ylabel('Agreement (%)', fontsize=12)
        ax.set_title('Agreement Rate Across Training Rounds', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Accuracy chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(metrics_df['round'], metrics_df['emily_accuracy_ne']*100, 'o-', label='Emily NE', color='#667BC6', linewidth=2, markersize=8)
        ax2.plot(metrics_df['round'], metrics_df['emily_accuracy_np']*100, 'o-', label='Emily NP', color='#DA7297', linewidth=2, markersize=8)
        ax2.plot(metrics_df['round'], metrics_df['michael_accuracy_ne']*100, 'o-', label='Michael NE', color='#FFEAA7', linewidth=2, markersize=8)
        ax2.plot(metrics_df['round'], metrics_df['michael_accuracy_np']*100, 'o-', label='Michael NP', color='#74B9FF', linewidth=2, markersize=8)
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Against Ground Truth', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        if len(metrics_df) >= 2:
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            imp_ne = (metrics_df.iloc[-1]['initial_agreement_ne'] - metrics_df.iloc[0]['initial_agreement_ne']) * 100
            imp_np = (metrics_df.iloc[-1]['initial_agreement_np'] - metrics_df.iloc[0]['initial_agreement_np']) * 100
            col1.metric("Agreement Improvement (NE)", f"{imp_ne:+.1f}%")
            col2.metric("Agreement Improvement (NP)", f"{imp_np:+.1f}%")
            col3.metric("Emily Avg Acc", f"{((metrics_df['emily_accuracy_ne'].mean() + metrics_df['emily_accuracy_np'].mean())/2)*100:.1f}%")
            col4.metric("Michael Avg Acc", f"{((metrics_df['michael_accuracy_ne'].mean() + metrics_df['michael_accuracy_np'].mean())/2)*100:.1f}%")

st.divider()