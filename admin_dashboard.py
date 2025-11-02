# admin_dashboard.py (Version 4: Added CSV Download Feature)

import streamlit as st
import pandas as pd
import io
import uuid
import os 
import plotly.express as px 
import matplotlib.pyplot as plt
import datetime
import spacy
# --- Imports from Friend's FastAPI Backend Logic ---
import subprocess
import sys
import re
import csv
from collections import Counter
from typing import List, Optional, Dict, Tuple 
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from functools import lru_cache 

# Suppress Matplotlib warnings/backend issues in Streamlit
plt.rcParams.update({'figure.max_open_warning': 0})

# --- 1. GLOBAL CONFIGURATION & MODEL LOADING ---

# Define MCA Color Palette
MCA_COLORS = {
    "MCA_BLUE_DARK": "#1b3a6d",       # Header/Primary Bar
    "MCA_BLUE_LIGHT": "#48a6a6",      # Accent Blue, Hover Color
    "MCA_RED": "#c53030",             # Red emphasis (Negative Sentiment)
    "MCA_GREEN": "#27ae60",           # Green accent (Positive Sentiment)
    "MCA_GRAY_BG": "#f7f7f7",         # Light background (used for app background)
    "MCA_LIGHT_BLUE_BG": "#e6f0fa",   # Very light blue (Subtle section background)
    "MCA_ORANGE_DARK": "#ff9933",     # Orange/Saffron
    "MCA_FOOTER_BG": "#354e66",       # Dark slate blue
    "NEUTRAL_GRAY": "#8c8c8c",        # Standard Gray for Neutral
}

# Map sentiment labels to MCA colors
SENTIMENT_COLORS = {
    "Positive": MCA_COLORS["MCA_GREEN"], 
    "Negative": MCA_COLORS["MCA_RED"], 
    "Neutral": MCA_COLORS["NEUTRAL_GRAY"],
}

# Global Model Loading
@st.cache_resource
def load_analysis_models():
    """Load heavy models once and cache them."""
    try:
        # Load the medium-sized spaCy model for NER and similarity checks
        NLP_MODEL = spacy.load("en_core_web_md")
    except NameError:
        st.error("spaCy not found. Please ensure spaCy is installed: pip install spacy")
        st.stop()
    except OSError:
        st.info("Downloading spaCy model 'en_core_web_md'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
        NLP_MODEL = spacy.load("en_core_web_md")

    SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
    
    # Use standard WordCloud STOPWORDS for better generation
    STOPWORDS_SET = set(STOPWORDS)

    return NLP_MODEL, SENTIMENT_ANALYZER, STOPWORDS_SET

# Check and install spacy if missing
try:
    import spacy
except ImportError:
    st.info("Installing spaCy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy
    
NLP_MODEL, SENTIMENT_ANALYZER, STOPWORDS_SET = load_analysis_models()

# --- 2. ADAPTED ANALYSIS CORE FUNCTIONS ---

@lru_cache(maxsize=128)
def expand_keywords(topic: str) -> List[str]:
    """Expands a single topic keyword into variations."""
    topic_low = topic.lower()
    return [topic_low, topic_low + "s", topic_low + "ing"]

def filter_comment_with_reason(
    comment: str, topic_doc: spacy.tokens.doc.Doc, topic_keywords: List[str], seen_comments: set
) -> Tuple[bool, str]:
    """
    Filters a single comment based on relevance and quality, returning the reason for exclusion.
    """
    comment_lower = comment.strip().lower()
    
    if not comment_lower: 
        return False, "Comment removed: insufficient detail."
    if not any(c.isalnum() for c in comment_lower): 
        return False, "Comment filtered: contains only emojis/symbols."
    if len(comment_lower.split()) < 3: 
        return False, "Comment removed: insufficient detail (under 3 words)."
    if comment_lower in seen_comments: 
        return False, "Comment filtered: duplicate entry."
    
    inappropriate = ["abuse", "spam", "fake", "nonsense", "http:", "https:"]
    if any(bad in comment_lower for bad in inappropriate): 
        return False, "Comment filtered: contains inappropriate language or links."
        
    if any(re.search(rf'\b{re.escape(kw)}\b', comment_lower) for kw in topic_keywords):
        seen_comments.add(comment_lower); return True, ""
        
    comment_doc = NLP_MODEL(comment_lower)
    if not comment_doc.has_vector or not comment_doc.vector_norm:
        return False, f"Comment filtered: could not be understood (no vector)."

    sim = topic_doc.similarity(comment_doc)
    if sim < 0.4:
        return False, f"Comment filtered: not relevant to the topic '{topic_doc.text}' (Similarity: {sim:.2F})."
        
    seen_comments.add(comment_lower)
    return True, ""

@lru_cache(maxsize=10000)
def analyze_sentiment(text: str) -> Tuple[float, str]:
    """Analyzes text and returns a compound score and a string label."""
    score = SENTIMENT_ANALYZER.polarity_scores(text)["compound"]
    if score >= 0.05: return score, "Positive"
    elif score <= -0.05: return score, "Negative"
    else: return score, "Neutral"

def generate_paragraph_summary(relevant_comments: List[str], topic: str) -> str:
    """Generates the narrative summary."""
    if not relevant_comments: return f"The comments on **{topic}** were too few or not relevant enough to generate a narrative summary."
    
    comments_tuple = tuple(relevant_comments) 
    analysis_results = [analyze_sentiment(c) for c in comments_tuple]
    sentiments = [s for _, s in analysis_results]
    sentiment_counts = Counter(sentiments)

    total = sum(sentiment_counts.values())
    if total == 0: overall_sent_str = "neutral"
    else:
        pos_perc = sentiment_counts.get("Positive", 0) / total
        neg_perc = sentiment_counts.get("Negative", 0) / total
        if pos_perc > 0.6: overall_sent_str = "overwhelmingly positive"
        elif neg_perc > 0.6: overall_sent_str = "overwhelmingly negative"
        elif pos_perc > neg_perc: overall_sent_str = "mixed, leaning positive"
        elif neg_perc > pos_perc: overall_sent_str = "mixed, leaning negative"
        else: overall_sent_str = "mixed"

    all_words = " ".join(relevant_comments).lower().split()
    topic_words = set(topic.lower().split())
    keywords_filtered = [w for w in all_words if w not in STOPWORDS_SET and w not in topic_words and len(w) > 2 and w.isalpha()]
    top_keywords = Counter(keywords_filtered).most_common(5)
    keyword_str = ", ".join([w for w, c in top_keywords]) or "various themes"

    detailed_summary = (
        f"A detailed review of **{total}** relevant comments about **{topic}** reveals a **{overall_sent_str}** sentiment. "
        f"Positive feedback (making up **{sentiment_counts.get('Positive', 0)}** comments) generally focuses on benefits and approval. "
        f"Negative remarks (**{sentiment_counts.get('Negative', 0)}** comments) often highlight concerns, dissatisfaction, or areas for improvement. "
        f"The most frequent themes identified, aside from the main topic, are **{keyword_str}**."
    )
    return detailed_summary

@st.cache_data(max_entries=10) 
def generate_wordcloud(text: str, color_map: str, background_color: str = "white") -> Optional[WordCloud]:
    """Generates a word cloud image object."""
    if not text.strip(): return None
    try:
        wc = WordCloud(
            width=400, height=300, background_color=background_color,
            max_words=100, collocations=False, stopwords=STOPWORDS_SET,
            colormap=color_map,
        ).generate(text)
        return wc
    except Exception:
        plt.close('all'); return None

def generate_sentiment_wordclouds(df: pd.DataFrame, policy_topic: str):
    """Generates four word clouds based on sentiment: Overall, Positive, Negative, Neutral."""
    
    # 1. Prepare texts
    overall_text = " ".join(df['comment'].tolist())
    pos_text = " ".join(df[df['Sentiment'] == 'Positive']['comment'].tolist())
    neg_text = " ".join(df[df['Sentiment'] == 'Negative']['comment'].tolist())
    neut_text = " ".join(df[df['Sentiment'] == 'Neutral']['comment'].tolist())
    
    # 2. Generate Word Clouds 
    wc_overall = generate_wordcloud(overall_text, color_map='winter_r', background_color=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
    wc_pos = generate_wordcloud(pos_text, color_map='Greens', background_color=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
    wc_neg = generate_wordcloud(neg_text, color_map='Reds', background_color=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
    wc_neut = generate_wordcloud(neut_text, color_map='Blues', background_color=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
    
    # 3. Display Word Clouds
    st.subheader("☁️ Sentiment-Specific Word Clouds")
    
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    def display_wc(container, wc, title):
        if wc:
            container.markdown(f"**{title}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            container.pyplot(fig)
        else:
            container.info(f"Not enough {title.lower()} comments for a word cloud.")

    display_wc(col1, wc_overall, "Overall Themes")
    display_wc(col2, wc_pos, "Positive Words")
    display_wc(col3, wc_neg, "Negative Words")
    display_wc(col4, wc_neut, "Neutral/Mixed Themes")
    
def plot_keyword_bar_chart(top_keywords: List[Tuple[str, int]], topic: str):
    """Generates a Plotly keyword horizontal bar chart for Streamlit."""
    if not top_keywords: 
        st.info("No common keywords found to plot.")
        return

    df_keywords = pd.DataFrame(top_keywords, columns=['Keyword', 'Count'])
    df_keywords = df_keywords.sort_values(by='Count', ascending=True) 

    fig = px.bar(
        df_keywords, 
        y='Keyword', 
        x='Count', 
        orientation='h',
        title=f'Top 5 Keywords for "{topic}"',
        color='Count',
        color_continuous_scale=px.colors.sequential.Teal,
    )
    fig.update_layout(
        xaxis_title="Frequency Count", 
        yaxis_title="Keyword",
        plot_bgcolor=MCA_COLORS["MCA_GRAY_BG"], 
        paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"],
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner="Running comprehensive NLP analysis...")
def process_comments_for_policy(comments_list: List[str], policy_topic: str) -> Tuple[pd.DataFrame, int, int, List[Dict[str, str]]]:
    """
    Runs the full analysis pipeline.
    Returns: df_res_sentiment, relevant_count, irrelevant_count, irrelevant_feedback (list of dicts)
    """
    topic_doc = NLP_MODEL(policy_topic.lower())
    topic_keywords = expand_keywords(policy_topic)
    
    irrelevant_feedback: List[Dict[str, str]] = []
    relevant_comments_map = {}; seen_comments = set()
    
    for comment in comments_list:
        is_ok, reason = filter_comment_with_reason(comment, topic_doc, topic_keywords, seen_comments)
        if is_ok: 
            relevant_comments_map[comment] = comment 
        else: 
            irrelevant_feedback.append({"Comment": comment, "Reason": reason})
        
    relevant_comments = list(relevant_comments_map.keys())
    relevant_count = len(relevant_comments)
    irrelevant_count = len(irrelevant_feedback) 

    if not relevant_comments:
        return pd.DataFrame({'Comment': [], 'Sentiment': [], 'Score': []}), 0, irrelevant_count, irrelevant_feedback

    analysis_results = []
    comments_tuple = tuple(relevant_comments)
    for comment in comments_tuple:
        score, sentiment = analyze_sentiment(comment)
        analysis_results.append({"Comment": comment, "Sentiment": sentiment, "Score": score})
        
    df_res_sentiment = pd.DataFrame(analysis_results)

    all_words = " ".join(relevant_comments).lower().split()
    topic_words = set(policy_topic.lower().split())
    keywords_filtered = [w for w in all_words if w not in STOPWORDS_SET and w not in topic_words and len(w) > 2 and w.isalpha()]
    top_themes = Counter(keywords_filtered).most_common(5)

    df_res_sentiment.attrs['top_themes'] = top_themes
    
    return df_res_sentiment, relevant_count, irrelevant_count, irrelevant_feedback

def load_external_comments(uploaded_file, text_input) -> Tuple[List[str], List[Optional[str]]]:
    """Handles loading comments from Streamlit file uploader or text area."""
    comments_list = []
    timestamps = []
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        
        try:
            decoded_content = file_bytes.decode("utf-8") 
        except UnicodeDecodeError:
            decoded_content = file_bytes.decode("cp1252") 
        
        csv_reader = csv.reader(io.StringIO(decoded_content))
        
        try:
            first_row = next(csv_reader)
            first_row_lower = [h.lower().strip() for h in first_row]
        except StopIteration:
            return [], []

        header_keywords = ['comment', 'text', 'timestamp', 'date']
        is_header = any(keyword in first_row_lower for keyword in header_keywords)
        
        comment_col = 0
        time_col = -1
        rows_to_process = []
        
        if is_header:
            header = first_row_lower
            if 'comment' in header: comment_col = header.index('comment')
            elif 'text' in header: comment_col = header.index('text')
            if 'timestamp' in header: time_col = header.index('timestamp')
            elif 'date' in header: time_col = header.index('date')
        else:
            rows_to_process.append(first_row)
        
        for row in csv_reader:
            rows_to_process.append(row)

        for row in rows_to_process:
            if len(row) > comment_col:
                comments_list.append(row[comment_col])
                timestamps.append(row[time_col] if time_col != -1 and len(row) > time_col else None)
        
    elif text_input.strip():
        comments_list = [c.strip() for c in text_input.split('\n') if c.strip()]
        timestamps = [None] * len(comments_list)
    
    return comments_list, timestamps


# --- 3. STREAMLIT APP STRUCTURE ---

st.set_page_config(page_title="MCA Admin Dashboard", layout="wide")

# Inject Custom CSS for Styling
custom_css = f"""
<style>
/* Main App Styling */
.stApp {{
    background-color: {MCA_COLORS["MCA_GRAY_BG"]}; 
}}

/* Header/Title Styling */
h1 {{
    color: {MCA_COLORS["MCA_BLUE_DARK"]}; 
    font-size: 2.5em;
    padding-top: 20px;
    padding-bottom: 5px;
}}
h3, h2 {{
    color: {MCA_COLORS["MCA_BLUE_DARK"]}; 
    border-left: 5px solid {MCA_COLORS["MCA_BLUE_LIGHT"]};
    padding-left: 10px;
    margin-top: 20px;
}}
h4 {{
    color: {MCA_COLORS["MCA_BLUE_DARK"]};
}}

/* Metric Cards */
[data-testid="stMetric"] {{
    background-color: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid #ddd;
}}

/* Custom Header with Image */
.custom-header {{
    background-color: {MCA_COLORS["MCA_BLUE_DARK"]};
    padding: 10px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 20px;
}}
.custom-header img {{
    height: 70px; /* Adjust size as needed */
    padding-right: 20px;
}}
.header-text h1 {{
    color: white !important; 
    font-size: 2em;
    margin: 0;
    padding: 0;
    border-bottom: none;
}}
.header-text p {{
    color: {MCA_COLORS["MCA_BLUE_LIGHT"]}; 
    font-size: 1em;
    margin: 0;
}}

/* Sidebar/Tabs */
[data-baseweb="tab"]:focus {{
    box-shadow: none !important;
    border-color: {MCA_COLORS["MCA_BLUE_DARK"]} !important;
}}
/* Content Separator */
.stDivider {{
    border-top: 1px solid {MCA_COLORS["MCA_BLUE_LIGHT"]};
    margin: 15px 0;
}}

/* Custom Download Button Styling */
div.stDownloadButton > button {{
    background-color: {MCA_COLORS["MCA_GREEN"]};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: bold;
}}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- CUSTOM HEADER IMPLEMENTATION ---
def display_header():
    """Displays the custom header with the MCA emblem."""
    # Helper to convert image to base64
    def image_to_base64(filepath):
        import base64
        try:
            with open(filepath, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            return None
            
    base64_img = image_to_base64('Emblem_of_India_svg.png')
    
    if base64_img:
        st.markdown(f"""
            <div class="custom-header">
                <div style="display: flex; align-items: center;">
                    <img src="data:image/png;base64,{base64_img}" alt="MCA Emblem">
                    <div class="header-text">
                        <h1>MCA Admin Policy Analysis Portal</h1>
                        <p>Empowering Business, Protecting Investors</p>
                    </div>
                </div>
                <button onclick="window.location.reload();" style="
                    background-color: {MCA_COLORS["MCA_BLUE_LIGHT"]}; 
                    color: white; 
                    border: none; 
                    padding: 8px 15px; 
                    border-radius: 5px; 
                    cursor: pointer;
                ">Logout</button>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.header("🛡️ Admin Policy Analysis Portal")
        st.sidebar.button("Logout", type="primary", on_click=lambda: st.session_state.clear())

display_header()
st.markdown("---")


# Ensure data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# -------------------------------------------------------------
## 🗄️ DATA INITIALIZATION/LOADING 
# -------------------------------------------------------------

if 'policies_df' not in st.session_state:
    if os.path.exists('data/policies.csv'):
        try:
            st.session_state.policies_df = pd.read_csv('data/policies.csv')
        except Exception:
            st.session_state.policies_df = pd.DataFrame({'id': [], 'title': [], 'summary': [], 'filename': []})
    else:
        st.session_state.policies_df = pd.DataFrame({'id': [], 'title': [], 'summary': [], 'filename': []})

if 'comments_df' not in st.session_state:
    if os.path.exists('data/comments.csv'):
        try:
            st.session_state.comments_df = pd.read_csv('data/comments.csv')
        except Exception:
            st.session_state.comments_df = pd.DataFrame({
                'policy_id': [], 'comment': [], 'timestamp': [],
                'profession': [], 'education': []
            })
    else:
        st.session_state.comments_df = pd.DataFrame({
            'policy_id': [], 'comment': [], 'timestamp': [],
            'profession': [], 'education': []
        })

# -------------------------------------------------------------
## 🚀 TABS AND FUNCTIONALITY 
# -------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🆕 Upload New Policy", "💬 Internal Comment Analysis", "🌐 External Data Analysis"])

## --- TAB 1: UPLOAD NEW POLICY ---
with tab1:
    st.header("Document Submission: Prepare for Stakeholder Feedback")
    st.markdown("---")
    with st.form("Policy_Upload_Form", clear_on_submit=True):
        title = st.text_input("Policy Title", placeholder="e.g., The Future of Sustainable Urban Planning")
        summary = st.text_area("Policy Brief/Summary (for Stakeholders)", height=150, placeholder="A clear, concise summary of the policy's key points and objectives.")
        uploaded_file = st.file_uploader("Upload Full Policy Document (PDF/DOCX)", type=['pdf', 'docx'])

        if st.form_submit_button("💾 Save New Policy and Publish", type="primary"):
            if title and summary and uploaded_file:
                file_ext = uploaded_file.name.split('.')[-1]
                unique_id = str(uuid.uuid4())
                filename = f"policy_{unique_id}.{file_ext}"
                
                with open(f"data/{filename}", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                new_policy = pd.DataFrame([{
                    'id': unique_id,
                    'title': title,
                    'summary': summary,
                    'filename': filename
                }])
                
                st.session_state.policies_df = pd.concat([st.session_state.policies_df, new_policy], ignore_index=True)
                st.session_state.policies_df.to_csv('data/policies.csv', index=False)
                
                st.success(f"Policy **'{title}'** uploaded successfully! Stakeholders can now view and comment.")
                st.rerun() 
            else:
                st.error("❗ Please complete the **Title**, **Summary**, and **File Upload** fields.")


## --- TAB 2: INTERNAL POLICY COMMENT ANALYSIS ---
with tab2:
    st.header("Deep Dive: Internal Stakeholder Feedback Analysis")
    st.markdown("---")
    
    policies_list = st.session_state.policies_df['title'].tolist()
    
    if not policies_list:
        st.info("No policies available for analysis. Please upload one via the 'Upload New Policy' tab.")
        
    else:
        analysis_policy_title = st.selectbox("Select Policy to Analyze:", policies_list, key="internal_policy_select")
        policy_id_to_analyze = st.session_state.policies_df[st.session_state.policies_df['title'] == analysis_policy_title].iloc[0]['id']
        policy_topic = analysis_policy_title 

        comments_df = st.session_state.comments_df
        filtered_comments_df = comments_df[comments_df['policy_id'] == policy_id_to_analyze].copy().reset_index(drop=True)
        
        if filtered_comments_df.empty:
            st.info("No comments submitted for this policy yet.")
        else:
            
            # --- Run Core Analysis ---
            comments_list = filtered_comments_df['comment'].tolist()
            df_res_sentiment, relevant_count, irrelevant_count, irrelevant_feedback = process_comments_for_policy(comments_list, policy_topic)
            
            # --- Overall Metrics ---
            st.subheader("📊 Overall Performance Metrics")
            col_met_a, col_met_b = st.columns(2)
            col_met_a.metric("Total Comments Submitted", len(filtered_comments_df))
            col_met_b.metric("Relevant Comments Analyzed", relevant_count, delta=f"-{irrelevant_count} Irrelevant", delta_color="inverse")

            st.markdown("---")

            if df_res_sentiment.empty:
                st.warning("No relevant comments found after filtering.")
            else:
                # Merge demographic data
                df_res = filtered_comments_df.merge(
                    df_res_sentiment, 
                    left_on='comment', 
                    right_on='Comment', 
                    how='inner'
                ).drop(columns=['Comment'])
                
                # --- Policy Narrative Summary ---
                st.subheader("📝 Policy Narrative Summary (AI-Generated)")
                relevant_comments_list = df_res['comment'].tolist()
                paragraph_summary = generate_paragraph_summary(relevant_comments_list, policy_topic)
                st.info(paragraph_summary)

                st.markdown("---")

                ## --- Sentiment Overview ---
                st.subheader("⭐ Sentiment Scoreboard")
                
                overall_sentiment_score = df_res['Score'].mean()
                overall_label = "Highly Positive" if overall_sentiment_score > 0.3 else "Positive" if overall_sentiment_score > 0.1 else "Neutral/Mixed" if overall_sentiment_score > -0.1 else "Negative"

                sentiment_counts_df = df_res['Sentiment'].value_counts().reset_index()
                sentiment_counts_df.columns = ['Sentiment', 'Count']
                
                pos = sentiment_counts_df[sentiment_counts_df['Sentiment'] == 'Positive']['Count'].sum()
                neg = sentiment_counts_df[sentiment_counts_df['Sentiment'] == 'Negative']['Count'].sum()

                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1.5])
                col_a.metric("Overall Sentiment Score", f"{overall_sentiment_score:.3f}", overall_label)
                col_b.metric("Positive Comments", pos)
                col_c.metric("Negative Comments", neg)
                
                with col_d:
                    # Plotly Pie Chart for Sentiment Distribution
                    fig_pie = px.pie(
                        sentiment_counts_df, 
                        values='Count', 
                        names='Sentiment', 
                        title='Sentiment Distribution',
                        color='Sentiment',
                        color_discrete_map=SENTIMENT_COLORS
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=False, paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
                    st.plotly_chart(fig_pie, use_container_width=True)


                st.markdown("---")

                ## --- Sentiment Trend Over Time ---
                st.subheader("📈 Sentiment Trend Over Time")
                df_res['Time'] = pd.to_datetime(df_res['timestamp'], errors='coerce')
                df_res = df_res.sort_values("Time").dropna(subset=['Time']).reset_index(drop=True)
                
                if not df_res.empty:
                    window_size = max(2, len(df_res)//10)
                    df_res['Score_Rolling'] = df_res['Score'].rolling(window=window_size, min_periods=1).mean()
                    
                    fig_line = px.line(
                        df_res,
                        x="Time",
                        y="Score_Rolling",
                        title=f"Sentiment Trend for {policy_topic} (Rolling Average, Window={window_size})",
                        line_shape="spline",
                        color_discrete_sequence=[MCA_COLORS["MCA_BLUE_DARK"]]
                    )
                    fig_line.add_hline(y=0.0, line_dash="dash", line_color="gray")
                    fig_line.update_layout(paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("Insufficient timestamps to plot a reliable sentiment trend over time.")

                st.markdown("---")

                ## --- Thematic Analysis & Word Clouds ---
                st.subheader("📢 Thematic and Keyword Analysis")
                
                # 4 WORD CLOUDS
                generate_sentiment_wordclouds(df_res, policy_topic)
                
                # 5. Bar Chart
                st.markdown("#### Top Keyword Frequencies")
                top_themes = df_res_sentiment.attrs.get('top_themes', [])
                plot_keyword_bar_chart(top_themes, policy_topic)

                st.markdown("---")

                ## --- Demographic Breakdown ---
                st.subheader("🧑‍🤝‍👩 Demographic Breakdown")
                
                col_prof, col_edu = st.columns(2)
                
                if 'profession' in df_res.columns and not df_res['profession'].empty:
                    with col_prof:
                        prof_sentiment = df_res.groupby('profession')['Score'].mean().sort_values(ascending=False).reset_index()
                        fig_prof = px.bar(prof_sentiment, x='profession', y='Score', 
                                          title='Sentiment by Profession', color='Score', 
                                          color_continuous_scale=px.colors.diverging.RdYlGn)
                        fig_prof.update_layout(paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
                        st.plotly_chart(fig_prof, use_container_width=True)
                else:
                    col_prof.warning("Profession data not found or is empty.")

                if 'education' in df_res.columns and not df_res['education'].empty:
                    with col_edu:
                        edu_sentiment = df_res.groupby('education')['Score'].mean().sort_values(ascending=False).reset_index()
                        fig_edu = px.bar(edu_sentiment, x='education', y='Score', 
                                          title='Sentiment by Education Level', color='Score',
                                          color_continuous_scale=px.colors.diverging.RdYlGn)
                        fig_edu.update_layout(paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
                        st.plotly_chart(fig_edu, use_container_width=True)
                else:
                    col_edu.warning("Education data not found or is empty.")
                    
                st.markdown("---")
                
                ## --- Raw Data & Filtered Feedback ---
                st.subheader("📋 Raw Data & Filtered Feedback")
                
                # Prepare CSV for Download
                csv_data = df_res[['timestamp', 'profession', 'education', 'comment', 'Sentiment', 'Score']].to_csv(index=False).encode('utf-8')
                
                col_raw_a, col_raw_b = st.columns([3, 1])
                with col_raw_a:
                    st.markdown("#### Relevant Comments Table")
                with col_raw_b:
                    st.download_button(
                        label="Download Analysis Data as CSV ⬇️",
                        data=csv_data,
                        file_name=f"{policy_topic.replace(' ', '_')}_analysis_{datetime.date.today()}.csv",
                        mime="text/csv",
                        key="internal_download_csv"
                    )

                st.dataframe(df_res[['timestamp', 'profession', 'education', 'comment', 'Sentiment', 'Score']], use_container_width=True)


                if irrelevant_feedback:
                    with st.expander(f"🚫 **View {irrelevant_count} Irrelevant/Filtered Comments**"):
                        df_irrelevant = pd.DataFrame(irrelevant_feedback)
                        st.caption("These comments were excluded from the analysis based on the quality and relevance filters.")
                        st.dataframe(df_irrelevant, use_container_width=True)



## --- TAB 3: EXTERNAL DATA ANALYSIS ---
with tab3:
    st.header("Analyze Unstructured External Data Sources")
    st.markdown("---")
    
    col_mode, col_topic = st.columns([1, 2])
    with col_mode:
        mode = st.radio(
            "Select Input Mode:",
            ["📂 File Upload", "✍ Paste Manually"],
            key="external_mode"
        )
    with col_topic:
        external_topic = st.text_input("Enter Topic for Analysis (e.g., 'Tax Relief', 'School Fees')", key="external_topic")
    
    uploaded_file = None
    text_input = ""
    
    if mode == "📂 File Upload":
        uploaded_file = st.file_uploader("Upload Comments File (CSV or TXT)", type=["csv", "txt"])
    elif mode == "✍ Paste Manually":
        st.info("Enter one comment per line below 👇")
        text_input = st.text_area("Paste your comments here", height=200, placeholder="Type or paste comments here...", key="external_text_area")

    if st.button("🚀 Run External Analysis", type="primary"):
        
        comments_list, timestamps = load_external_comments(uploaded_file, text_input)

        if not external_topic.strip():
            st.error("Please enter a **Policy Topic** to run the analysis.")
        elif not comments_list:
            st.error("Please provide comments via **upload or paste**.")
        else:
            total_count = len(comments_list)
            
            # --- Run Core Analysis ---
            df_res_sentiment, relevant_count, irrelevant_count, irrelevant_feedback = process_comments_for_policy(comments_list, external_topic)
            
            st.markdown("### Results Summary")
            col_met_c, col_met_d = st.columns(2)
            col_met_c.metric("Total Comments Loaded", total_count)
            col_met_d.metric("Relevant Comments Analyzed", relevant_count, delta=f"-{irrelevant_count} Irrelevant", delta_color="inverse")

            st.markdown("---")

            if df_res_sentiment.empty:
                st.warning("No relevant comments found after filtering.")
            else:
                df_res = df_res_sentiment.rename(columns={'Comment': 'comment'}).copy()
                df_res['Time'] = timestamps[:len(df_res)] if timestamps else [None] * len(df_res)
                
                # --- External Narrative Summary ---
                st.subheader("📝 External Data Narrative Summary (AI-Generated)")
                relevant_comments_list = df_res['comment'].tolist()
                paragraph_summary = generate_paragraph_summary(relevant_comments_list, external_topic)
                st.info(paragraph_summary)
                
                st.markdown("---")

                # --- Sentiment Overview & Pie Chart ---
                st.subheader("⭐ Sentiment Scoreboard")

                overall_sentiment_score = df_res['Score'].mean()
                overall_label = "Highly Positive" if overall_sentiment_score > 0.3 else "Positive" if overall_sentiment_score > 0.1 else "Neutral/Mixed" if overall_sentiment_score > -0.1 else "Negative"

                sentiment_counts_df = df_res['Sentiment'].value_counts().reset_index()
                sentiment_counts_df.columns = ['Sentiment', 'Count']

                pos = sentiment_counts_df[sentiment_counts_df['Sentiment'] == 'Positive']['Count'].sum()
                neg = sentiment_counts_df[sentiment_counts_df['Sentiment'] == 'Negative']['Count'].sum()
                
                col_a_ext, col_b_ext, col_c_ext, col_d_ext = st.columns([1, 1, 1, 1.5])
                col_a_ext.metric("Overall Sentiment Score", f"{overall_sentiment_score:.3f}", overall_label)
                col_b_ext.metric("Positive Comments", pos)
                col_c_ext.metric("Negative Comments", neg)
                
                with col_d_ext:
                    # Plotly Pie Chart for Sentiment Distribution
                    fig_pie = px.pie(
                        sentiment_counts_df, 
                        values='Count', 
                        names='Sentiment', 
                        title='Sentiment Distribution',
                        color='Sentiment',
                        color_discrete_map=SENTIMENT_COLORS
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(showlegend=False, paper_bgcolor=MCA_COLORS["MCA_LIGHT_BLUE_BG"])
                    st.plotly_chart(fig_pie, use_container_width=True)


                st.markdown("---")

                # --- Thematic Analysis & Word Clouds ---
                st.subheader("📢 Thematic and Keyword Analysis")
                
                # 4 WORD CLOUDS
                generate_sentiment_wordclouds(df_res, external_topic)
                
                # Bar Chart
                st.markdown("#### Top Keyword Frequencies")
                top_themes = df_res_sentiment.attrs.get('top_themes', [])
                plot_keyword_bar_chart(top_themes, external_topic)
                         
                st.markdown("---")

                # --- Raw Data & Filtered Feedback ---
                st.subheader("📋 Raw Analysis Data & Filtered Feedback")
                
                # Prepare CSV for Download
                csv_data_ext = df_res[['Time', 'comment', 'Sentiment', 'Score']].to_csv(index=False).encode('utf-8')

                col_raw_a_ext, col_raw_b_ext = st.columns([3, 1])
                with col_raw_a_ext:
                    st.markdown("#### Relevant Comments Table")
                with col_raw_b_ext:
                    st.download_button(
                        label="Download Analysis Data as CSV ⬇️",
                        data=csv_data_ext,
                        file_name=f"{external_topic.replace(' ', '_')}_analysis_external_{datetime.date.today()}.csv",
                        mime="text/csv",
                        key="external_download_csv"
                    )

                st.dataframe(df_res[['Time', 'comment', 'Sentiment', 'Score']], use_container_width=True)


                if irrelevant_feedback:
                    with st.expander(f"🚫 **View {irrelevant_count} Irrelevant/Filtered Comments**"):
                        df_irrelevant = pd.DataFrame(irrelevant_feedback)
                        st.caption("These comments were excluded from the analysis based on the quality and relevance filters.")
                        st.dataframe(df_irrelevant, use_container_width=True)