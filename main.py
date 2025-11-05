# ===========================
# AI Comment Analysis API - Full Feature Backend (V4.3.2)
# ===========================
#
# To run this server:
# 1. Save this file as main.py
#
# 2. Install requirements:
#    pip install fastapi uvicorn spacy pandas vaderSentiment
#    pip install matplotlib wordcloud scikit-learn numpy python-multipart
#
# 3. Download spaCy model:
#    python -m spacy download en_core_web_md
#
# 4. Run the server:
#    uvicorn main:app --reload
#
# 5. Access the interactive API docs (Swagger):
#    http://127.0.0.1:8000/docs
#
# ===========================

## --- 1. Imports ---

# --- Standard Library ---
import os
import subprocess
import sys
import re
import io
import base64
import csv
from collections import Counter
from typing import List, Optional, Dict, Tuple # Correct and centralized import for type hints
from functools import lru_cache # For API Caching

# --- Third-Party Libraries ---
import matplotlib
matplotlib.use("Agg") # Use 'Agg' backend for non-GUI server environment
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
import pandas as pd
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    import spacy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
    import spacy

## --- 2. Global Initialization & Model Loading ---

try:
    # Load the medium-sized spaCy model for NER and similarity checks
    NLP_MODEL = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model 'en_core_web_md'...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
    NLP_MODEL = spacy.load("en_core_web_md")

# Initialize models and constants once on startup
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
VADER_LEXICON = SENTIMENT_ANALYZER.lexicon
STOPWORDS_SET = set(WordCloud().stopwords)
SENTIMENT_COLORS = {
    "Positive": "#2ca02c", # Green
    "Negative": "#d62728", # Red
    "Neutral": "#8c8c8c" ,  # Gray
}

## --- 3. API Setup ---

app = FastAPI(
    title="AI Comment Analysis API",
    description="A comprehensive API for comment analysis, providing summaries, charts, and downloadable reports.",
    version="4.3.2" # Fixed version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## --- 4. Chart & Image Generation Functions ---

def generate_base64_image(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure to a Base64 encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def generate_pie_chart_b64(sentiment_counts: Dict[str, int]) -> str:
    """Generates a Base64 sentiment pie chart."""
    labels = list(sentiment_counts.keys())
    sizes = list(sentiment_counts.values())
    colors = [SENTIMENT_COLORS[label] for label in labels]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white'}, textprops={'color': 'black'}
    )
    ax.axis('equal')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return generate_base64_image(fig)

def generate_bar_chart_b64(top_keywords: List[Tuple[str, int]]) -> str:
    """Generates a Base64 keyword horizontal bar chart."""
    if not top_keywords: return ""
    words = [w for w, c in top_keywords]; counts = [c for w, c in top_keywords]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.barh(words, counts, color='#1f77b4')
    ax.invert_yaxis(); ax.set_xlabel('Frequency')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    return generate_base64_image(fig)

def generate_line_chart_b64(df_time: pd.DataFrame) -> Optional[str]:
    """Generates a Base64 sentiment-over-time line chart."""
    if df_time.empty or len(df_time) < 2: return None
    df_trend = df_time.set_index("Time")['Score'].resample('D').mean().dropna()
    if len(df_trend) < 2: return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_trend.index, df_trend.values, marker='o', linestyle='-', color='blue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Average Sentiment Score'); ax.set_xlabel('Date')
    fig.autofmt_xdate()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    return generate_base64_image(fig)

def generate_wordcloud_b64(text: str) -> Optional[str]:
    """Generates a Base64 word cloud image."""
    if not text.strip(): return None
    try:
        wc = WordCloud(
            width=600, height=300, background_color=None,
            mode="RGBA", max_words=100, collocations=False
        ).generate(text)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        return generate_base64_image(fig)
    except Exception:
        plt.close('all'); return None

## --- 5. NLP & Content Generation Functions ---

def expand_keywords(topic: str) -> List[str]:
    """Expands a single topic keyword into variations (e.g., 'tax' -> ['tax', 'taxes', 'taxing'])."""
    topic_low = topic.lower()
    return [topic_low, topic_low + "s", topic_low + "ing"]

def filter_comment_with_reason(
    comment: str,
    topic_doc: spacy.tokens.doc.Doc,
    topic_keywords: List[str],
    seen_comments: set
# FIX APPLIED: Corrected return type hint to use Tuple[...]
) -> Tuple[bool, str]: # type: ignore
    """
    Filters a single comment based on relevance and quality.
    """
    comment_lower = comment.strip().lower()
    
    if not comment_lower: return False, "Comment removed: insufficient detail."
    if not any(c.isalnum() for c in comment_lower): return False, "Comment filtered: contains only emojis/symbols."
    if len(comment_lower.split()) < 3: return False, "Comment removed: insufficient detail (under 3 words)."
    if comment_lower in seen_comments: return False, "Comment filtered: duplicate entry."
    inappropriate = ["abuse", "spam", "fake", "nonsense", "http:", "https:"]
    if any(bad in comment_lower for bad in inappropriate): return False, "Comment filtered: contains inappropriate language or links."
        
    if any(re.search(rf'\b{re.escape(kw)}\b', comment_lower) for kw in topic_keywords):
        seen_comments.add(comment_lower); return True, ""
        
    comment_doc = NLP_MODEL(comment_lower)
    if not comment_doc.has_vector or not comment_doc.vector_norm:
               return False, f"Comment filtered: could not be understood."

    sim = topic_doc.similarity(comment_doc)
    if sim < 0.4:
        return False, f"Comment filtered: not relevant to the topic '{topic_doc.text}' (Similarity: {sim:.2f})."
        
    seen_comments.add(comment_lower)
    return True, ""

def analyze_sentiment(text: str) -> Tuple[float, str]:
# FIX APPLIED: Corrected return type hint to use Tuple[...]
    """Analyzes text and returns a compound score (-1.0 to 1.0) and a string label."""
    score = SENTIMENT_ANALYZER.polarity_scores(text)["compound"]
    if score >= 0.05: return score, "Positive"
    elif score <= -0.05: return score, "Negative"
    else: return score, "Neutral"

def get_sentiment_tooltip(comment: str, sentiment: str, score: float) -> str:
    """Generates a tooltip summary for a comment by finding its most impactful words."""
    words = re.findall(r'\b\w+\b', comment.lower())
    sentiment_words = []
    for word in words:
        if word in VADER_LEXICON:
            sentiment_words.append((word, VADER_LEXICON[word]))
    if not sentiment_words:
        return f"Classified as {sentiment} (Score: {score:.2f}). No specific keywords found."
    sentiment_words.sort(key=lambda x: x[1])
    tooltip = f"Classified as {sentiment} (Score: {score:.2f}). "
    if sentiment == "Positive" and sentiment_words:
        pos_words = [f"'{w}' ({s})" for w, s in reversed(sentiment_words) if s > 0][:2]
        if pos_words: tooltip += f"Key positive words: {', '.join(pos_words)}."
    elif sentiment == "Negative" and sentiment_words:
        neg_words = [f"'{w}' ({s})" for w, s in sentiment_words if s < 0][:2]
        if neg_words: tooltip += f"Key negative words: {', '.join(neg_words)}."
    return tooltip

def extract_entities(comments: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Extracts and counts Named Entities (People, Orgs, Locations) from a list of comments."""
    entity_counts = {"PERSON": Counter(), "ORG": Counter(), "GPE": Counter()}
    for doc in NLP_MODEL.pipe(comments):
        for ent in doc.ents:
            if ent.label_ in entity_counts:
                entity_counts[ent.label_][ent.text.strip()] += 1
    entities_json = {}
    for label, counts in entity_counts.items():
        entities_json[label] = counts.most_common(10)
    return entities_json

def generate_summaries_and_top_content(
    comments: List[str], topic: str, sentiment_counts: Dict[str, int]
# FIX APPLIED: Corrected return type hint to use Tuple[...]
) -> Tuple[str, str, List[Dict], List[Tuple[str, int]]]:
    """Generates the brief/detailed summaries, top repeated comments (with tooltips), and top keywords."""
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
    all_words = " ".join(comments).lower().split()
    topic_words = set(topic.lower().split())
    keywords_filtered = [w for w in all_words if w not in STOPWORDS_SET and w not in topic_words and len(w) > 2 and w.isalpha()]
    top_keywords = Counter(keywords_filtered).most_common(5)
    keyword_str = ", ".join([w for w, c in top_keywords]) or "various themes"
    top_comments_list = []
    for comment, freq in Counter(comments).most_common(10):
        score, sentiment = analyze_sentiment(comment)
        tooltip = get_sentiment_tooltip(comment, sentiment, score)
        top_comments_list.append({
            "comment": comment, "frequency": freq, "sentiment": sentiment,
            "color": SENTIMENT_COLORS[sentiment], "tooltip": tooltip
        })
    brief_summary = (f"Analysis of {total} comments on **{topic}** shows a **{overall_sent_str}** public opinion. Key discussion topics include **{keyword_str}**.")
    detailed_summary = (f"A detailed review of **{total}** relevant comments about **{topic}** reveals a **{overall_sent_str}** sentiment. Positive feedback (making up **{sentiment_counts.get('Positive', 0)}** comments) generally focuses on benefits and approval. Negative remarks (**{sentiment_counts.get('Negative', 0)}** comments) often highlight concerns, dissatisfaction, or areas for improvement. The most frequent themes identified, aside from the main topic, are **{keyword_str}**. The top repeated comments (shown below) capture the most common public opinions and recurring questions.")
    return brief_summary, detailed_summary, top_comments_list, top_keywords

## --- 6. Pydantic Models (API Data Structure) ---

class PasteRequest(BaseModel):
    topic: str = Field(..., example="Public Transport")
    comments: str = Field(..., example="The new bus line is great!\nToo expensive.")

class FilteredFeedback(BaseModel):
    comment: str; reason: str

class TopComment(BaseModel):
    comment: str; frequency: int; sentiment: str
    color: str = Field(..., example="#d62728")
    tooltip: str = Field(..., description="A summary explaining the 'why' behind the comment's sentiment.")

class ChartPayload(BaseModel):
    pie_chart_b64: str = Field(..., description="Base64 encoded PNG for the sentiment pie chart.")
    keyword_bar_chart_b64: str = Field(..., description="Base64 encoded PNG for the top keywords bar chart.")

class EntityPayload(BaseModel):
    PERSON: List[Tuple[str, int]] = Field(..., description="Top 10 people mentioned, [text, count]")
    ORG: List[Tuple[str, int]] = Field(..., description="Top 10 organizations, [text, count]")
    GPE: List[Tuple[str, int]] = Field(..., description="Top 10 locations, [text, count]")

class TooltipPayload(BaseModel):
    brief_summary: str = "A one-paragraph overview of the analysis."
    detailed_summary: str = "An in-depth look at the sentiment balance and key themes."
    top_comments: str = "The most frequently repeated comments. Hover over a specific comment to see a summary of why it was classified with its sentiment."
    named_entities: str = "Proper nouns (People, Organizations, Locations) automatically extracted from the comments."
    charts: str = "Visual breakdown of sentiment (Pie) and top keywords (Bar)."
    sentiment_trend: str = "Shows the average sentiment score (from -1.0 to +1.0) aggregated per day."
    filter_stats: str = "A high-level count of how many comments were loaded vs. how many were filtered out as irrelevant or spam."
    filtered_feedback: str = "Comments that were removed because they were duplicates, irrelevant, or spam. Hover over a comment to see the specific reason."
    word_clouds: str = "A visual representation of the most common words. The larger the word, the more often it appeared."

class AnalysisResponse(BaseModel):
    brief_summary: str
    detailed_summary: str
    top_comments: List[TopComment]
    named_entities: EntityPayload
    charts: ChartPayload
    sentiment_trend_chart_b64: Optional[str]
    total_comments: int
    filtered_comments_count: int
    filtered_feedback: List[FilteredFeedback]
    wordcloud_positive_b64: Optional[str]
    wordcloud_negative_b64: Optional[str]
    wordcloud_neutral_b64: Optional[str]
    wordcloud_combined_b64: Optional[str]
    tooltips: TooltipPayload
    download_csv_endpoint: str = Field(..., description="The single endpoint for the frontend to call to download the CSV.")
    
    class Config:
        fields = {
            'brief_summary': ..., 'detailed_summary': ..., 'top_comments': ...,
            'named_entities': ..., 'charts': ..., 'sentiment_trend_chart_b64': ...,
            'total_comments': ..., 'filtered_comments_count': ..., 'filtered_feedback': ...,
            'wordcloud_positive_b64': ..., 'wordcloud_negative_b64': ...,
            'wordcloud_neutral_b64': ..., 'wordcloud_combined_b64': ...,
            'tooltips': ..., 'download_csv_endpoint': ...
        }

## --- 7. Core Analysis Logic & Caching ---

@lru_cache(maxsize=32)
def _run_core_analysis(
    topic: str, comments_tuple: Tuple[str, ...], timestamps_tuple: Tuple[Optional[str], ...]
) -> Tuple[AnalysisResponse, pd.DataFrame]:
    """This is the core, synchronous, cached function."""
    comments_list = list(comments_tuple)
    timestamps_list = list(timestamps_tuple) if timestamps_tuple else None
    if not topic.strip(): raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    if not comments_list: raise HTTPException(status_code=400, detail="Comments list cannot be empty.")
    topic_doc = NLP_MODEL(topic.lower()); topic_keywords = expand_keywords(topic)
    filtered_comments_list: List[FilteredFeedback] = []
    relevant_comments_map = {}; seen_comments = set()
    for idx, comment in enumerate(comments_list):
        is_ok, reason = filter_comment_with_reason(comment, topic_doc, topic_keywords, seen_comments)
        if is_ok: relevant_comments_map[idx] = comment
        else: filtered_comments_list.append(FilteredFeedback(comment=comment, reason=reason))
    if not relevant_comments_map:
        raise HTTPException(status_code=404, detail=f"No relevant comments found for the topic '{topic}' after filtering.")
    relevant_indices = list(relevant_comments_map.keys())
    relevant_comments = list(relevant_comments_map.values())
    analysis_results = []
    for idx, comment in zip(relevant_indices, relevant_comments):
        score, sentiment = analyze_sentiment(comment)
        ts = timestamps_list[idx] if timestamps_list and idx < len(timestamps_list) else None
        analysis_results.append({"Comment": comment, "Sentiment": sentiment, "Score": score, "Time": ts})
    df_results = pd.DataFrame(analysis_results)
    sentiment_counts = df_results["Sentiment"].value_counts().to_dict()
    df_time = pd.DataFrame()
    if "Time" in df_results.columns and not df_results["Time"].isnull().all():
        df_time = df_results.copy(); df_time["Time"] = pd.to_datetime(df_time["Time"], errors="coerce")
        df_time = df_time.dropna(subset=["Time"])
    brief_summary, detailed_summary, top_comments, top_keywords = \
        generate_summaries_and_top_content(relevant_comments, topic, sentiment_counts)
    entities_payload = extract_entities(relevant_comments)
    pie_chart_b64 = generate_pie_chart_b64(sentiment_counts)
    bar_chart_b64 = generate_bar_chart_b64(top_keywords)
    line_chart_b64 = generate_line_chart_b64(df_time)
    pos_text = " ".join(df_results[df_results["Sentiment"] == "Positive"]["Comment"])
    neg_text = " ".join(df_results[df_results["Sentiment"] == "Negative"]["Comment"])
    neu_text = " ".join(df_results[df_results["Sentiment"] == "Neutral"]["Comment"])
    all_text = " ".join(df_results["Comment"])
    response = AnalysisResponse(
        brief_summary=brief_summary, detailed_summary=detailed_summary, top_comments=top_comments,
        named_entities=entities_payload,
        charts=ChartPayload(pie_chart_b64=pie_chart_b64, keyword_bar_chart_b64=bar_chart_b64),
        sentiment_trend_chart_b64=line_chart_b64, total_comments=len(comments_list),
        filtered_comments_count=len(filtered_comments_list), filtered_feedback=filtered_comments_list,
        wordcloud_positive_b64=generate_wordcloud_b64(pos_text),
        wordcloud_negative_b64=generate_wordcloud_b64(neg_text),
        wordcloud_neutral_b64=generate_wordcloud_b64(neu_text),
        wordcloud_combined_b64=generate_wordcloud_b64(all_text),
        tooltips=TooltipPayload(filter_stats=f"The system loaded {len(comments_list)} comments and filtered out {len(filtered_comments_list)} as irrelevant or spam, leaving {len(relevant_comments)} for analysis."),
        download_csv_endpoint="PLACEHOLDER"
    )
    return response, df_results

## --- 8. API Endpoints ---

def parse_csv_data(decoded_content: str) -> Tuple[List[str], List[Optional[str]]]:
    """
    Parses CSV content, automatically detecting if a header is present.
    """
    comments_list = []
    timestamps_list = []
    try:
        csv_reader = csv.reader(io.StringIO(decoded_content))
        first_row = next(csv_reader)
        first_row_lower = [h.lower().strip() for h in first_row]
    except StopIteration:
        return [], []

    header_keywords = ['comment', 'text', 'timestamp', 'date']
    is_header = any(keyword in first_row_lower for keyword in header_keywords)
    comment_col = 0
    time_col = -1
    
    if is_header:
        header = first_row_lower
        if 'comment' in header: comment_col = header.index('comment')
        elif 'text' in header: comment_col = header.index('text')
        if 'timestamp' in header: time_col = header.index('timestamp')
        elif 'date' in header: time_col = header.index('date')
        for row in csv_reader:
            if len(row) > comment_col:
                comments_list.append(row[comment_col])
                timestamps_list.append(row[time_col] if time_col != -1 and len(row) > time_col else None)
    else:
        comment_col = 0
        if len(first_row) > 1: time_col = 1
        if len(first_row) > comment_col:
            comments_list.append(first_row[comment_col])
            timestamps_list.append(first_row[time_col] if time_col != -1 and len(first_row) > time_col else None)
        for row in csv_reader:
            if len(row) > comment_col:
                comments_list.append(row[comment_col])
                timestamps_list.append(row[time_col] if time_col != -1 and len(row) > time_col else None)
                
    return comments_list, timestamps_list

@app.post("/analyze/paste/", response_model=AnalysisResponse, summary="Analyze comments from a pasted text block")
async def analyze_pasted_text(request: PasteRequest):
    """Analyzes comments provided in a single multiline text block."""
    comments_list = [c.strip() for c in request.comments.splitlines() if c.strip()]
    comments_tuple = tuple(comments_list); timestamps_tuple = tuple()
    try:
        response, _ = _run_core_analysis(request.topic, comments_tuple, timestamps_tuple)
        response.download_csv_endpoint = "/download/paste/"
        return response
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/analyze/upload/", response_model=AnalysisResponse, summary="Analyze comments from an uploaded CSV file")
async def analyze_csv_upload(
    topic: str = Form(..., description="The topic to analyze against."),
    file: UploadFile = File(..., description="A CSV file with comments.")
):
    """
    Analyzes comments from a CSV file, with or without headers.
    (Includes encoding fix for common errors like '0x92').
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    try:
        contents = await file.read()
        
        # --- ENCODING FIX IMPLEMENTATION ---
        try:
            # Try standard UTF-8 first
            decoded_content = contents.decode("utf-8") 
        except UnicodeDecodeError:
            # Fall back to the common Windows encoding (CP1252)
            decoded_content = contents.decode("cp1252") 
        # --- END FIX ---
            
        comments_list, timestamps_list = parse_csv_data(decoded_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading or parsing CSV file: {e}")
    if not comments_list:
        raise HTTPException(status_code=400, detail="No comments found in CSV file.")
    comments_tuple = tuple(comments_list); timestamps_tuple = tuple(timestamps_list)
    try:
        response, _ = _run_core_analysis(topic, comments_tuple, timestamps_tuple)
        response.download_csv_endpoint = "/download/upload/"
        return response
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# --- Download Endpoints (Updated) ---

def create_csv_response(df: pd.DataFrame, topic: str) -> StreamingResponse:
    """Utility to convert a DataFrame to a CSV StreamingResponse."""
    stream = io.StringIO(); df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    filename = f"analysis_{topic.replace(' ', '_')}.csv"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response

@app.post("/download/paste/", summary="Download analysis from pasted text as CSV")
async def download_pasted_text_csv(request: PasteRequest):
    """Re-runs analysis (or pulls from cache) and returns a CSV file."""
    comments_list = [c.strip() for c in request.comments.splitlines() if c.strip()]
    comments_tuple = tuple(comments_list); timestamps_tuple = tuple()
    try:
        _, df_results = _run_core_analysis(request.topic, comments_tuple, timestamps_tuple)
        return create_csv_response(df_results, request.topic)
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/download/upload/", summary="Download analysis from uploaded CSV as CSV")
async def download_csv_upload_csv(
    topic: str = Form(..., description="The topic to analyze against."),
    file: UploadFile = File(..., description="The same CSV file originally uploaded.")
):
    """Re-runs analysis (or pulls from cache) and returns a CSV file."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")
    try:
        contents = await file.read()
        
        # --- ENCODING FIX IMPLEMENTATION ---
        try:
            decoded_content = contents.decode("utf-8") 
        except UnicodeDecodeError:
            decoded_content = contents.decode("cp1252") 
        # --- END FIX ---
            
        comments_list, timestamps_list = parse_csv_data(decoded_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading or parsing CSV file: {e}")
    if not comments_list:
        raise HTTPException(status_code=400, detail="No comments found in CSV file.")
    comments_tuple = tuple(comments_list); timestamps_tuple = tuple(timestamps_list)
    try:
        _, df_results = _run_core_analysis(topic, comments_tuple, timestamps_tuple)
        return create_csv_response(df_results, topic)
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

## --- 9. Server Entry Point ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)