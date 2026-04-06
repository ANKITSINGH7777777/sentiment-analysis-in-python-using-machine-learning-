import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(layout="wide", page_title="Sentiment Analysis App")

st.title("Sentiment Analysis App")
st.markdown("Enter any text below to get its sentiment analyzed instantly!")

# --- Interactive Sentiment Analyzer ---
st.header("Analyze Your Text")
user_input = st.text_area("Enter your sentence here:", "This product is fantastic! I'm very happy with my purchase, it exceeded my expectations.", height=150)

# Initialize VADER analyzer once
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    if not text.strip():
        return "Neutral", 0.0, "😐", {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    vs = analyzer.polarity_scores(text)
    compound = vs['compound']
    sentiment_label = "Neutral"
    emoji = "😐"
    if compound >= 0.05:
        sentiment_label = "Positive"
        emoji = "😊"
    elif compound <= -0.05:
        sentiment_label = "Negative"
        emoji = "😠"
    return sentiment_label, compound, emoji, vs

if st.button("Analyze"):
    if user_input:
        sentiment_label, compound_score, emoji, vader_scores = get_vader_sentiment(user_input)

        st.subheader("Analysis Results:")

        # Display sentiment with emoji
        st.markdown(f"**Overall Sentiment:** {emoji} {sentiment_label}")

        # Display confidence/intensity
        st.markdown(f"**Sentiment Intensity (VADER Compound Score):** `{compound_score:.2f}` (ranges from -1.0 to 1.0, where 1.0 is very positive)")

        # Optional: TextBlob analysis (for additional context)
        blob_sentiment = TextBlob(user_input).sentiment
        st.write("**TextBlob Analysis:**")
        st.info(f"Polarity: {blob_sentiment.polarity:.2f} (ranges from -1.0 to 1.0)")
        st.info(f"Subjectivity: {blob_sentiment.subjectivity:.2f} (ranges from 0.0 to 1.0, where 1.0 is very subjective)")

        # --- Add Visualization for VADER Scores ---
        st.write("**VADER Score Distribution:**")
        vader_bar_df = pd.DataFrame({
            'Sentiment Type': ['Positive', 'Neutral', 'Negative'],
            'Score': [vader_scores['pos'], vader_scores['neu'], vader_scores['neg']]
        })

        fig_vader = px.bar(vader_bar_df,
                           x='Sentiment Type',
                           y='Score',
                           color='Sentiment Type',
                           color_discrete_map={'Positive':'green', 'Neutral':'blue', 'Negative':'red'},
                           title='VADER Polarity Sub-Scores',
                           range_y=[0, 1])
        st.plotly_chart(fig_vader, use_container_width=True)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("--- ")
st.markdown("This app uses VADER Sentiment Analysis and TextBlob for text sentiment processing.")
