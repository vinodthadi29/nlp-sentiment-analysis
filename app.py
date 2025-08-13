import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Load Model & Dataset ---
loaded_model = joblib.load('sentiment_model.pkl')
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')
df.columns = [x.lower() for x in df.columns]  # Standardize column names

# --- Model Accuracy ---
X = df['review']
y = df['liked']
y_pred = loaded_model.predict(X)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

# --- AWS-Optimized Page Config ---
st.set_page_config(
    page_title="Restaurant Review Sentiment Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea textarea {
        font-size: 16px !important;
        color: #000000 !important; /* Black Text for Contrast */
        background-color: #FAFAFA !important;
    }
    .stButton>button {
        background-color: #ff6b6b;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ff3b3b;
    }
    .metric-box {
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; color: #FF5733;'>🍽️ Restaurant Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.write("Enter your restaurant review below and let our AI analyze the sentiment!")

# --- Sidebar: Model Performance Metrics ---
st.sidebar.header("📊 Model Performance Metrics")
st.sidebar.metric("🔍 Accuracy", f"{accuracy:.2%}")

fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)
ax.set_title("Confusion Matrix")
st.sidebar.pyplot(fig)

# --- Sentiment Distribution Chart ---
st.sidebar.header("📝 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(5, 3))
sns.countplot(x='liked', data=df, palette='coolwarm', ax=ax)
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_title("Distribution of Reviews")
st.sidebar.pyplot(fig)

# --- User Input Section ---
review_input = st.text_area("✍️ **Your Restaurant Review:**", height=150, placeholder="Type your review here...")

# --- Sentiment Prediction Function ---
def predict_sentiment(review):
    if review:
        prediction = loaded_model.predict([review])
        return prediction[0]
    return None

# --- Analyze Button ---
if st.button("🔍 Analyze Review"):
    sentiment_result = predict_sentiment(review_input)

    if sentiment_result is not None:
        st.markdown("---")
        st.subheader("🔎 **Sentiment Analysis Result:**")

        if sentiment_result == 1:
            st.success("✅ This is a **positive** review! 😊 We appreciate your kind words.")
        else:
            st.error("❌ This is a **negative** review. 😔 We're sorry to hear about your experience.")

        # --- Restaurant Recommendations ---
        st.markdown("---")
        st.subheader("🍽️ **Recommended Restaurants**")

        if sentiment_result == 1:
            st.write("🎉 Based on your **positive** feedback, here are some highly-rated restaurants you might enjoy:")
            st.info("✅ The Gourmet House\n✅ Italian Delight\n✅ Urban Bistro")
        else:
            st.write("🔄 Since you didn't enjoy your experience, here are some alternative restaurants to try:")
            st.info("🔄 Fresh Bites\n🔄 Comfort Cuisine\n🔄 Tasty Treats")

    else:
        st.warning("⚠️ Please enter a review to analyze.")

# --- Footer ---
st.markdown("---")
st.markdown("<h4 style='text-align: center;'>📧 Contact us: vinodthadi29@gmail.com | 📞 9502738939</h4>", unsafe_allow_html=True)
