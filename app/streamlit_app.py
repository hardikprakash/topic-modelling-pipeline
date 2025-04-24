import torch
torch.classes.__path__ = []

import streamlit as st
import pandas as pd
from preprocessing import preprocess_reviews
from topic_modelling import model_topics
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="Topic Modelling Pipeline", layout="wide")
st.title("Topic Modelling with BERTopic")

st.sidebar.header("1. Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("### CSV will be truncated to 10000 entries to save compute.")    
    if len(df) > 10000:
        df = df.sample(10000).reset_index(drop=True)

    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    st.sidebar.header("2. Column Selection")
    columns = df.columns.tolist()

    review_column = st.sidebar.selectbox("Select the review column (text)", columns)
    
    groupby_column = st.sidebar.selectbox(
        "Optional: Select a column to group by", 
        ["None"] + [col for col in columns if col != review_column]
    )

    if st.sidebar.button("Run Topic Modeling"):
        st.info("Preprocessing text...")
        df["Cleaned Reviews"] = df[review_column].astype(str).apply(preprocess_reviews)
        
        with st.spinner("Generating topic models..."):
            if groupby_column == "None":
                topic_models = model_topics(df, groupby_column=None)
            else:
                topic_models = model_topics(df, groupby_column=groupby_column)
        
        st.success("Topic modeling complete!")

        has_topics = False
        for group in topic_models:
            topic_info = topic_models[group]["model"].get_topic_info()
            valid_topics = topic_info[topic_info.Topic != -1]
            if len(valid_topics) > 0:
                has_topics = True
                break
                
        if not has_topics:
            st.warning("⚠️ No topics could be generated from the provided data. Try a different dataset with more text content or select a different text column.")
            st.stop()

        if groupby_column == "None":
            st.header(f"📊 Results for All Data")
        
        else:
            st.header(f"📊 Results Grouped by {groupby_column}")
            # Navigation selectbox.
            if len(topic_models) > 1:
                group_selector = st.selectbox(
                    "Jump to group:", 
                    options=list(topic_models.keys())
                )
                st.markdown(f"[Jump to {group_selector}](#{group_selector.replace(' ', '-').lower()})")

        for group in topic_models:
            
            st.markdown(f"<div id='{group.replace(' ', '-').lower()}'></div>", unsafe_allow_html=True)
            
            if groupby_column == "None":
                st.write(f"## Results")
            else:
                st.write(f"## 📊 {group}")
            
            st.metric("Coherence Score", f"{topic_models[group]['coherence']:.4f}")
            
            topic_info = topic_models[group]["model"].get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]
            
            if len(topic_info) == 0:
                st.warning(f"No topics could be generated for {group}.")
                continue
            
            st.write("### Top Topics")
            st.dataframe(topic_info[["Topic", "Name", "Count"]].reset_index(drop=True))
            
            # Only display visualizations if we have topics
            if len(topic_info) > 0:
                # Bar Chart
                top_topics = topic_info.sort_values(by="Count", ascending=False).head(5)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=top_topics, x="Count", y="Name", palette="viridis", ax=ax)
                ax.set_title(f"Top 5 Topics for {group}")
                ax.set_xlabel("Number of Documents")
                ax.set_ylabel("Topic")
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Explore dropdown
                with st.expander("Explore Topic Details"):
                    for _, row in topic_info.iterrows():
                        st.write(f"**Topic {row['Topic']:>2}:** {row['Name']} (Count: {row['Count']})")
            
            st.divider()
