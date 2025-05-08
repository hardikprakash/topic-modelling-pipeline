# Topic Modelling Pipeline

A streamlined application for extracting insights from airline customer reviews through advanced topic modelling techniques.

**Hosted at:** https://topic-modelling-pipeline.streamlit.app

## Overview

This project implements a natural language processing pipeline that identifies key themes and topics in text data using BERTopic, an advanced topic modeling algorithm. The web application provides an intuitive interface for uploading review data and generating insights that reveal underlying patterns and themes.

## Features

- **Interactive Web Interface**: User-friendly Streamlit application for data upload and exploration.
- **Advanced Text Preprocessing**: Specialized NLP pipeline using spaCy for text cleaning and normalization.
- **Transformer-based Topic Modeling**: Utilizes BERTopic with sentence transformers for state-of-the-art results.
- **Coherence Measurement**: Evaluates topic quality through coherence score metrics.
- **Group-based Analysis**: Compare topics across different categories (airlines, travel classes, etc.).
- **Interactive Visualizations**: Clear presentation of topic distributions and key themes.

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd topic-modelling-pipeline
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the spaCy language model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

## Usage Guide

1. **Upload Data**: Select a CSV file containing review data.
2. **Select Columns**: Choose which column contains the review text.
3. **Optional Grouping**: Select a column to group results by (e.g., airline, class).
4. **Run Analysis**: Click "Run Topic Modeling" to process the data.
5. **Explore Results**: Navigate through the generated topics and their distributions.

## Project Structure

- **Preprocessing**:
  - Converts text to lowercase.
  - Removes special characters.
  - Tokenizes text using spaCy.
  - Removes stop words.
  - Applies lemmatization.

- **Topic Modelling**:
  - Uses Sentence Transformers to create document embeddings.
  - Applies BERTopic to identify coherent topics.
  - Calculates coherence scores to evaluate quality.
  - Presents topics with their representative terms.

## Requirements

- `streamlit`
- `pandas`
- `matplotlib`
- `seaborn`
- `bertopic`
- `sentence-transformers`
- `spacy`
- `gensim`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [BERTopic](https://github.com/MaartenGr/BERTopic) for the topic modeling framework.
- [Sentence Transformers](https://www.sbert.net/) for document embeddings.
- [spaCy](https://spacy.io/) for NLP preprocessing.
