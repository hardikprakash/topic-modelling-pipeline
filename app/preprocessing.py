import re
import spacy
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

tokenizer = nlp.tokenizer

def preprocess_reviews(review):
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review.lower())
    doc = nlp(review)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])