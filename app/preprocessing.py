import re
import spacy

nlp = spacy.load('en_core_web_sm')
tokenizer = nlp.tokenizer

def preprocess_reviews(review):
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review.lower())
    doc = nlp(review)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])