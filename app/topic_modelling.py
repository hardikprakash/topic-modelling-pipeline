from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess

def compute_gensim_coherence(tokenized_docs, topic_model, top_n_words=10, coherence='c_v'):

    dictionary = Dictionary(tokenized_docs)

    topic_words_dict = topic_model.get_topics()
    topics = [
        [word for word, _ in topic_words_dict[topic_id][:top_n_words]]
        for topic_id in topic_words_dict
        if topic_id != -1 and isinstance(topic_words_dict[topic_id], list)
    ]

    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=coherence
    )
    return coherence_model.get_coherence()

def model_topics(df, groupby_column):
    topic_models = {}

    if groupby_column is not None:
        for groupname, group in df.groupby(groupby_column):
            docs = group["Cleaned Reviews"].dropna().tolist()

            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
            topics, _ = topic_model.fit_transform(docs)

            topic_models[groupname] = {
                    "model": topic_model,
                    "topics": topics,
                    "docs": docs
            }

            tokenized_docs = [simple_preprocess(doc) for doc in docs]

            coherence = compute_gensim_coherence(tokenized_docs, topic_model)
            topic_models[groupname]["coherence"] = coherence

        return topic_models  

    else:
        docs = df['Cleaned Reviews'].dropna().tolist()
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
        topics, _ = topic_model.fit_transform(docs)

        topic_models['Default Group'] = {
                "model": topic_model,
                "topics": topics,
                "docs": docs
        }

        tokenized_docs = [simple_preprocess(doc) for doc in docs]

        coherence = compute_gensim_coherence(tokenized_docs, topic_model)
        topic_models['Default Group']["coherence"] = coherence

        return topic_models
