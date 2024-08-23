# pip install gensim nltk
# 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import strip_punctuation
import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Download NLTK resources (stopwords and punkt tokenizer)
# nltk.download("stopwords")
# nltk.download("punkt")


def preprocess_text(text):
    
    return tokens


class Trainer:
    def __init__(self):
        ...

    def train(self, input_comments, topic_num=14):
        # Preprocess each document
        comments = [preprocess_text(comment) for comment in input_comments]

        dictionary = corpora.Dictionary(comments)   # Create a dictionary from the preprocessed comments
        corpus = [dictionary.doc2bow(comment) for comment in comments]  # Create a bag-of-words representation of the documents
        lda_model = LdaModel(corpus, num_topics=topic_num, id2word=dictionary)  # Build the LDA model

        # topics = lda_model.print_topics(num_words=5)  # Print the topics and their top words
        # for topic in topics:
        #     print(topic)
        
        # Now you can use the model to assign topics to new documents or analyze existing documents.
        topic_ids = []
        for doc in corpus:
            topic_id = max(lda_model[doc], key=lambda item: item[1])[0]
            topic_ids.append(topic_id)

        print(topic_ids)

if __name__ == '__main__':
    trainer = Trainer()

    # Sample list of comments
    input_comments = [
        "Topic modeling is a popular technique in natural language processing.",
        "LDA stands for Latent Dirichlet Allocation.",
        "Texts can be represented as a mixture of topics.",
        "Gensim is a powerful library for topic modeling in Python.",
        "Python is a versatile programming language used for various tasks.",
    ]
    

    trainer.train(input_comments, 2)