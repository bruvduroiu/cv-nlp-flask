import re

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('spanish')

acceptable_chars = '[\w+#\.]+'

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search(acceptable_chars, token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search(acceptable_chars, token):
            filtered_tokens.append(token)
    return filtered_tokens