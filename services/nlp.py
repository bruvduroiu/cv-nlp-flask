import os
import re

import numpy as np
import pandas as pd
import nltk
from sklearn import feature_selection
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib

from util.tokenize import (
    tokenize_and_stem,
    tokenize_only
)

TFIDF_PICKLE_PATH = 'model/tfidf.pkl'
GMM_PICKLE_PATH = 'model/gmm.pkl'

class WordCluster:
    def __init__(self, n_clusters=5, dataset_path='data/data.csv', retrain=False, json=False):
        self.n_clusters = n_clusters
        self.dataset_path = dataset_path
        if json:
            self.data = pd.read_json(dataset_path)
        else:
            self.data = pd.read_csv(dataset_path, encoding='utf-8')
        self.X = self._preprocess_data(self.data)

        if retrain or not os.path.exists(TFIDF_PICKLE_PATH) or not os.path.exists(GMM_PICKLE_PATH):
            self.train(self.X)
        else:
            self.load_from_pickle()

    def _preprocess_data(self, data, document_column='requirements'):
        return data[document_column]

    def load_from_pickle(self):
        self.gmm = joblib.load(GMM_PICKLE_PATH)
        self.tfidf_vectorizer = joblib.load(TFIDF_PICKLE_PATH)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.X)
        clusters = self.gmm.predict(self.tfidf_matrix.toarray())
        self.data = self.data.assign(clusters=clusters)

    def get_topic_json(self, X):
        if not X:
            X = self.X
        vectorized_doc = self.tfidf_vectorizer.transform(X)
        predictions = self.gmm.predict_proba(vectorized_doc.toarray())
        top_predictions = predictions.argsort()[0][::-1][:self.n_clusters]

        topic_jobs = {}
        for i, cluster in enumerate(top_predictions):
            doc_idx = self.data[self.data['clusters']==cluster].index
            job_idx = list(cosine_similarity(vectorized_doc, self.tfidf_matrix[doc_idx]).argsort()[0][::-1][:5])
            job_ids = self.data[self.data['clusters']==cluster].id.values
            job_ids = job_ids[job_idx]

            jobs = {}
            for j, job_id in enumerate(job_ids):
                jobs[j] = self.data[self.data.id==job_id].to_dict('records')
            topic_jobs['Topic {}'.format(i)] = jobs 

        return topic_jobs

    def train(self, X):
        # X = X
        # self.data = self.data
        stopwords = nltk.corpus.stopwords.words('english')

        totalvocab_stemmed = []
        totalvocab_tokenized = []

        for doc in X:
            all_words_stemmed = tokenize_and_stem(doc)
            totalvocab_stemmed.extend(all_words_stemmed)
            
            all_words_tokenized = tokenize_only(doc)
            totalvocab_tokenized.extend(all_words_tokenized)

        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.7, 
                                                max_features=200000,
                                                min_df=0.2, 
                                                stop_words='english',
                                                use_idf=True, 
                                                tokenizer=tokenize_and_stem, 
                                                ngram_range=(1,3))

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(X)
        joblib.dump(self.tfidf_vectorizer, TFIDF_PICKLE_PATH)

        self.gmm = GaussianMixture(n_components=self.n_clusters)
        self.gmm.fit(self.tfidf_matrix.toarray())
        joblib.dump(self.gmm, GMM_PICKLE_PATH)

        clusters = self.gmm.predict(self.tfidf_matrix.toarray())

        self.data = self.data.assign(clusters=clusters)

