#!/usr/bin/env python

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC

#nltk.download('punkt')

df = pd.read_csv('tweets_public_es.csv', index_col='tweet_id', encoding='utf-16')
test_size=0.25
train, test = train_test_split(df, test_size=test_size)

y_train = train['airline_sentiment'].values
y_test= test['airline_sentiment'].values

# define training, test data. tokeninze text column of the dataframe
train["tokenized_sents"] = train["text"].apply(nltk.word_tokenize)
test["tokenized_sents"] = test["text"].apply(nltk.word_tokenize)


## build vocabulary. word embbeding
#min_count --> number of times a word should appear
model = Word2Vec(train['tokenized_sents'], min_count=10, workers=4 , size=100)

# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)

# get the most common words
print(model.wv.index2word[0], model.wv.index2word[1], model.wv.index2word[2])

# access vector for one word
#print("model demora:" + str(model['demora']))





#The Word2Vec metric tends to place two words close to each other if they occur in similar contexts  that is, w and w' are close to each other if the words that tend to show up near w also tend to show up near w'
#It s positive when the words are close to each other, negative when the words are far.  For two completely random words, the similarity is pretty close to 0.
#print("similarity:" + str(model.similarity("facturar","problema")))

#what words in Word2Vec are closest (positive) / farthest (negaive)
#print(model.most_similar(positive=["problema"]))
print("most similar:" + str(model.wv.most_similar("problema")))

#analogies
#model.wv.most_similar(positive=['woman', 'king'], negative=['man'])

# visualize a trained word embedding model using Principal Component Analysis.
# fit a 2d PCA model to the vectors
# X = model[model.wv.vocab]
# pca = PCA(n_components=2)
# result = pca.fit_transform(X)
# # create a scatter plot of the projection
# pyplot.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
# for i, word in enumerate(words):
# 	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
# pyplot.show()


#Word2Vec provides word embeddings only. If you want to characterize documents by embeddings, you'll need to perform an
# averaging/summing/max operation on embeddings of all words from each document to have a D-dimensional vector that can be used
# for classification

#Now that we have our word2vec model, how do we use it to transform each documents into a feature vector?

#We'll define a transformer (with sklearn interface) to convert a document into its corresponding vector

#averaging word vectors for all words in a text.
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# We will build a sklearn-compatible transformer that is initialised with a word -> vector dictionary. (w2v)
#zip the two lists containing vectors and words. the resulting list contains `(word, wordvector)` tuples.
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

mean_embedding_vectorizer = MeanEmbeddingVectorizer(w2v)
mean_embedding_vectorizer.transform(train["tokenized_sents"])  #not sure

#use svm as classifier
svm = SVC(C=1000.0, kernel='rbf', degree=3, gamma=0.00010000000000000001, coef0=0.0, shrinking=True,
                    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                    decision_function_shape="ovr", random_state=None)

svm_w2v = Pipeline([
    ("word2vec vectorizer", mean_embedding_vectorizer),
    ("svm", svm)])

#I am not sure if the "fit" is necessary or it is done inside pipeline. I think is necessary considering what I've read...
svm_w2v.fit(train["tokenized_sents"], y_train)
y_pred = svm_w2v.predict(test["tokenized_sents"])
score = accuracy_score(y_test, y_pred)
print score

#user TFID embbedding and another classifier
#copied from a tutorial this one. they do not do "fit" "transform" before passing to the pipeline (maybe is implicit on the method)
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v_tfidf.fit(train["tokenized_sents"], y_train)
y_pred2 = etree_w2v_tfidf.predict(test["tokenized_sents"])
score2 = etree_w2v_tfidf.score(y_test, y_pred2)  #score of pipeline
#score2 = accuracy_score(y_test, y_pred2)
print score2




