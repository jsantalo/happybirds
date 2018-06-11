import sys
sys.path.insert(0, '..')

import core.train as training
import core.trans as transform
import core.test as testing
import nltk
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
import input.load_data as load_data


import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
matplotlib.rcParams['figure.dpi'] = 200

language = 'spanish' #options: spanish, english
df = load_data.load_dataset(lan=language)

# Read CSV file
#df = pd.read_csv('../happybirds/data/tweets_public.csv', index_col='tweet_id')
# Force datatime on the `tweet_created` column
#df.tweet_created = pd.to_datetime(df.tweet_created)
# How many tweets do we have?
print("Number of tweets:", df.shape[0])

# Show first rows on dataframe
#print(df.head())
#df.text.head()

# ejecutamos n_iterations veces la parte de validación y nos quedamos con la media
n_iterations = 1
score_mean = 0
test_size=0.25
validation_size=0.25
train, test = train_test_split(df, test_size=test_size)


for i in range(n_iterations):
    ctrain, validate = train_test_split(train, test_size=validation_size)

    print("ctrain.size :")
    print(len(ctrain))
    print("validate.size :")
    print(len(validate))

    transpk = transform.Trans()
    trainpk = training.Train()


    #pretransform function goes here, no?
    ctrain, ctrainr = transpk.pre_transform(df=ctrain)
    #validate dataset must be pretransformed too
    validate, validater = transpk.pre_transform(df=validate)

    # define training, test data. tokeninze text column of the dataframe
    ctrain["tokenized_sents"] = ctrain["text"].apply(nltk.word_tokenize)
    validate["tokenized_sents"] = validate["text"].apply(nltk.word_tokenize)

    #---dictionrary generator based on regular Count Vectorizer
    #trainpk.fit_bigram(data=ctrain.text, bow_size=200)
    #cv = trainpk.count_vectorizer

    #---dictionary generator based on get_vocabulaty per sentiment
    #bow_size2 = 200
    #trainpk.get_vocabulary_per_sentiment(ctrain, bow_size2, lemma_extraction=False, language_text=language,
    #                                     exclude_neutral = False)

    #---word2vec  (tal y como dejo el codigo no se usa, pero lo dejo porque se pude usar)
    #model = Word2Vec(ctrain['tokenized_sents'], min_count=10, workers=4, size=100)
    #w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    #mean_embedding_vectorizer = transform.MeanEmbeddingVectorizer(w2v)
    #mean_embedding_vectorizer.transform(ctrain["tokenized_sents"])

    #x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=ctrain, dfr=ctrainr)
    #x_train = transpk.normalize_train_data(x_train)
    y_train = ctrain['airline_sentiment'].values

    #x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=validate, dfr=validater)
    #x_validate = transpk.normalize_validate_data(x_validate)
    y_validate = validate['airline_sentiment'].values

    #trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    #best results so far with a rbf kernel, C=1000, gamma=0.0001 and quite independent of number of words 200 or 1000 with usual classifier
    #trainpk.model = SVC(C=1000.0, kernel=kernel, degree=3, gamma=0.00010000000000000001, coef0=0.0, shrinking=True,
    #                    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
    #                    decision_function_shape="ovr", random_state=None)

    #trainpk.model = LinearSVC(penalty="l2", dual=True, tol=0.0001, C=1000.0, multi_class="ovr",
    #                fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)


    #modelo que uso para el pipeline. no difiere mucho del linear
    svm = SVC(C=10.0, kernel='rbf', degree=3, gamma=0.0010000000000000001, coef0=0.0, shrinking=True,
              probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
              decision_function_shape="ovr", random_state=None)

    svc_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    svc_tfidf.fit(ctrain["tokenized_sents"], y_train)
    y_pred3 = svc_tfidf.predict(validate["tokenized_sents"])
    score = accuracy_score(y_validate, y_pred3)
    print(score)

    #-- se podria sustituir este en el tfid. pero me daba peor..
    #mean_embedding_vectorizer2 = transform.TfidfEmbeddingVectorizer(w2v)

    #--- pipeline de word2vec y svm. Da peor que tfid + svc
    # svm_w2v = Pipeline([
    #    ("word2vec vectorizer", mean_embedding_vectorizer2),
    #    ("svm", svm)])
    # svm_w2v.fit(ctrain["tokenized_sents"], y_train)
    # y_pred = svm_w2v.predict(validate["tokenized_sents"])
    # score = accuracy_score(y_validate, y_pred)


    #trainpk.fit(x_train, y_train)
    #print(y_train)
    #y_pred = trainpk.predict(x_validate)

    #testing.correlation_to_sentiment(x_train, ctrain, trainpk.get_vocabulary())

    #score = testing.score_model(y_validate, y_pred, True)
    score_mean = score_mean + score

print("promedio validacion:")
print(score_mean / n_iterations)