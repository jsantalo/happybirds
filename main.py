import sys
sys.path.insert(0, '..')

import core.train as training
import core.trans as transform
import core.test as testing

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
    ctrain, ctrainr = transpk.pre_transform(df=ctrain, language=language)
    #validate dataset must be pretransformed too
    validate, validater = transpk.pre_transform(df=validate, language=language)

    #---dictionrary generator based on regular Count Vectorizer
    #trainpk.fit_bigram(data=ctrain.text, bow_size=5000)
    #cv = trainpk.count_vectorizer

    #---dictionary generator based on get_vocabulaty per sentiment
    bow_size2 = 2000
    trainpk.get_vocabulary_per_sentiment(ctrain, bow_size2, lemma_extraction=False, language_text=language,
                                         exclude_neutral = False,col_text = 'text')

    x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=ctrain, dfr=ctrainr)
    y_train = ctrain['airline_sentiment'].values

    x_train = transpk.normalize_train_data(x_train)
    #print(x_train.head(100))
    x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=validate, dfr=validater)
    x_validate = transpk.normalize_validate_data(x_validate)
    y_validate = validate['airline_sentiment'].values

    #print(x_train.describe())
    #print(x_validate.describe())

    #trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    kernel = "rbf"
    #kernel="sigmoid"
    #best results so far with a rbf kernel, C=1000, gamma=0.0001 and quite independent of number of words 200 or 1000 with usual classifier

    #trainpk.model = SVC(C=10.0, kernel=kernel, degree=3, gamma=0.00010000000000000001, coef0=0.0, shrinking=True,
     #                   probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
      #                  decision_function_shape="ovr", random_state=None)
    trainpk.optimize_happy_SVC(x_train, y_train, kernel=kernel)
    #trainpk.model = LinearSVC(penalty="l2", dual=True, tol=0.0001, C=1000.0, multi_class="ovr",
    #                fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

    trainpk.fit(x_train, y_train)
    print(y_train)
    y_pred = trainpk.predict(x_validate)

    #testing.correlation_to_sentiment(x_train, ctrain, trainpk.get_vocabulary())

    score = testing.score_model(y_validate, y_pred, True)
    score_mean = score_mean + score

print("promedio validacion:")
print(score_mean / n_iterations)