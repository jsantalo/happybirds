import sys
sys.path.insert(0, '..')

import core.train as training
import core.trans as transform
import core.test as testing

#pd.options.display.max_columns = None
#pd.options.display.max_rows = None

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



import matplotlib

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
matplotlib.rcParams['figure.dpi'] = 200

import seaborn as sns


# Read CSV file
df = pd.read_csv('../happybirds/data/tweets_public.csv', index_col='tweet_id')
# Force datatime on the `tweet_created` column
df.tweet_created = pd.to_datetime(df.tweet_created)
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

    #pretransform function here


    #dictionrary generator (Count Vectorizer)
    trainpk.fit_bigram(data=ctrain.text, bow_size=1000)
    cv = trainpk.count_vectorizer
    # bow_size2=50
    # trainpk.get_vocabulary_per_sentiment(ctrain,bow_size2, lemma_extraction=False,ngram_range=(1, 2))

    x_train = transpk.transform(count_vectorizer=cv, df=ctrain)
    y_train = ctrain['airline_sentiment'].values
    print(x_train.head(100))
    x_validate = transpk.transform(count_vectorizer=cv, df=validate)
    y_validate = validate['airline_sentiment'].values

    #print(x_train.describe())
    #print(x_validate.describe())

    trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

    trainpk.fit(x_train, y_train)
    print(y_train)
    y_pred = trainpk.predict(x_validate)

    #testing.correlation_to_sentiment(x_train, ctrain, trainpk.get_vocabulary())

    score = testing.score_model(y_validate, y_pred, True)
    score_mean = score_mean + score

print("promedio validacion:")
print(score_mean / n_iterations)