import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import core.train as training
import core.trans as transform


def score_model(y_test, y_pred, verbose=False):
    score = accuracy_score(y_test, y_pred)
    if verbose:
        print("Model score is: {}".format(score))

    return score


class Test:

    def __init__(self, df=None, test_size=0.25, validation_size=0.25, n_iterations=10):

        self.test_size = test_size
        self.validation_size = validation_size
        self.n_iterations = n_iterations

        self.df = df

    def load_from_csv(self, filename, index_col='tweet_id'):
        self.df = pd.read_csv(filename, index_col=index_col)

    def train_validate_test(self, filename=None, verbose=False):

        if filename is not None:
            self.load_from_csv(filename)

        train, test = train_test_split(self.df, test_size=self.test_size)

        rdf = pd.DataFrame(columns=['score_train', 'score_validate', 'trans', 'train'])

        for i in range(self.n_iterations):

            if verbose:
                print("Train iteration {}".format(i))

            ctrain, validate = train_test_split(train, test_size=self.validation_size)

            transpk = transform.Trans()
            trainpk = training.Train()

            trainpk.fit_bigram(data=ctrain.text, bow_size=1000)

            x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=ctrain)
            y_train = ctrain['airline_sentiment'].values

            trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

            trainpk.fit(x_train, y_train)

            y_train_pred = trainpk.predict(x_train)
            score_train = score_model(y_train, y_train_pred)

            if verbose:
                print("Model train score is: {}".format(score_train))

            x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=validate)
            y_validate = validate['airline_sentiment'].values

            y_validate_pred = trainpk.predict(x_validate)

            score_validate = score_model(y_validate, y_validate_pred)

            if verbose:
                print("Model validation score is: {}".format(score_validate))

            rdf.loc[len(rdf)] = [score_train, score_validate, transpk, trainpk]

        best = rdf.ix[rdf['score_validate'].argmax()]

        score_train = best['score_train']
        score_validate = best['score_validate']
        transpk = best['trans']
        trainpk = best['train']

        x_test = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=test)
        y_test = test['airline_sentiment'].values

        y_pred = trainpk.predict(x_test)

        score_test = score_model(y_test, y_pred)

        print("Model best train score is: {}".format(score_train))
        print("Model best validation score is: {}".format(score_validate))
        print("Model test score is: {}".format(score_test))



