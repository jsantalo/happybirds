import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import core.train as training
#get some problems with this imports, do not know why..
import core.trans as transform
import output.kaggle_submit as kaggle_submit



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

        self.transpk = None
        self.trainpk = None


    def train_validate_test(self, filename=None, verbose=False, kaggle_filename=None, language='english'):

        if filename is not None:
            self.df = load_data.load_dataset(filename=filename, lan=language)
        else:
            self.df = load_data.load_dataset(lan=language)

        validation_size = self.validation_size
        n_iterations = self.n_iterations

        if kaggle_filename is None:

            train, test = train_test_split(self.df, test_size=self.test_size)

            generate_kaggle = False

        else:

            train = self.df
            test = load_data.load_dataset(filename=kaggle_filename, lan=language)

            generate_kaggle = True

        rdf = pd.DataFrame(columns=['score_train', 'score_validate', 'trans', 'train'])

        for i in range(n_iterations):

            if verbose:
                print("Train iteration {}".format(i))

            ctrain, validate = train_test_split(train, test_size=validation_size)

            transpk = transform.Trans()
            trainpk = training.Train()

            ctrain, ctrainr = transpk.pre_transform(df=ctrain)

            trainpk.fit_bigram(data=ctrain.text, bow_size=1000, language=language)

            #print(trainpk.count_vectorizer.vocabulary_)
            x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=ctrain, dfr=ctrainr)
            y_train = ctrain['airline_sentiment'].values

            trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

            #trainpk.model = svm.SVC(kernel='rbf')

            trainpk.fit(x_train, y_train)

            y_train_pred = trainpk.predict(x_train)
            score_train = score_model(y_train, y_train_pred)

            if verbose:
                print("Model train score is: {}".format(score_train))

            if validation_size > 0:
                validate, validater = transpk.pre_transform(df=validate)
                x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=validate, dfr=validater)
                y_validate = validate['airline_sentiment'].values

                y_validate_pred = trainpk.predict(x_validate)

                score_validate = score_model(y_validate, y_validate_pred)
            else:
                score_validate = None

            if verbose:
                print("Model validation score is: {}".format(score_validate))

            rdf.loc[len(rdf)] = [score_train, score_validate, transpk, trainpk]

        if validation_size > 0:
            best = rdf.ix[rdf['score_validate'].argmax()]
        else:
            best = rdf.ix[rdf['score_train'].argmax()]

        score_train = best['score_train']
        score_validate = best['score_validate']
        self.transpk = best['trans']
        self.trainpk = best['train']

        test, testr = transpk.pre_transform(df=test)
        x_test = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=test, dfr=testr)

        y_pred = trainpk.predict(x_test)

        if generate_kaggle:
            kaggle_submit.create_submit_file(test, y_pred)
            score_test = None

        else:
            y_test = test['airline_sentiment'].values
            score_test = score_model(y_test, y_pred)

        print("Model best train score is: {}".format(score_train))
        print("Model best validation score is: {}".format(score_validate))
        print("Model test score is: {}".format(score_test))
