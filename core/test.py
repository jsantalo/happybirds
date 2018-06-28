import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import core.train as training
import input.load_data as load_data
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

        self.kernel="rbf"
        #self.kernel="sigmoid"


    def train_validate_test(self, filename=None, verbose=False, kaggle_filename=None, language='english', mode='standard', normalize=False):

        if verbose:
            print("Loading Dataset")

        if filename is not None:
            self.df = load_data.load_dataset(filename=filename, lan=language)
        else:
            self.df = load_data.load_dataset(lan=language)

        if mode == 'standard':
            validation_size = self.validation_size
            n_iterations = self.n_iterations

            train, test = train_test_split(self.df, test_size=self.test_size)

        elif mode == 'kaggle':

            validation_size = self.validation_size
            n_iterations = self.n_iterations

            if verbose:
                print("Loading kaggle testing set")

            train = self.df
            test = load_data.load_dataset(filename=kaggle_filename, encoding='utf-8', lan=language)

        elif mode == 'optimize':
            validation_size = 0
            n_iterations = 1

            train = self.df

        rdf = pd.DataFrame(columns=['score_train', 'score_validate', 'trans', 'train'])

        for i in range(n_iterations):

            if verbose:
                print("Train iteration {}".format(i))

            ctrain, validate = train_test_split(train, test_size=validation_size)

            transpk = transform.Trans()
            trainpk = training.Train()

            ctrain, ctrainr = transpk.pre_transform(df=ctrain, language=language)

            trainpk.fit_bigram(data=ctrain.text, bow_size=100, language=language)

            if hasattr(ctrain, 'tokenized_corpus'):
                trainpk.fit_word2vec(data=ctrain.tokenized_corpus)

            #print(trainpk.count_vectorizer.vocabulary_)
            #x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=ctrain, dfr=ctrainr)
            x_train = transpk.transform(count_vectorizer=trainpk.count_vectorizer, word2vec=trainpk.word2vec, df=ctrain, dfr=ctrainr)
            y_train = ctrain['airline_sentiment'].values

            if normalize:
                trainpk.normalize_train_data(x_train)

            if mode == 'optimize':
                if normalize:
                    trainpk.optimize_happy_SVC(transform.normalize_validate_data(trainpk.scaler, x_train), y_train)
                else:
                    trainpk.optimize_happy_SVC(x_train, y_train)
                print('SVM optimization done')
                #trainpk.optimize_happy_RF(x_train, y_train)
                #print('RF optimization done')
                #self.transpk = transpk
                #self.trainpk = trainpk
                return 0
            else:

                if normalize:
                    x_train = transform.normalize_validate_data(trainpk.scaler, x_train)

                ### Init RF testing
                # trainpk.model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

                ### The best parameters are {'max_depth': 11, 'n_estimators': 112} with a score of 0.59
                #trainpk.model = RandomForestClassifier(n_estimators=112, max_depth=11, n_jobs=-1)

                ### The best parameters are {'C': 1.0, 'gamma': 0.00615848211066026} with a score of 0.61
                #trainpk.model = SVC(C=1, kernel=self.kernel, degree=3, gamma=0.00615848211066026, coef0=0.0,
                #                    shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None,
                #                    verbose=False, max_iter=-1, decision_function_shape="ovr", random_state=None)

                trainpk.model = SVC(C=1.4384498882876628, kernel=self.kernel, gamma=0.00615848211066026, shrinking=True,
                    probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                    decision_function_shape="ovr", random_state=None)

            trainpk.fit(x_train, y_train)

            y_train_pred = trainpk.predict(x_train)
            score_train = score_model(y_train, y_train_pred)

            if verbose:
                print("Model train score is: {}".format(score_train))

            if validation_size > 0:
                validate, validater = transpk.pre_transform(df=validate)
                #x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=validate, dfr=validater)
                x_validate = transpk.transform(count_vectorizer=trainpk.count_vectorizer, word2vec=trainpk.word2vec, df=validate, dfr=validater)

                if normalize:
                    x_validate = transform.normalize_validate_data(trainpk.scaler, x_validate)

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
        #x_test = transpk.transform(count_vectorizer=trainpk.count_vectorizer, df=test, dfr=testr)
        x_test = transpk.transform(count_vectorizer=trainpk.count_vectorizer, word2vec=trainpk.word2vec, df=test, dfr=testr)

        if normalize:
            x_test = transform.normalize_validate_data(trainpk.scaler, x_test)

        y_pred = trainpk.predict(x_test)

        if mode == 'kaggle':
            kaggle_submit.create_submit_file(test, y_pred)
            score_test = None
        elif mode == 'standard':
            y_test = test['airline_sentiment'].values
            score_test = score_model(y_test, y_pred)

        print("Model best train score is: {}".format(score_train))
        print("Model best validation score is: {}".format(score_validate))
        print("Model test score is: {}".format(score_test))
