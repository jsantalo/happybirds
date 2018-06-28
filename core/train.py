from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
#import core.test as testing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

import multiprocessing

from gensim.models import Word2Vec

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


def encode_sentiment(sentiment, sentiment_options):
    i = 0
    for s in sentiment_options:
        if (s == sentiment):
            sentiment_encoded = i
            break
        i = i + 1
    return sentiment_encoded


def correlation_to_sentiment(x_train, ctrain, voc, list_correlations=True, list_stats=False, number_features='all',threshold_drop=0.01):
    # prints correlation between characteristics in x_train and airline sentiment in ctrain.
    # voc is the vocabulary of the count_vectorizer. if needed input trainpk.get_vocabulary()
    # returns the correlation matrix sorted
    sentiment_options = sorted(ctrain['airline_sentiment'].unique())
    print("found %d sentiments: %s" % (len(sentiment_options), sentiment_options))
    x_train['airline_sentiment'] = ctrain['airline_sentiment']
    x_train['sentiment_encoded'] = ctrain['airline_sentiment'].apply(lambda x: encode_sentiment(x, sentiment_options))
    corrmat = x_train.corr()
    cor_dict = corrmat['sentiment_encoded'].to_dict()
    del cor_dict['sentiment_encoded']
    corr_mat_sorted = sorted(cor_dict.items(), key=lambda x: -abs(x[1]))

    if number_features == 'all':
        number_features = len(corr_mat_sorted)

    if list_correlations:
        print("\nList of features in descencing order with its correlation to sentiment:\n")
        for ele in corr_mat_sorted[0:number_features]:
            if isinstance(ele[0], int):
                print("%d [%s] : \t\t%f " % (ele[0], list(voc.keys())[list(voc.values()).index(ele[0])], ele[1]))
            else:
                print("{0}: \t\t{1}".format(*ele))
            if abs(ele[1]) < threshold_drop:
                #feature selection
                if isinstance(ele[0], int):
                    #print(voc)
                    #print(voc[list(voc.keys())[list(voc.values()).index(ele[0])]])
                    del(voc[list(voc.keys())[list(voc.values()).index(ele[0])]])

    if list_stats:
        print("\nMean values of feature grouped by sentiment:\n")
        for ele in corr_mat_sorted[0:number_features]:
            if isinstance(ele[0], int):
                print(
                    "the vocabular term for %d is: %s " % (ele[0], list(voc.keys())[list(voc.values()).index(ele[0])]))
            print(x_train[[ele[0], 'airline_sentiment']].groupby('airline_sentiment').mean())

    return voc


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Train:

    def __init__(self):
        self.count_vectorizer = None
        self.model = None
        self.word2vec = None
        self.grid = None
        self.scaler = None

    def fit_word2vec(self, data, vector_size=100, window_size=5, min_count=5):

        w2vmodel = Word2Vec(sentences=data,
                            size=vector_size,
                            window=window_size,
                            min_count=min_count,
                            negative=20,
                            iter=50,
                            seed=1000,
                            workers=multiprocessing.cpu_count())

        self.word2vec = w2vmodel.wv

    def normalize_train_data(self, data):
        self.scaler = StandardScaler().fit(data)

    def get_vocabulary(self, source='bigram'):

        if source == 'bigram':
            return self.count_vectorizer.vocabulary_
        elif source == 'word2vec':
            return self.word2vec.vocab
        else :
            return None

    def fit_bigram(self, data, bow_size, lemma_extraction=False, language='english'):

        if lemma_extraction:
            tokenizer = LemmaTokenizer()
            #tokenizer = nltk.PorterStemmer()
            #tokenizer = nltk.LancasterStemmer()

        else:
            tokenizer = None

        #TODO In case we remove accents, adapt code
        stopwords_list = stopwords.words(language)

        self.count_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                                lowercase=True,
                                                token_pattern=r'\b[A-Za-z]+\b',
                                                stop_words=stopwords_list,
                                                min_df=1,
                                                max_features=bow_size,
                                                tokenizer=tokenizer)

        self.count_vectorizer.fit(data)

        return self.count_vectorizer

    def get_vocabulary_per_sentiment(self, df, bow_size2, lemma_extraction=False,language_text='english',exclude_neutral=False,col_text='text'):
        # example to call this function...
        #--- bow_size2 = 100
        #--- voc = trainpk.get_vocabulary_per_sentiment(ctrain, bow_size2, lemma_extraction=False,language_text=language)
        #--- print(len(voc))
        # filter by pos neg and neu
        #hacerlo en un bucle

        sentiment_options = sorted(df['airline_sentiment'].unique())
        if exclude_neutral:
            sentiment_options =['negative','positive']
        voc = dict()

        for sentiment in sentiment_options:
            df_sent=df[df['airline_sentiment'] == sentiment]

            # extract bow_size2 per sentiment_option terms and see correlation with sentiment
            self.fit_bigram(data=df_sent[col_text], bow_size=bow_size2,lemma_extraction=lemma_extraction,language=language_text)
            aux_voc = self.get_vocabulary()

            # Extract de word count and setup the dataframe for correlation analisis
            x = self.count_vectorizer.transform(df[col_text])
            dfr2 = pd.DataFrame(x.toarray())
            dfr2['tweet_id'] = df.index
            dfr2 = dfr2.set_index('tweet_id')
            # see correlation with sentiment
            print("------- %s WORDs CORRELATIONS-------"% sentiment)
            aux_voc = correlation_to_sentiment(dfr2, df, aux_voc, list_correlations=True, list_stats=False,threshold_drop=0.02)
            #use most correlated only!!!!!!!!!!!!

            #add "sentiment" vocabulary to general VOC
            voc_len = len(voc)
            print(aux_voc)
            n = 0
            for k in aux_voc.keys():
                if k not in voc.keys():
                    voc.update({k: voc_len + n})
                    n = n + 1
            print("lenght of vocabulary: %d"% len(voc))

        # create new count_vecrtorizer with combined vocabulary
        self.count_vectorizer=CountVectorizer(vocabulary=voc)
        return voc

    def set_model(self, dmodel, *model_args, **model_kwargs):
        self.model = dmodel(*model_args, **model_kwargs)

    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.
    # used in optimize_happy_SVC

    def optimize_happy_SVC(self, X, y,kernel="rbf"):

        print("SVM optimization start")

        C_range = np.logspace(0, 3, 20)
        gamma_range = np.logspace(-6, 0, 20)
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=multiprocessing.cpu_count())
        grid.fit(X, y)

        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

        scores = grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.45, midpoint=grid.best_score_-0.05))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('SM Validation accuracy')
        plt.show()

        #finally we create a model with best parameters
        self.model = SVC(C=grid.best_params_['C'], kernel=kernel, gamma=grid.best_params_['gamma'], shrinking=True,
                           probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                          decision_function_shape="ovr", random_state=None)

        self.grid = grid

    def optimize_happy_RF(self, X, y):

        print("Random Forest optimization start")

        n_estimators = np.logspace(1, 3, 20, dtype=np.dtype('int'))
        max_depth = np.logspace(0.75, 2.75, 20, dtype=np.dtype('int'))
        #max_features = ['auto', 'sqrt', 'log2']
        param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, n_jobs=multiprocessing.cpu_count())
        grid.fit(X, y)

        print("The best parameters are %s with a score of %0.2f"
              % (grid.best_params_, grid.best_score_))

        scores = grid.cv_results_['mean_test_score'].reshape(len(max_depth),
                                                             len(n_estimators))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=0.45, midpoint=grid.best_score_ - 0.05))
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.colorbar()
        plt.xticks(np.arange(len(n_estimators)), n_estimators, rotation=45)
        plt.yticks(np.arange(len(max_depth)), max_depth)
        plt.title('RF Validation accuracy')
        plt.show()

        self.model = RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],
                                            max_depth=grid.best_params_['max_depth'],
                                            n_jobs=multiprocessing.cpu_count())

        self.grid = grid

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
