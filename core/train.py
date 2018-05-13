from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import core.test as testing
import pandas as pd


def encode_sentiment(sentiment, sentiment_options):
    i = 0
    for s in sentiment_options:
        if (s == sentiment):
            sentiment_encoded = i
            break
        i = i + 1
    return sentiment_encoded


def correlation_to_sentiment(x_train, ctrain, voc, list_correlations=True, list_stats=False, number_features='all'):
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

    if list_stats:
        print("\nMean values of feature grouped by sentiment:\n")
        for ele in corr_mat_sorted[0:number_features]:
            if isinstance(ele[0], int):
                print(
                    "the vocabular term for %d is: %s " % (ele[0], list(voc.keys())[list(voc.values()).index(ele[0])]))
            print(x_train[[ele[0], 'airline_sentiment']].groupby('airline_sentiment').mean())

    return corr_mat_sorted

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class Train:

    def __init__(self):
        self.count_vectorizer = None
        self.model = None

    def get_vocabulary(self):
        return self.count_vectorizer.vocabulary_

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


    def get_vocabulary_per_sentiment(self, df, bow_size2, lemma_extraction=False,language_text='english'):
        # example to call this function...
        #--- bow_size2 = 100
        #--- voc = trainpk.get_vocabulary_per_sentiment(ctrain, bow_size2, lemma_extraction=False)
        #--- print(len(voc))
        # filter by pos neg and neu
        #hacerlo en un bucle

        sentiment_options = sorted(df['airline_sentiment'].unique())
        voc = dict()

        for sentiment in sentiment_options:
            df_sent=df[df['airline_sentiment'] == sentiment]

            # extract bow_size2 per sentiment_option terms and see correlation with sentiment
            self.fit_bigram(data=df_sent.text, bow_size=bow_size2,lemma_extraction=lemma_extraction,language=language_text)
            aux_voc = self.get_vocabulary()

            # Extract de word count and setup the dataframe for correlation analisis
            x = self.count_vectorizer.transform(df.text)
            dfr2 = pd.DataFrame(x.toarray())
            dfr2['tweet_id'] = df.index
            dfr2 = dfr2.set_index('tweet_id')
            # see correlation with sentiment
            print("------- %s WORDs CORRELATIONS-------"% sentiment)
            correlation_to_sentiment(dfr2, df, aux_voc, list_correlations=True, list_stats=False)

            #add "sentiment" vocabulary to general VOC
            voc_len = len(voc)
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

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
