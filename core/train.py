from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import core.test as testing
import pandas as pd



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

    def get_vocabulary_per_sentiment(self, df, bow_size2, lemma_extraction=False):
        # example to call this function...
        #--- bow_size2 = 100
        #--- voc = trainpk.get_vocabulary_per_sentiment(ctrain, bow_size2, lemma_extraction=False)
        #--- print(len(voc))
        # filter by pos neg and neu
        df_neg = df[df['airline_sentiment'] == 'negative']
        df_neu = df[df['airline_sentiment'] == 'neutral']
        df_pos = df[df['airline_sentiment'] == 'positive']

        # extract bow_size2 negative terms and see correlation with sentiment
        self.fit_bigram(data=df_neg.text, bow_size=bow_size2)
        neg_voc = self.get_vocabulary()
        voc = neg_voc

        # Extract de word count and setup the dataframe for correlation analisis
        x = self.count_vectorizer.transform(df.text)
        dfr2 = pd.DataFrame(x.toarray())
        dfr2['tweet_id'] = df.index
        dfr2 = dfr2.set_index('tweet_id')
        # see correlation with sentiment
        print("------NEGATIVE WORDs CORRELATIONS-------")
        testing.correlation_to_sentiment(dfr2, df, neg_voc, list_correlations=True, list_stats=False)

        # extract bow_size2 positive terms and see correlation with sentiment
        #*****if neutral vocabulary needed repeat this part with neutral df****
        self.fit_bigram(data=df_pos.text, bow_size=bow_size2)
        pos_voc = self.get_vocabulary()  # continuar aqui

        # Extract de word count and setup the dataframe for correlation analisis
        x = self.count_vectorizer.transform(df.text)
        dfr2 = pd.DataFrame(x.toarray())
        dfr2['tweet_id'] = df.index
        dfr2 = dfr2.set_index('tweet_id')
        # see correlation with sentiment
        print("------- POSITIVE WORDs CORRELATIONS-------")
        testing.correlation_to_sentiment(dfr2, df, pos_voc, list_correlations=True, list_stats=False)

        voc_len = len(voc)
        n = 0
        for k in pos_voc.keys():
            if k not in voc.keys():
                voc.update({k: voc_len + n})
                n = n + 1
        #**** if neutral vocabulary needed repeat this part with neutral df****
        # creamos el nuevo count_vecrtorizer con el nuevo vocabulario
        self.count_vectorizer=CountVectorizer(vocabulary=voc)
        return voc

    def set_model(self, dmodel, *model_args, **model_kwargs):
        self.model = dmodel(*model_args, **model_kwargs)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
