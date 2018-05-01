from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords




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

    def fit_bigram(self, data, bow_size, lemma_extraction=False):

        if lemma_extraction:
            tokenizer = LemmaTokenizer()
        else:
            tokenizer = None

        spanish_stopwords = stopwords.words('spanish')

        self.count_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                                lowercase=True,
                                                token_pattern=r'\b[A-Za-z]+\b',
                                                stop_words=spanish_stopwords,
                                                min_df=1,
                                                max_features=bow_size,
                                                tokenizer=tokenizer)
        self.count_vectorizer.fit(data)

        return self.count_vectorizer

    def set_model(self, dmodel, *model_args, **model_kwargs):
        self.model = dmodel(*model_args, **model_kwargs)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
