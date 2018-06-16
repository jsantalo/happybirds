import pandas as pd
from nltk.corpus import stopwords

from nltk.tokenize import TweetTokenizer
from nltk.tokenize import RegexpTokenizer

from nltk import PorterStemmer
from nltk import LancasterStemmer
from nltk import SnowballStemmer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import CountVectorizer
import emoji
import re
from operator import itemgetter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

stop = set(stopwords.words('english')) # when working on the Spanish, change to "spanish"

porter = PorterStemmer()
lancaster = LancasterStemmer()

def uppercase_ratio_extract(text):
    upper = 0
    for char in text:
        if char.isupper():
            upper += 1
    r = upper / len(text)
    return r


def uppercase_ratio_extract_dataframe(df, col_txt='text'):
    return df[col_txt].apply(uppercase_ratio_extract)


def remove_url(text):
    return re.sub(r"http\S+", "", text)


def remove_url_dataframe(df, col_txt='text'):
    return df[col_txt].apply(remove_url)



def count_urls(text):
    return len(re.findall(r"http\S+", text))


def count_url_dataframe(df, col_txt='text'):
    return df[col_txt].apply(count_urls)


def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI


def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return 1
    return 0


def tweet_has_emoji(df, col_txt='text'):
    return df[col_txt].apply(text_has_emoji)


#find if the tweet has a particular emoji
def tweet_has_single_emoji(df, emoji_name, col_txt='text'):
    return df[col_txt].apply((lambda x: text_has_single_emoji(x, emoji_name)))

def text_has_single_emoji(text, emoji_name):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            character_emoji_name = emoji.demojize(character)
            if character_emoji_name == emoji_name:
                #print("character:" + character + ",character_emoji_name:" + character_emoji_name + ",emoji_name:" + emoji_name)
                return 1
    return 0

#create one column for each emoji. If the tweet has the particular emoji row== 1, else, row== 0
def emoji_hot_encoding(df, dfr, emoji_sorted_dict):
    # iterate the emojis dictionary to create one column per emoji.
    for k, v in emoji_sorted_dict:
        # print(emoji.emojize(k + ":" + str(v)))
        dfr[k] = tweet_has_single_emoji(df, k, col_txt='text')
    return dfr


def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux = [' '.join(r.findall(s)) for s in a_list]
    return (aux)

def extract_len_emojis(a_list):
    #num_emojis=0
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    for s in a_list:
        num_emojis=len(r.findall(s))
    return num_emojis


def add_emoji_column_to_df(df, col_txt='text'):
    return df[col_txt].apply(lambda x: extract_emojis([x]))

def add_emoji_len_column_to_df(df, col_txt='text'):
    return df[col_txt].apply(lambda x: extract_len_emojis([x]))

def remove_repeated(text):
    #remove repeated characters form text
    return re.subn(r'(\D\S)\1\1{1,}',r'\1\1', text)

def count_and_remove_puntuation(text):
    #remove repeated characters form text
    return re.subn(r'\.{3,}|!{1,}|\?{1,}',r'', text)

def count_and_remove_exclamation(text):
    #remove repeated characters form text
    return re.subn(r'!{1,}',r'', text)

def count_and_remove_qmark(text):
    #remove repeated characters form text
    return re.subn(r'\?{1,}',r'', text)

def count_and_remove_3dot(text):
    #remove repeated characters form text
    return re.subn(r'\.{3,}',r'', text)

def count_and_remove_single_dot(text):
    #remove repeated characters form text
    return re.subn(r'\.',r'', text)

def count_text_length(text):
    return len(text)


def count_text_length_dataframe(df, col_txt='text'):
    return df[col_txt].apply(count_text_length)

def create_hot_encoding(df, col):
    df_hot_encoding = pd.get_dummies(df[col])
    return df_hot_encoding

def create_hot_encoding_dataframe(dfr, df):
    df_hot_encoding = create_hot_encoding(df, 'negativereason')
    dfr = pd.concat([dfr, df_hot_encoding], axis=1)
    return dfr

def create_hot_encoding_dataframe_airline(dfr, df):
    df_hot_encoding = create_hot_encoding(df, 'airline')
    dfr = pd.concat([dfr, df_hot_encoding], axis=1)
    return dfr

def clean_text_lemmatize(df):
    df["text_original"] = df["text"]
    df["text"] = df["text"].apply(lambda x: x.lower())
    df["text"] = df["text"].apply(lambda x: re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',x)) # convert all links to "URL"
    df["text"] = df["text"].apply(lambda x: re.sub('@[^\s]+','ATUSER',x)) # convert all users to "ATUSER"
    df["text"] = df["text"].apply(lambda x: re.sub("[^a-zA-Z]+", " ", x)) # keep alphabetic only
    df['tweet_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    tt = TweetTokenizer()
    df['list_of_words'] = df['tweet_without_stopwords'].apply(tt.tokenize) # tokenize text

    df["words_list_porter"] = df["list_of_words"].apply(lambda x:[porter.stem(y) for y in x]) # stemmer porter
    df["words_list_lancaster"] = df["list_of_words"].apply(lambda x:[lancaster.stem(y) for y in x]) # stemmer lancaster
    df["words_lancaster"] = df['words_list_lancaster'].apply(lambda x: ' '.join(x)) #convert list to string for TfidfVectorizer  if used
    df["words_porter"] = df['words_list_porter'].apply(lambda x: ' '.join(x))
    del df["words_list_lancaster"]
    del df["words_list_porter"]
    return df

def remove_tweets_with_word(df, word):
    #print(df.text.count())
    new_df = df[df.text.str.contains(word) == False]
    return new_df

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(word2vec))])
        else:
            self.dim=0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])



#averaging word vectors for all words in a text.
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(word2vec))])
        else:
            self.dim=0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def tokenization (data, col_txt='text', language='english'):

    tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
    # stemmer = LancasterStemmer()
    stemmer = SnowballStemmer(language)

    tokenized_corpus = []

    for i, tweet in enumerate(data[col_txt]):
        tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@') and t not in stopwords.words(language) ]
        tokenized_corpus.append(tokens)

    return tokenized_corpus


class Trans:

    def __init__(self):
        pass


    def pre_transform(self, df, col_text='text', language='english'):

        #dataframe to enter inside the Classifier
        dfr = pd.DataFrame()

        #drop duplicates should be after remove url, otherwise the url create a different tweet
        #df = df.drop_duplicates(subset='text')  # remove dupicate tweets by text

        #df = remove_tweets_with_word(df, "Spanair")

        dfr['tweet_id'] = df.index
        dfr = dfr.set_index('tweet_id')

        dfr['count_url'] = count_url_dataframe(df, col_txt=col_text)

        df[col_text] = remove_url_dataframe(df, col_txt=col_text)

        # Column to count uppercase ratio
        dfr['upper_ratio'] = uppercase_ratio_extract_dataframe(df, col_txt=col_text)
        
        #df[col_text], dfr['puntuation_removed'] = zip(*df[col_text].apply(count_and_remove_puntuation))
        df[col_text], dfr['3dot'] = zip(*df[col_text].apply(count_and_remove_3dot))
        df[col_text], dfr['question_marks'] = zip(*df[col_text].apply(count_and_remove_qmark))
        df[col_text], dfr['exclamation'] = zip(*df[col_text].apply(count_and_remove_exclamation))
        df[col_text], dfr['single_dot'] = zip(*df[col_text].apply(count_and_remove_single_dot)) #always after 3 dot extraction
        df[col_text], dfr['number_of_subs_made'] = zip(*df[col_text].apply(remove_repeated))
        #very few in english text [~16 from 4941 tweets]
        #I am getting a warning "variable is trying to set a copy of itself" --> how to deal with it??

        #df = clean_text_lemmatize(df)
        df[col_text] = df[col_text].apply(lambda x: x.lower())

        df['tokenized_corpus'] = tokenization(df, col_txt=col_text, language=language);

        return df, dfr


    # Return a new dataframe with the transformations done
    def transform(self, df, count_vectorizer=None, word2vec=None,  dfr=None, col_txt='text'):

        if dfr is None:
            dfr = pd.DataFrame()

        if count_vectorizer is not None:
            # Extract de word count and setup the dataframe
            x = count_vectorizer.transform(df[col_txt])
            dftmp = pd.DataFrame(x.toarray())
            dftmp['tweet_id'] = df.index
            dftmp = dftmp.set_index('tweet_id')

            dfr = pd.concat([dfr, dftmp], axis=1)

        if word2vec is not None and hasattr(df, 'tokenized_corpus'):

            colnames = []
            for i in range(0, word2vec.vector_size):
                colnames.append('w2v_' + str(i))

            dftmp = pd.DataFrame(columns=colnames)
            for i, token_vec in df.tokenized_corpus.iteritems():
                vec = [word2vec[token] for token in token_vec if token in word2vec.vocab]
                if len(vec) > 0:
                    mean = np.mean(np.array(vec), axis=0)
                else:
                    mean = np.zeros(word2vec.vector_size)
                dftmp = dftmp.append(pd.DataFrame(mean.reshape(-1, len(mean)), index=[i], columns=colnames))
            dftmp.index.names = ['tweet_id']

            dfr = pd.concat([dfr, dftmp], axis=1)

        dfr['has_emoji'] = tweet_has_emoji(df, col_txt=col_txt)

        dfr['text_length'] = count_text_length_dataframe(df, col_txt=col_txt)
        #number of emojis in text can replace tweet_has_emoji --> comment on Wednesday
        dfr['num_emoji']=add_emoji_len_column_to_df(df, col_txt=col_txt)

        #hot encoding of 'negativereason' and add columns to 'dfr'
        #dfr = create_hot_encoding_dataframe(dfr, df)

        # hot encoding of 'airline' and add columns to 'dfr'
        # not better results so far, comment if needed
        # dfr = create_hot_encoding_dataframe_airline(dfr, df)

        #create columns with month, day, hour. I think DatetimeIndex method converts in local time using timezone
        #dfr['Month'] = pd.DatetimeIndex(df['tweet_created']).month
        dfr['Day'] = pd.DatetimeIndex(df['tweet_created']).day
        dfr['Hour'] = pd.DatetimeIndex(df['tweet_created']).hour
        dfr['dayofweek'] = pd.DatetimeIndex(df['tweet_created']).dayofweek
        #print(dfr.describe())


        # create an ordered dictionary with number of emoji appearences for all the dataset
        emoji_sorted_dict = self.appearences_emoji_total(df)

        #hot encoding with one column per emoji present in the emoji_sorted_dict
        #COMENTED AT THE MOMENT because we have to talk about it on Tuesday
        #dfr = emoji_hot_encoding(df, dfr, emoji_sorted_dict)

        return dfr

    def normalize_train_data(self, dfr):
        #normalize data to the mean and variance of train data
        self.scaler = StandardScaler()
        dfr = self.scaler.fit_transform(dfr)
        return dfr
    def normalize_validate_data(self, validate):
        #normalize validate data with the mean and variance of train data.
        #Must ve called after "normalize_train_data"
        validate = self.scaler.transform(validate)
        return validate

    # analyze which emojis appear more frequently.
    # Once we know the more frequents, classify as positive or negative taking into account the correlation with other P/N tweets
    def appearences_emoji_total(self, df, col_txt='text'):

        # new dictionary with key=emoji_name, value=num_apperarences
        self.emoji_dict = {}

        # fill the dictionary with the emojis and number of appearences
        df[col_txt].apply((lambda x: self.text_dict_emoji(x)))

        #if we want to correlate with sentiment or something....
        #df[col_txt].apply((lambda x: self.text_dict_emoji(x, df["airline_sentiment"])))

        #sort DESC the dictionary.
        # At the beginning the idea was to get the emojis with more appearences or more correlation, but at the moment we are using all the emojis. Then, this function is not necessary.
        emoji_sorted_dict = sorted(self.emoji_dict.items(), key=itemgetter(1), reverse=True)
        return emoji_sorted_dict

        if verbose:
            print("dictionary emoji")
            print(self.emoji_dict)
            sorted_dict = sorted(self.emoji_dict.items(), key=itemgetter(1), reverse=True)  # ordenem el dict
            print("dictionary ordenat")
            print(sorted_dict)
            for k, v in sorted_dict:
                print(emoji.emojize(k + ":" + str(v)))


    #fill the dictionary with the emojis and appearences
    def text_dict_emoji(self, text):
        for character in text:
            if character in emoji.UNICODE_EMOJI:
                # print(character)
                emoji_name = emoji.demojize(character)
                #print(emoji_name)
                if emoji_name in self.emoji_dict: #if the emoji exist in dictionary
                    num = self.emoji_dict[emoji_name]
                    self.emoji_dict[emoji_name] = num + 1
                else:
                    self.emoji_dict[emoji_name] = 1