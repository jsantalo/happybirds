import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
#pip install emoji #install if not already installed
import emoji 
import re


# Josep
def uppercase_ratio_extract(text):
    upper = 0
    for char in text:
        if char.isupper():
            upper += 1
    r = upper / len(text)
    return r

def uppercase_ratio_extract_dataframe(df):
    df['upper_ratio'] = df['text'].apply(uppercase_ratio_extract)
    return df

#Bori
def stop_words_extract(df):
    stop = set(stopwords.words('english'))
    df['tweet_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


def tokenizer_extraction(df):
    tt = TweetTokenizer()
    df['list_of_words'] = df['tweet_without_stopwords'].apply(tt.tokenize)

    df["words"] = df['list_of_words'].apply(lambda x: ' '.join(x))

    return df


#data = df["text"]

def get_unigram(data):
    count_model = CountVectorizer(ngram_range=(1, 1))
    X = count_model.fit_transform(data)
    count_model.vocabulary_

    return X


def get_bigram(data):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
    X_2 = bigram_vectorizer.fit_transform(data)

    feature_index = bigram_vectorizer.vocabulary_.get('departure time')  # debug
    X_2[:, feature_index]  # debug --> donde esta “departure time”
    bigram_vectorizer.vocabulary_  # debug --> las veces que sale cada bigram
    return X_2



def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI

def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return True
    return False

def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux=[' '.join(r.findall(s)) for s in a_list]
    return(aux)

def add_emoji_column_to_df(df):   
    df['emoji']=df['text'].apply(lambda x: extract_emojis([x]))
    return(df)

#unig = get_unigram(texts)
#unig.toarray()  # matriz de unigrams

#big = get_bigram(texts)
#big.toarray()  # matriz --> casi todo 0 porque mucha info
