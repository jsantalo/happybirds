import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer


def uppercase_ratio_extract(text):
    upper = 0
    for char in text:
        if char.isupper():
            upper += 1
    r = upper / len(text)
    return r


def stop_words_extract(df):
    stop = set(stopwords.words('english'))
    df['tweet_without_stopwords'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    return df


def tokenizer_extraction(df):
    tt = TweetTokenizer()
    df['list_of_words'] = df['tweet_without_stopwords'].apply(tt.tokenize)

    df["words"] = df['list_of_words'].apply(lambda x: ' '.join(x))

    return df
