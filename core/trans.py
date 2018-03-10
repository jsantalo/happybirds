import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import emoji
import re


def uppercase_ratio_extract(text):
    upper = 0
    for char in text:
        if char.isupper():
            upper += 1
    r = upper / len(text)
    return r


def uppercase_ratio_extract_dataframe(df, col_txt='text'):
    return df[col_txt].apply(uppercase_ratio_extract)


def char_is_emoji(character):
    return character in emoji.UNICODE_EMOJI


def text_has_emoji(text):
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            return 1
    return 0


def tweet_has_emoji(df, col_txt='text'):
    return df[col_txt].apply(text_has_emoji)


def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux = [' '.join(r.findall(s)) for s in a_list]
    return (aux)


def add_emoji_column_to_df(df, col_txt='text'):
    return df[col_txt].apply(lambda x: extract_emojis([x]))

def count_text_length(text):
    return len(text)


def count_text_length_dataframe(df, col_txt='text'):
    return df[col_txt].apply(count_text_length)



class Trans:

    def __init__(self):
        pass

    # Return a new dataframe with the transformations done
    def transform(self, df, count_vectorizer, col_txt='text'):

        # Extract de word count and setup the dataframe
        x = count_vectorizer.transform(df[col_txt])
        dfr = pd.DataFrame(x.toarray())
        dfr['tweet_id'] = df.index
        dfr = dfr.set_index('tweet_id')

        # Column to count uppercase ratio
        dfr['upper_ratio'] = uppercase_ratio_extract_dataframe(df)

        dfr['has_emoji'] = tweet_has_emoji(df)

        dfr['text_length'] = count_text_length_dataframe(df)

        return dfr


