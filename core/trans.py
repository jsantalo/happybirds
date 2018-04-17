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

def add_emoji_len_column_to_df(df_e):
    # add a column in the df with all emojis if any, mas rapido porque checkea si hay o no antes de asignar
    #df_e['emoji']=""
    for i in df_e.index:
        if(text_has_emoji(df_e.text[i])):
            [df_e.loc[i,'emoji'],df_e.loc[i,'num_emoji']]=(extract_emojis([df_e.text[i]]))
        else:
            df_e.loc[i,'num_emoji']=0
    return(df_e['num_emoji']) # es un poco más rapido

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

        #hot encoding of 'negativereason' and add columns to 'dfr'
        #dfr = create_hot_encoding_dataframe(dfr, df)

        #create columns with month, day, hour. I think DatetimeIndex method converts in local time using timezone

        dfr['Month'] = pd.DatetimeIndex(df['tweet_created']).month
        dfr['Day'] = pd.DatetimeIndex(df['tweet_created']).day
        dfr['Hour'] = pd.DatetimeIndex(df['tweet_created']).hour
        # dfr['Year'] = pd.DatetimeIndex(df['tweet_created']).year

        return dfr


