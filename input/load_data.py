import pandas as pd


def load_dataset(filename='', lan='en'):

    if lan == 'en':
        encoding = 'utf-8'
        if filename == '':
            filename = '../data/tweets_public.csv'
    elif lan == 'es':
        encoding = 'utf-16'
        if filename == '':
            filename = '../data/tweets_public_es.csv'
    else:
        raise Exception('Language not defined')

    return pd.read_csv(filename, index_col='tweet_id', encoding=encoding)
