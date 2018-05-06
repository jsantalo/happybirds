import pandas as pd


def load_dataset(filename='', lan='english'):

    if lan == 'english':
        encoding = 'utf-8'
        if filename == '':
            filename = '../happybirds/data/tweets_public.csv'
    elif lan == 'spanish':
        encoding = 'utf-16'
        if filename == '':
            filename = '../happybirds/data/tweets_public_es.csv'
    else:
        raise Exception('Language not defined')

    return pd.read_csv(filename, index_col='tweet_id', encoding=encoding)
