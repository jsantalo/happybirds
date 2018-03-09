import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
#pip install emoji #install if not already installed
import emoji 
import re
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

def obtain_data_representation(df, boW_size=200, test=None):
    # If there is no test data, split the input
    if test is None:
        # Divide data in train and test
        train, test = train_test_split(df, test_size=0.25)
        df.airline_sentiment = pd.Categorical(df.airline_sentiment)
    else:
        # Otherwise, all is train
        train = df

    # Create a Bag of Words (BoW), by using train data only
    vocabular_bi = get_bigram(train.text, boW_size)
    vocabular_bi.update({'😭': boW_size, '😆': boW_size + 1, '👍': boW_size + 2})

    cv = CountVectorizer(vocabulary=vocabular_bi)

    x_train = cv.fit_transform(train['text'])
    y_train = train['airline_sentiment'].values

    # Obtain BoW for the test data, using the previously fitted one
    x_test = cv.transform(test['text'])
    try:
        y_test = test['airline_sentiment'].values
    except:
        # It might be the submision file, where we don't have target values
        y_test = None

    return {
        'train': {
            'x': x_train,
            'y': y_train
        },
        'test': {
            'x': x_test,
            'y': y_test
        }
    }




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


def get_unigram(data):
    count_model = CountVectorizer(ngram_range=(1, 1))
    count_model.fit(data)
    return count_model.vocabulary_


def get_bigram(data, boW_size, lemma_extraction=False):

    if(lemma_extraction):
        tokenizer = LemmaTokenizer()
    else:
        tokenizer = None

    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                        lowercase=True,
                                        token_pattern=r'\b\w+\b',
                                        stop_words='english',
                                        min_df=1,
                                        max_features=boW_size,
                                        tokenizer=tokenizer)
    bigram_vectorizer.fit(data)
    return bigram_vectorizer.vocabulary_



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

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
