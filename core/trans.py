import pandas as pd
from nltk.corpus import stopwords
stop = set(stopwords.words('english')) # when working on the Spanish, change to "spanish"
from nltk.tokenize import TweetTokenizer
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
import emoji
import re
from operator import itemgetter


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
        #number of emojis in text can replace tweet_has_emoji --> comment on Wednesday
        dfr['len_emoji']=add_emoji_len_column_to_df(df)

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

        # create an ordered dictionary with number of emoji appearences for all the tweet
        self.appearences_emoji_total(df)

        return dfr


    # analyze which emojis appear more frequently.
    # Once we know the more frequents, classify as positive or negative taking into account the correlation with other P/N tweets
    def appearences_emoji_total(self, df, col_txt='text'):
        self.emoji_dict = {}  # new dictionary with key=emoji_name, value=num_apperarences

        df[col_txt].apply(self.text_dict_emoji)
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
                # print(emoji_name)
                if emoji_name in self.emoji_dict:
                    num = self.emoji_dict[emoji_name]
                    self.emoji_dict[emoji_name] = num + 1
                else:
                    self.emoji_dict[emoji_name] = 1