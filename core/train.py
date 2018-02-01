import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import core.transform as trans

def obtain_data_representation(df,boW_size=200, test=None):
    # If there is no test data, split the input
    if test is None:
        # Divide data in train and test
        train, test = train_test_split(df, test_size=0.25)
        df.airline_sentiment = pd.Categorical(df.airline_sentiment)
    else:
        # Otherwise, all is train
        train = df

    # Create a Bag of Words (BoW), by using train data only
    vocabular_bi=trans.get_bigram(df.text,boW_size)
    vocabular_bi.update({'😭':boW_size,'😆':boW_size+1,'👍':boW_size+2})
    
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


def train_model(dataset, dmodel, *model_args, **model_kwargs):

    model = dmodel(*model_args, **model_kwargs)

    # Train it
    model.fit(dataset['train']['x'], dataset['train']['y'])

    # Predict new values for test
    y_pred = model.predict(dataset['test']['x'])

    # Print accuracy score unless its the submission dataset
    if dataset['test']['y'] is not None:
        score = accuracy_score(dataset['test']['y'], y_pred)
        print("Model score is: {}".format(score))

    # Done
    return model, y_pred,score
