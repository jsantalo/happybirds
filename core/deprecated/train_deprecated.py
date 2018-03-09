import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from core import trans

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


def train_score(df):
    # cada vez que lo ejecutas da un resultado diferente, ojo, hay que hacer la crossvalidation? k-flod
    scoreNB_mean = 0
    scoreKN_mean = 0
    boW_size = 4000
    vocabular_bi = None

    for i in range(10):
        # df_emoji=add_emoji_column_to_df(df)
        # print(df_emoji['emoji'])
        dataset, vocabular_bi = trans.obtain_data_representation(df, boW_size)

        # Train a Bernoulli Naive Bayes
        modelNB, _, scoreNB = train_model(dataset, BernoulliNB)

        # Train a K Nearest Neighbors Classifier
        modelKN, _, scoreKN = train_model(dataset, KNeighborsClassifier)
        scoreNB_mean = scoreNB_mean + scoreNB
        scoreKN_mean = scoreKN_mean + scoreKN

    print(scoreNB_mean / 10)
    print(scoreKN_mean / 10)

    print((vocabular_bi))