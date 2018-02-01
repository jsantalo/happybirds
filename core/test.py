import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import core.train as train



#cada vez que lo ejecutas da un resultado diferente, ojo, hay que hacer la crossvalidation? k-flod
scoreNB_mean=0
scoreKN_mean=0
boW_size=4000
vocabular_bi=None
for i in range(10):
    #df_emoji=add_emoji_column_to_df(df)
    #print(df_emoji['emoji'])
    dataset,vocabular_bi = train.obtain_data_representation(df,boW_size)


    # Train a Bernoulli Naive Bayes
    modelNB,_,scoreNB= train.train_model(dataset, BernoulliNB)
    
    # Train a K Nearest Neighbors Classifier
    modelKN,_,scoreKN= train.train_model(dataset, KNeighborsClassifier)
    scoreNB_mean=scoreNB_mean+scoreNB
    scoreKN_mean=scoreKN_mean+scoreKN
    
print(scoreNB_mean/10)
print(scoreKN_mean/10)

print((vocabular_bi))