import datetime
import pandas as pd
import core.train as train

def create_submit_file(df_submission, ypred):
    date = datetime.datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    filename = 'submission_' + date + '.csv'

    df_submission['airline_sentiment'] = ypred
    df_submission[['airline_sentiment']].to_csv(filename)

    print('Submission file created: {}'.format(filename))
    print('Upload it to Kaggle InClass')


# Read submission and retrain with whole data
df_submission = pd.read_csv('tweets_submission.csv', index_col='tweet_id')
# We use df_submision as test, otherwise it would split df in train/test
submission_dataset = train.obtain_data_representation(df, df_submission)
# Predict for df_submission
_, y_pred = train.train_model(submission_dataset, BernoulliNB)

# Create submission file with obtained y_pred
create_submit_file(df_submission, y_pred)
