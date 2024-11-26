import pickle
import pandas as pd

from argparse import ArgumentParser
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

input_file_path = './datasets/loan_data.csv'
output_file_path = './datasets/dv_model.pkl'


def range_credit_score(credit_score):
    if (300 <= credit_score) & (credit_score <= 579):
        return 'Poor'
    elif (580 <= credit_score) & (credit_score <= 669):
        return 'Fair'
    elif (670 <= credit_score) & (credit_score <= 739):
        return 'Good'
    elif (740 <= credit_score) & (credit_score <= 799):
        return 'Very Good'
    elif (800 <= credit_score) & (credit_score <= 850):
        return 'Excellent'
    else:
        return 'Unknown'


def get_features(dataframe):
    num = dataframe.select_dtypes(include=['number'], exclude='object').columns.values
    cat = dataframe.select_dtypes(include=['object']).columns.values
    num = list(num)
    cat = list(cat)
    return num, cat


def train_train_logistic_regression_model(dataframe, features, target, C=1.0):
    dv = DictVectorizer(sparse=False)
    dicts = dataframe[features].to_dict(orient='records')
    X_train = dv.fit_transform(dicts)
    model_lr = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model_lr.fit(X_train, target)
    return dv, model_lr


if __name__ == '__main__':
    parser = ArgumentParser(description="Train model")
    parser.add_argument('--input', type=str,
                        help=f"Specify path to dataset. '{input_file_path}' will be used by default.",
                        default=input_file_path)
    parser.add_argument('--out', type=str,
                        help=f"Specify path to output file. '{output_file_path}' will be used by default.",
                        default=output_file_path)
    args = parser.parse_args()
    input_file_path = args.input
    output_file_path = args.out
    print("Loading dataset...")
    print(input_file_path)

df = pd.read_csv(f'{input_file_path}', sep=',')
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['credit_score_label'] = df['credit_score'].apply(range_credit_score)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.loan_status.values
y_test = df_test.loan_status.values

del df_full_train['loan_status']
del df_test['loan_status']

numerical, categorical = get_features(df_full_train)
all_features = categorical + numerical

dv, model = train_train_logistic_regression_model(df_full_train, all_features, y_full_train, C=1.0)

with open(output_file_path, 'wb') as f:
    pickle.dump((dv, model), f)
    print("Saving model to disk...")
    print(output_file_path)
