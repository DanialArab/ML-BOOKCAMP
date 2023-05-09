
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

n_splits = 10
C = [0.001, 0.01, 0.1, 0.5, 1, 10]
c_final = 1


print('Reading data')


file_directory_project_2 = 'D:/0_Machine_Learning/Software -- ML Journey/books/1.ML_bookcamps/WA_Fn-UseC_-Telco-Customer-Churn.csv'
df_2 = pd.read_csv(file_directory_project_2)


def dataframe_maker(dic_of_new_datapoint):
    df = pd.DataFrame.from_dict(dic_of_new_datapoint, orient='index').T
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    return df


def column_name_corrector(df):
    import re
    new_columns = []
    for one_col in df.columns:
        split = re.findall('[a-z]+|[A-Z][a-z]*', one_col)
        if len(split) > 1:
            final = split[0].capitalize() + '_' + \
                ''.join(re.findall('[a-z]+|[A-Z][a-z]*', one_col)[1:])
            new_columns.append(final)
        else:
            final = split[0].capitalize()

            new_columns.append(final)
        # print (final)
    return new_columns


def dtypes_corrector(df):
    numerical_columns_names = ['Senior_Citizen',
                               'Tenure', 'Monthly_Charges', 'Total_Charges']

    df[numerical_columns_names] = df[numerical_columns_names].astype(float)

    return  # df.dtypes


def space_corrector(df):
    for one_col in df.columns:

        if df[one_col].dtype == 'object':
            # print (one_col)
            df[one_col] = df[one_col].str.replace(' ', '_')

    return  # df


def data_encoder(df, categorical_columns_names, encoder):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    categorical_columns = df[categorical_columns_names]

    feature_names = encoder.get_feature_names_out(categorical_columns_names)
    encoded_new_data = encoder.transform(categorical_columns)

    encoded_df = pd.DataFrame(encoded_new_data, columns=feature_names)

    return feature_names, encoded_df


def train(df, categorical_columns_names, numerical_columns_names, C):
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression

    categorical_columns = df[categorical_columns_names]

    numerical_columns = df[numerical_columns_names]

    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(categorical_columns)

    feature_names = encoder.get_feature_names_out(categorical_columns.columns)
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

    dataset = pd.concat([numerical_columns, encoded_df,
                        df['Churn']], axis=1, ignore_index=False)


#     null_cols = dataset.columns[dataset.isnull().any()]
#     print(dataset[null_cols].isnull().sum())

    X = dataset.drop(['Churn'], axis=1)
    y = dataset['Churn']

    lr_model = LogisticRegression(solver='liblinear', C=C)
    model = lr_model.fit(X, y)

    return encoder, model


def predict(df, numerical_columns_names, encoded_df, model):

    numerical_columns = df[numerical_columns_names]

    X = pd.concat([numerical_columns, encoded_df], axis=1)
    y_pred = model.predict_proba(X)[:, 1]

#     null_cols = X.columns[X.isnull().any()]
#     print(X[null_cols].isnull().sum())

    return y_pred

# Data preparation


print('Column name correction is performing')

df_2.columns = column_name_corrector(df_2)

df_2['Total_Charges'] = df_2['Total_Charges'].str.replace(
    ' ', '0')  # .astype(float)

df_2.loc[df_2['Churn'] == 'No', 'Churn'] = 0
df_2.loc[df_2['Churn'] == 'Yes', 'Churn'] = 1
df_2['Churn'] = df_2['Churn'].astype(int)

df_2.dtypes

print('Data types correction is performing')


dtypes_corrector(df_2)


df_2.dtypes


print('Space correction in the string columns is performing')


space_corrector(df_2)

numerical_columns_names = ['Tenure', 'Monthly_Charges', 'Total_Charges']

categorical_columns_names = ['Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service',
                             'Multiple_Lines', 'Internet_Service', 'Online_Security',
                             'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV',
                             'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method']


numerical_columns = df_2[numerical_columns_names]
categorical_columns = df_2.drop(
    ['Tenure', 'Monthly_Charges', 'Total_Charges', 'Churn'], axis=1)

X = pd.concat([categorical_columns, numerical_columns], axis=1)
y = df_2['Churn']

X_train_full, X_test, y_train_full, y_test = train_test_split(X,
                                                              y,
                                                              test_size=0.2,
                                                              random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_full,
                                                  y_train_full,
                                                  test_size=0.25,
                                                  random_state=42)

df_train_full = pd.concat([X_train_full, y_train_full], axis=1)

print('Model training is performing')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

for c in C:
    aucs = []

    for train_idx, val_idx in kfold.split(df_train_full):

        df_train = df_train_full.iloc[train_idx].reset_index()
        df_val = df_train_full.iloc[val_idx].reset_index()

        y_train = df_train.Churn.values
        y_val = df_val.Churn.values

        encoder, model = train(
            df_train, categorical_columns_names, numerical_columns_names, c)
        # y_pred = predict(df_val, numerical_columns_names, model)

        feature_names, encoded_df_val = data_encoder(
            df_val, categorical_columns_names, encoder)
        df_val_final = pd.concat([df_val[numerical_columns_names], pd.DataFrame(
            encoded_df_val, columns=feature_names)], axis=1)
        y_pred = predict(df_val_final, numerical_columns_names,
                         encoded_df_val, model)

        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

    print(f"c = {c}: auc = {np.mean(aucs):.3f} +- {np.std(aucs):.3f}")


print(
    f'Training the final model with c = {c_final} and saving model and encoder object')

encoder, model = train(df_train_full.reset_index(
), categorical_columns_names, numerical_columns_names, C=c_final)

with open('model_and_encoder.pkl', 'wb') as f_out:
    pickle.dump({'encoder': encoder, 'model': model}, f_out)
