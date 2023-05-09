
import pickle
import pandas as pd


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


def predict(df, numerical_columns_names, encoded_df, model):

    numerical_columns = df[numerical_columns_names]

    X = pd.concat([numerical_columns, encoded_df], axis=1)
    y_pred = model.predict_proba(X)[0, 1]

#     null_cols = X.columns[X.isnull().any()]
#     print(X[null_cols].isnull().sum())

    return y_pred


numerical_columns_names = ['Tenure', 'Monthly_Charges', 'Total_Charges']

categorical_columns_names = ['Gender', 'Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service',
                             'Multiple_Lines', 'Internet_Service', 'Online_Security',
                             'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV',
                             'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method']


with open('model_and_encoder.pkl', 'rb') as f_in:

    data = pickle.load(f_in)

    encoder = data['encoder']
    model = data['model']


customer_new = {
    'customerid': '8879-zkjof',
    'gender': 'Female',
    'SeniorCitizen': '0',
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'One year',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Bank transfer (automatic)',
    'tenure': 41,
    'MonthlyCharges': 79.85,
    'TotalCharges': 3320.75,
}


df_new_customer = dataframe_maker(customer_new)
df_new_customer.columns = column_name_corrector(df_new_customer)
dtypes_corrector(df_new_customer)
space_corrector(df_new_customer)
feature_names, encoded_new_data = data_encoder(
    df_new_customer, categorical_columns_names, encoder)

customer_data_final = pd.concat([df_new_customer[numerical_columns_names], pd.DataFrame(
    encoded_new_data, columns=feature_names)], axis=1)
churn_probability = predict(customer_data_final, numerical_columns_names, encoded_new_data, model)
print(f"The probability of the churn for this customer is {churn_probability * 100:.2f} %")
