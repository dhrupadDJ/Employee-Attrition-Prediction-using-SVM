import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def drop_columns(df, columns):
    df = df.drop(columns=columns, axis=1)
    return df

def create_dummies(df, columns):
    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df

def map_columns(df, mappings):
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)
    return df

def split_data(df, target_column, test_size=0.2, random_state=1, stratify=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    return x_train, x_test, y_train, y_test

def scale_data(x_train, x_test):
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)
    return x_train_scaled, x_test_scaled
