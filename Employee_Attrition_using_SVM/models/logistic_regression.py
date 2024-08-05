from sklearn.linear_model import LogisticRegression

def train_logistic_regression(x_train, y_train):
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    return model.predict(x_test)
