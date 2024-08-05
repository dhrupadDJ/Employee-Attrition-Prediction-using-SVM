from sklearn.svm import SVC

def train_svm(x_train, y_train, kernel='linear', degree=3):
    model = SVC(kernel=kernel, degree=degree)
    model.fit(x_train, y_train)
    return model

def predict(model, x_test):
    return model.predict(x_test)
