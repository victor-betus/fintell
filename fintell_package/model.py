import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_name=None):

    if model_name == 'svc':
        model = LinearSVC()

    elif model_name == 'random_forest':
        model = RandomForestClassifier()

    elif model_name == 'naive_bayes':
        model = MultinomialNB(alpha=0.01)

    else:
        model = LogisticRegression()

    model.fit(X_train, y_train)

    return model

def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    return accuracy, f1, report

def predict_model(X_new, model):
    y_pred_new = model.predict(X_new)
    return y_pred_new
