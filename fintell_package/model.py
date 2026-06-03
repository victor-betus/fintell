import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score

# def train_model(X_train, y_train, model_name='Naive_bayes'):
#     if model_name == 'SVC'
#         pass
#     else:
#         model = MultinomialNB()


#     model.fit(X_train, y_train)
#     return model



# def evaluate_model():
#     pass

# def predict_model():
#     pass
