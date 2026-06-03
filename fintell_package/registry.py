import joblib

def save_model(model, tfidf, model_name=''):
    joblib.dump(model, f'../models/model_{model_name}.pkl')
    joblib.dump(tfidf, f'../models/tfidf.pkl')

def load_model(model_name=''):
    model = joblib.load(f'../models/model_{model_name}.pkl')
    tfidf = joblib.load(f'../models/tfidf.pkl')
    return model, tfidf
