from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def stopwords(df):
    """avec un dataframe en input, découpe les strings de la colonne 'review_text' en tokens,
    enlève les stopwords,
    les re-join pour obtenir une liste en output.
    """

    ### Gère mal :
    ###   - les apostrophes
    ###   - les apostrophes les majuscules
    ### => devrait disparaitre avec le cleaning

    separateur = ' '
    df['review_text'] = df['review_text'].apply(word_tokenize).apply(lambda tokens: [w for w in tokens if w not in stop_words]).apply(separateur.join)
    return df
