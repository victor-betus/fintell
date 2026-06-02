import re
import spacy

# Chargement du modèle spaCy anglais, lourd, ne pas mettre dans la fonction
nlp = spacy.load(
    "en_core_web_sm", # accronyme pour : anglais / modèle principal / entraîné sur du texte web / small
    disable=["ner", "parser"] #retirer du packagage les personnes, lieux, organisations et la grammaire profonde
)


def lemmatiser_avis(texte: str) -> str:

    # Création du document spaCy
    doc = nlp(texte)

    # Lemmatisation
    lemmes = []

    for token in doc:
        lemmes.append(token.lemma_)

    # Reconstruction de la phrase sous forme de chaîne de caractères
    return " ".join(lemmes)

# Application de la fonction à la colonne review_text
df["review_text_lemma"] = df["review_text"][:10].apply(lemmatiser_avis)

# Affichage des premières lignes pour vérifier le résultat
df[["review_text", "review_text_lemma"]].head(10)
