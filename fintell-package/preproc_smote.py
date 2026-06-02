from imblearn.over_sampling import SMOTE
import pandas as pd

def pourcentage_labels(df: pd.DataFrame,
                       colonne: str):

    """
    Display the distribution of sentiment labels in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the labels.

    colonne : str
        Name of the column containing sentiment labels.

    Returns
    -------
    None
        Prints the number and percentage of samples for each class.
    """

    total = int(df[colonne].value_counts().sum())
    counts = df[colonne].value_counts()

    nb_positif = int(counts.iloc[0])
    positif = round(int(counts.iloc[0])/total * 100,2)

    nb_negatif = int(counts.iloc[1])
    negatif = round(int(counts.iloc[1])/total * 100,2)

    nb_neutral = int(counts.iloc[2])
    neutral = round(int(counts.iloc[2])/total * 100,2)

    return print(f"Nombre d'avis : {total} - 100%\nPositif : {nb_positif} - {positif}%\nNégatif : {nb_negatif} - {negatif}%\nNeutral : {nb_neutral} - {neutral}%")


def oversampling_smote(X_train : pd.DataFrame,
                       y_train : pd.DataFrame,
                       random_state : int,
                       k_neighbors : int):

    """
    Apply SMOTE oversampling on the training dataset.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features after preprocessing/vectorization.

    y_train : pd.DataFrame
        Target labels associated with X_train.

    random_state : int
        Random seed used to ensure reproducibility.

    k_neighbors : int
        Number of nearest neighbors used by SMOTE to generate
        synthetic samples.

    Returns
    -------
    X_train_smote : pd.DataFrame
        Resampled training features.

    y_train_smote : pd.DataFrame
        Resampled training labels.
    """


    smote = SMOTE(
        sample_strategy='not_majority',
        random_state = random_state,
        k_neighbors = k_neighbors
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    print("Over sampling completed ✅")

    return X_train_smote, y_train_smote
