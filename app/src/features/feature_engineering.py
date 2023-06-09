

def feature_engineering(train_df, test_df):
    """
        Función para encapsular la tarea de ingeniería de variables

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """
    train_df = create_domain_knowledge_features(train_df)
    test_df = create_domain_knowledge_features(test_df)

    return train_df.copy(), test_df.copy()


def create_domain_knowledge_features(df):
    """
        Función la creación de variables de contexto

        Args:
           df (DataFrame):  Dataset.
        Returns:
           DataFrame. Dataset.
    """
    # creación de variable Child de tipo booleana
    df['Child'] = 0
    df.loc[df.Age < 16, 'Child'] = 1
    return df.copy()
