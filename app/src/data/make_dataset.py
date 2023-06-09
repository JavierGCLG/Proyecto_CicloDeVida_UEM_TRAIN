import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from ..features.feature_engineering import feature_engineering
from app import cos


def make_dataset(path, timestamp, target, cols_to_remove, model_type='RandomForest'):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           path (str):  Ruta hacia los datos.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    print('---> Getting data')
    df = get_raw_data_from_local(path)
    print('---> Train / test split')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=50)
    print('---> Transforming data')
    train_df, test_df = transform_data(train_df, test_df, timestamp, target, cols_to_remove)
    print('---> Feature engineering')
    train_df, test_df = feature_engineering(train_df, test_df)
    print('---> Preparing data for training')
    train_df, test_df = pre_train_data_prep(train_df, test_df, model_type, timestamp, target)

    return train_df.copy(), test_df.copy()


def get_raw_data_from_local(path):

    """
        Función para obtener los datos originales desde local

        Args:
           path (str):  Ruta hacia los datos.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """

    df = pd.read_csv(path)
    return df.copy()


def transform_data(train_df, test_df, timestamp, target, cols_to_remove):

    """
        Función que permite realizar las primeras tareas de transformación
        de los datos de entrada.

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.
           timestamp (float):  Representación temporal en segundos.
           target (str):  Variable dependiente a usar.
           cols_to_remove (list): Columnas a retirar.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    # Quitando columnas no usables
    print('------> Removing unnecessary columns')
    train_df = remove_unwanted_columns(train_df, cols_to_remove)
    test_df = remove_unwanted_columns(test_df, cols_to_remove)

    # Quitando valores nulos en la variable objetivo
    print('------> Removing missing targets')
    train_df = remove_missing_targets(train_df, target)
    test_df = remove_missing_targets(test_df, target)

    # cambio de tipo
    train_df['Pclass'] = train_df['Pclass'].astype(str)
    test_df['Pclass'] = test_df['Pclass'].astype(str)


    # Separamos la variable objetivo antes del encoding
    train_target = train_df[target].copy()
    test_target = test_df[target].copy()
    train_df.drop(columns=[target], inplace=True)
    test_df.drop(columns=[target], inplace=True)

    # generación de dummies
    print('------> Encoding data')
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)
    # alineación de train y test para tener las mismas columnas
    train_df, test_df = train_df.align(test_df, join='inner', axis=1)

    # guardando las columnas resultantes en IBM COS
    print('---------> Saving encoded columns')
    cos.save_object_in_cos(train_df.columns, 'encoded_columns', timestamp)

    # volvemos a unir la variable objetivo a los datasets
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_target.reset_index(drop=True, inplace=True)
    test_target.reset_index(drop=True, inplace=True)
    train_df = train_df.join(train_target)
    test_df = test_df.join(test_target)

    return train_df.copy(), test_df.copy()


def pre_train_data_prep(train_df, test_df, model_type, timestamp, target):
    """
        Función que realiza las últimas transformaciones sobre los datos
        antes del entrenamiento (imputación de nulos y escalado)

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.
           model_type (str):  Tipo de modelo a usar.
           timestamp (float):  Representación temporal en segundos
           target (str):  Variable dependiente a usar.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    # Separamos la variable objetivo antes de la imputación y escalado
    train_target = train_df[target].copy()
    test_target = test_df[target].copy()
    train_df.drop(columns=[target], inplace=True)
    test_df.drop(columns=[target], inplace=True)

    # imputación de nulos
    print('------> Inputing missing values')
    train_df, test_df = input_missing_values(train_df, test_df, timestamp)

    # restringimos el escalado solo a ciertos modelos
    if model_type.upper() in ['SVM', 'KNN', 'NaiveBayes']:
        print('------> Scaling features')
        train_df, test_df = scale_data(train_df, test_df)

    # volvemos a unir la variable objetivo a los datasets
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    train_target.reset_index(drop=True, inplace=True)
    test_target.reset_index(drop=True, inplace=True)
    train_df = train_df.join(train_target)
    test_df = test_df.join(test_target)

    return train_df.copy(), test_df.copy()


def input_missing_values(train_df, test_df, timestamp):
    """
        Función para la imputación de nulos

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.
           timestamp (float):  Representación temporal en segundos.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """
    # creamos el imputador que usará la mediana como sustitutivo
    imputer = SimpleImputer(strategy='median')

    # ajustamos las medianas en base a los datos de train
    train_df = pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    # imputamos los datos de test
    test_df = pd.DataFrame(imputer.transform(test_df), columns=test_df.columns)

    # guardamos el imputador para futuros nuevos datos
    print('------> Saving imputer on the cloud')
    cos.save_object_in_cos(imputer, 'imputer', timestamp)

    return train_df.copy(), test_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
        Función para quitar variables innecesarias

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)


def remove_missing_targets(df, target):
    """
        Función para quitar los valores nulos en la variable objetivo

        Args:
           df (DataFrame):  Dataset.

        Returns:
           DataFrame. Dataset.
    """
    return df[~df[target].isna()].copy()


def scale_data(train_df, test_df):
    """
        Función para el escalado de variables

        Args:
           train_df (DataFrame):  Dataset de train.
           test_df (DataFrame):  Dataset de test.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    # objeto de escalado en el rango (0,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # ajuste y escalado en datos de train
    train_df = scaler.fit_transform(train_df)
    # escalado de datos de test
    test_df = scaler.transform(test_df)

    return train_df.copy(), test_df.copy()


