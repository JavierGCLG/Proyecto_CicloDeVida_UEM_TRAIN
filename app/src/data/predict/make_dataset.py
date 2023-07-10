import pandas as pd
from app.src.features.feature_engineering import feature_engineering
from app import init_cols
from app.src.utils.utils import load_model_config
import pickle


def make_dataset(data, artifacts_path):

    """
    Función que permite crear el dataset usado para el entrenamiento
    del modelo.

    Args:
       data (List):  Lista con la observación llegada por request.
       artifacts_path (str):  Ruta local a los artefactos del modelo

    Returns:
       DataFrame. Dataset a inferir.
    """
    model_info = load_model_config()
    print("---> Getting data")
    data_df = get_raw_data_from_request(data)
    print("---> Transforming data")
    data_df = transform_data(data_df, artifacts_path, model_info["cols_to_remove"])
    print("---> Feature engineering")
    data_df = feature_engineering(data_df)
    print("---> Preparing data for training")
    data_df = pre_train_data_prep(data_df, artifacts_path)

    return data_df.copy()


def get_raw_data_from_request(data):

    """
    Función para obtener nuevas observaciones desde request

    Args:
       data (List):  Lista con la observación llegada por request.

    Returns:
       DataFrame. Dataset con los datos de entrada.
    """
    return pd.DataFrame(data, columns=init_cols)


def transform_data(data_df, artifacts_path, cols_to_remove):
    """
    Función que permite realizar las primeras tareas de transformación
    de los datos de entrada.

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo
        cols_to_remove (list): Columnas a retirar.

    Returns:
       DataFrame. Dataset transformado.
    """

    print("------> Removing unnecessary columns")
    data_df = remove_unwanted_columns(data_df, cols_to_remove)

    data_df["Pclass"] = data_df["Pclass"].astype(str)

    # creando dummies originales
    print("------> Encoding data")
    print("---------> Getting encoded columns from MLFlow")
    # obteniendo las columnas presentes en el entrenamiento desde MLFlow
    with open(f"{artifacts_path}/encoded_columns.pkl", "rb") as inp:
        enc_cols = pickle.load(inp)
    # columnas dummies generadas en los datos de entrada
    data_df = pd.get_dummies(data_df)

    # agregando las columnas dummies faltantes en los datos de entrada
    data_df = data_df.reindex(columns=enc_cols, fill_value=0)

    return data_df.copy()


def pre_train_data_prep(data_df, artifacts_path):

    """
    Función que realiza las últimas transformaciones sobre los datos
    antes del entrenamiento (imputación de nulos)

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo.

    Returns:
        DataFrame. Datasets de salida.
    """
    data_df = input_missing_values(data_df, artifacts_path)

    return data_df.copy()


def input_missing_values(data_df, artifacts_path):

    """
    Función para la imputación de nulos

    Args:
        data_df (DataFrame):  Dataset de entrada.
        artifacts_path (str):  Ruta local a los artefactos del modelo

    Returns:
        DataFrame. Datasets de salida.
    """

    print("------> Inputing missing values")
    # obtenemos el objeto SimpleImputer desde MLFlow
    print("------> Getting imputer from MLFlow")
    with open(f"{artifacts_path}/imputer.pkl", "rb") as inp:
        imputer = pickle.load(inp)
    data_df = pd.DataFrame(imputer.transform(data_df), columns=data_df.columns)

    return data_df.copy()


def remove_unwanted_columns(df, cols_to_remove):
    """
    Función para quitar variables innecesarias

    Args:
       df (DataFrame):  Dataset.
       cols_to_remove: List(srt). Columnas a eliminar.

    Returns:
       DataFrame. Dataset.
    """
    return df.drop(columns=cols_to_remove)
