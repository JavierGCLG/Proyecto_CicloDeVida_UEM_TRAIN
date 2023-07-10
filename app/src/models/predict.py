from app.src.data.predict.make_dataset import make_dataset
from mlflow.sklearn import load_model
from mlflow.artifacts import download_artifacts


def predict_pipeline(data):

    """
    Función para gestionar el pipeline completo de inferencia
    del modelo.

    Args:
        path (str):  Ruta hacia los datos.

    Returns:
        list. Lista con las predicciones hechas.
    """

    print("------> Loading artifacts from the model in Production from MLFlow")
    artifacts_path = load_artifacts()
    print(artifacts_path)

    # cargando y transformando los datos de entrada
    data_df = make_dataset(data, artifacts_path)

    print("------> Loading the model object in Production from MLFlow")
    model = load_production_model()

    print("------> Obtaining prediction")
    # realizando la inferencia con los datos de entrada
    return model.predict(data_df).tolist()


def load_production_model(model_name="titanic_model", stage="Production"):
    """
     Función para cargar el modelo de MLFlow

     Args:
         model_name (str):  Nombre del modelo registrado en MLFlow.
         stage (str): Estado del modelo en MLFlow

    Returns:
        obj. Objeto del modelo.
    """

    return load_model(model_uri=f"models:/{model_name}/{stage}")


def load_artifacts(model_name="titanic_model", stage="Production"):
    """
     Función para cargar el modelo de MLFlow

     Args:
         model_name (str):  Nombre del modelo registrado en MLFlow.
         stage (str): Estado del modelo en MLFlow

    Returns:
        str. Ruta local a los articacts
    """

    return download_artifacts(artifact_uri=f"models:/{model_name}/{stage}")
