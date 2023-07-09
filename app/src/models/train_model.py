from ..data.make_dataset import make_dataset
from ..evaluation.evaluate_model import evaluate_model
from ..utils.utils import SerializableModel
from app import ROOT_DIR
from sklearn.ensemble import RandomForestClassifier
import time
from yaml import safe_load
import mlflow
from app import ROOT_DIR


def training_pipeline(path, model_info_db_name="models-db"):
    """
    Función para gestionar el pipeline completo de entrenamiento
    del modelo.

    Args:
        path (str):  Ruta hacia los datos.

    Kwargs:
        model_info_db_name (str):  base de datos a usar para almacenar
        la info del modelo.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config()
    # variable dependiente a usar
    target = model_config["target"]
    # columnas a retirar
    cols_to_remove = model_config["cols_to_remove"]

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    train_df, test_df = make_dataset(path, ts, target, cols_to_remove)

    # separación de variables independientes y dependiente
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target]).copy()
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target]).copy()

    # definición del modelo (Random Forest)
    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_features=model_config["max_features"],
        random_state=50,
        n_jobs=-1,
    )

    print("---> Training a model with the following configuration:")
    print(model_config)
    with mlflow.start_run(run_name=f"{model_config['model_name']}_{str(ts)}") as run:
        # Ajuste del modelo con los datos de entrenamiento
        model.fit(X_train, y_train)
        print("------> Logging metadata in MLFlow")
        mlflow.log_param("n_estimators", model_config["n_estimators"])
        mlflow.log_param("max_features", model_config["max_features"])
        mlflow.log_param("target", target)
        mlflow.log_param("cols_to_remove", cols_to_remove)
        evaluate_model(model, X_test, y_test, ts, model_config["model_name"])
        # guardado del modelo en IBM COS
        print(
            f"------> Saving the model {model_config['model_name']}_{str(ts)} in MLFlow"
        )
        save_model(model)


def save_model(model):
    """
    Función para loguear el modelo en MLFlow

    Args:
        obj (sklearn-object): Objeto de modelo entrenado.
    """

    mlflow.log_artifact(
        f"{ROOT_DIR}/models/objects/encoded_columns.pkl",
    )
    mlflow.log_artifact(
        f"{ROOT_DIR}/models/objects/imputer.pkl",
    )

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SerializableModel(
            model=model,
        ),
    )


def load_model_config(path="models/config/model_config.yaml"):
    """
    Función para cargar la info del modelo desde IBM Cloudant.

    Args:
        db_name (str):  Nombre de la base de datos.

    Returns:
        dict. Documento con la configuración del modelo.
    """
    with open(f"{ROOT_DIR}/{path}", "r") as stream:
        return safe_load(stream)
