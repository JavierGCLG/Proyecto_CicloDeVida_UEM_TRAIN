from app.src.data.train.make_dataset import make_dataset
from app.src.evaluation.evaluate_model import evaluate_model
from app.src.utils.utils import load_model_config
from app import ROOT_DIR
from sklearn.ensemble import RandomForestClassifier
import time
import mlflow
from app import ROOT_DIR


def training_pipeline(path):
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
    train_df, test_df = make_dataset(path, target, cols_to_remove)

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
    # Se inicia el RUN (ejecución del entrenamiento de un modelo)
    with mlflow.start_run(run_name=f"{model_config['model_name']}_{str(ts)}") as run:
        # Ajuste del modelo con los datos de entrenamiento
        model.fit(X_train, y_train)
        print("------> Logging metadata in MLFlow")
        # se loguean los parámetros del modelo
        mlflow.log_param("n_estimators", model_config["n_estimators"])
        mlflow.log_param("max_features", model_config["max_features"])
        mlflow.log_param("target", target)
        mlflow.log_param("cols_to_remove", cols_to_remove)
        evaluate_model(model, X_test, y_test, ts, model_config["model_name"])
        # guardado del modelo y artifacts en MLFlow
        print(
            f"------> Saving the model {model_config['model_name']}_{str(ts)} and artifacts in MLFlow"
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
        artifact_path="model",
    )
    mlflow.log_artifact(f"{ROOT_DIR}/models/objects/imputer.pkl", artifact_path="model")

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
    )
