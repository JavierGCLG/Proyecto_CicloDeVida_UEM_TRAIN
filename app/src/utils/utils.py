import pickle
from yaml import safe_load
from pathlib import Path

from app import ROOT_DIR


def save_object_locally(obj, name):
    """
    Función para guardar objeto en local.

    Args:
        obj (bin): Objeto binario a guardar.
        name (str). Nombre del archivo
    """
    # creating a new directory called objects
    Path(f"{ROOT_DIR}/models/objects").mkdir(parents=True, exist_ok=True)
    # objeto serializado
    with open(f"{ROOT_DIR}/models/objects/{name}.pkl", "wb") as outp:
        pickle.dump(obj, outp)


def load_model_config(path="models/config/model_config.yaml"):
    """
    Función para cargar la info del modelo desde local.

    Args:
        path (str):  Ruta local al archivo.
    """
    with open(f"{ROOT_DIR}/{path}", "r") as stream:
        return safe_load(stream)
