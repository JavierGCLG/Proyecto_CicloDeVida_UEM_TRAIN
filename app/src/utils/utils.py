from mlflow.pyfunc import PythonModel
import pickle
from app import ROOT_DIR


class SerializableModel(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def save_object_locally(obj, name):
    # objeto serializado
    with open(f"{ROOT_DIR}/models/objects/" + name + ".pkl", "wb") as outp:
        pickle.dump(obj, outp)
