from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
from datetime import datetime


def evaluate_model(model, X_test, y_test, timestamp, model_name):
    """
        Esta función permite realizar una evaluación del modelo entrenado
        y crear un diccionario con toda la información relevante del mismo

        Args:
           model (sklearn-object):  Objecto del modelo entrenado.
           X_test (DataFrame): Variables independientes en test.
           y_test (Series):  Variable dependiente en test.
           timestamp (float):  Representación temporal en segundos.
           model_name (str):  Nombre del modelo

        Returns:
           dict. Diccionario con la info del modelo
    """

    # obtener predicciones usando el modelo entrenado
    y_pred = model.predict(X_test)

    # extraer la importancia de variables
    feature_importance_values = model.feature_importances_

    # Nombre de variables
    features = list(X_test.columns)
    
    # creación del diccionario de info del modelo
    model_info = {}
    #fi_df = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # info general del modelo
    model_info['_id'] = 'model_' + str(int(timestamp))
    model_info['name'] = 'model_' + str(int(timestamp))
    # fecha de entrenamiento (dd/mm/YY-H:M:S)
    model_info['date'] = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    model_info['model_used'] = model_name
    # objectos usados en el modelo (encoders, imputer)
    model_info['objects'] = {}
    model_info['objects']['encoders'] = 'encoded_columns_'+str(int(timestamp))
    model_info['objects']['imputer'] = 'imputer_' + str(int(timestamp))
    # métricas usadas
    model_info['model_metrics'] = {}
    # model_info['model_metrics']['feature_importances'] = dict(zip(fi_df.area, fi_df.importance))
    model_info['model_metrics']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    model_info['model_metrics']['accuracy_score'] = accuracy_score(y_test, y_pred)
    model_info['model_metrics']['precision_score'] = precision_score(y_test, y_pred)
    model_info['model_metrics']['recall_score'] = recall_score(y_test, y_pred)
    model_info['model_metrics']['f1_score'] = f1_score(y_test, y_pred)
    model_info['model_metrics']['roc_auc_score'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # status del modelo (en producción o no)
    model_info['status'] = "none"

    return model_info



