from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow


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
    # fi_df = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

    # Se loguean los tags asociados al modelo en MLFlow
    # info general del modelo
    mlflow.set_tag("_id", "model_" + str(int(timestamp)))
    mlflow.set_tag("model_name", model_name)

    # # Se loguean las métricas asociadas al modelo en MLFlow
    # model_info['model_metrics']['feature_importances'] = dict(zip(fi_df.area, fi_df.importance))
    mlflow.log_metric("accuracy_score", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.log_metric("recall_score", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    mlflow.log_metric(
        "roc_auc_score", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    )
