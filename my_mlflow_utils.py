import my_mlflow_utils as mlflow_utils
import mlflow.keras

def log_model_and_metrics(model, loss, accuracy):
    mlflow.start_run()
    mlflow.keras.log_model(model, "model")
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.end_run()
