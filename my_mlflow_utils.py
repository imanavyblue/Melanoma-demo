import mlflow
import mlflow.keras

def log_model_and_metrics(model, loss, accuracy, recall):
    mlflow.start_run()
    mlflow.keras.log_model(model, "model")
    mlflow.log_param("learning_rate", 0.0001)  # ค่า learning_rate ที่ใช้
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_recall", recall)  # บันทึกค่า recall
    mlflow.end_run()
