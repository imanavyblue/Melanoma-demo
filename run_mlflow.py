import my_mlflow_utils as mlflow_utils
import mlflow.keras

def main():
    # สมมติว่าคุณมีโมเดลและค่าที่ต้องการ
    model = tf.keras.models.load_model("Inception_V3.h5")  # โหลดหรือสร้างโมเดลที่นี่
    loss = 0.05  # ค่า loss ที่ประเมินได้
    accuracy = 0.95  # ค่า accuracy ที่ประเมินได้
    
    # บันทึกโมเดลและเมตริก
    mlflow_utils.log_model_and_metrics(model, loss, accuracy)

if __name__ == "__main__":
    main()
