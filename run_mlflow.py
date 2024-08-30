import os
import gdown
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
import tensorflow as tf

def download_folder_from_google_drive(folder_url, destination):
    gdown.download(folder_url, destination, quiet=False, fuzzy=True)

def create_model():
    input_shape = (224, 224, 3)
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)

    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.optimizers.SGD(learning_rate=0.0001),
        metrics=['accuracy', 'Recall']  # เพิ่ม Recall เป็น metrics
    )

    return model

def train_and_log_model(model, train_data, val_data):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
    )

    loss, accuracy, recall = model.evaluate(val_data)
    mlflow_utils.log_model_and_metrics(model, loss, accuracy, recall)

def main():
    # URL ของโฟลเดอร์จาก Google Drive
    folder_url = 'https://drive.google.com/drive/folders/1KilXUDLq23K8ttEw33fjzikq0B6E4R57?usp=sharing'
    destination = 'data/'

    # สร้างโฟลเดอร์ปลายทางถ้ายังไม่มี
    os.makedirs(destination, exist_ok=True)


    # ดาวน์โหลดข้อมูลจาก Google Drive
    download_folder_from_google_drive(folder_url, destination)

    # กำหนดไดเรกทอรีข้อมูลการฝึกอบรมและ validation
    train_dir = os.path.join(destination, 'train_data')
    val_dir = os.path.join(destination, 'validation_data')

    # โหลดข้อมูล
    train_data, val_data = load_data(train_dir, val_dir)

    # สร้างโมเดล
    model = create_model()

    # ฝึกอบรมและบันทึกโมเดล
    train_and_log_model(model, train_data, val_data)

if __name__ == "__main__":
    main()
