import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from prometheus_client import start_http_server, Summary, Gauge

model = tf.keras.models.load_model("Inception_V3.h5")
class_names = ["Benign", "Malignant"]

REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
INFERENCE_COUNT = Gauge('inference_count', 'Number of inferences processed')

@REQUEST_TIME.time()
def predict(img):
    INFERENCE_COUNT.inc()
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    confidence_dict = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    return confidence_dict

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=2, label="Predicted Class"),
    title="Melanoma Classification with InceptionV3",
    description="Upload an image to classify it into one of the classes."
)

def start_metrics_server():
    start_http_server(8000)

if __name__ == "__main__":
    start_metrics_server()
    interface.launch()
