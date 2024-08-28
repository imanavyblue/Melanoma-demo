import mlflow
import mlflow.pyfunc

def main():
    # กำหนดพารามิเตอร์ที่ต้องการ
    parameters = {"learning_rate": 0.0001}

    # รันโปรเจกต์ MLflow โดยใช้พารามิเตอร์ที่กำหนด
    mlflow.run(".", entry_point="main", parameters=parameters)

if __name__ == "__main__":
    main()
