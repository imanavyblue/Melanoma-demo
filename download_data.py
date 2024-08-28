import os
import gdown

def download_folder_from_google_drive(folder_url, destination):
    gdown.download(folder_url, destination, quiet=False, fuzzy=True)

if __name__ == "__main__":
    folder_url = 'https://drive.google.com/drive/folders/1j_hlGK9tKr88h9W_Du2X6p8yKhk8fepb?usp=sharing'  # URL ของโฟลเดอร์จาก Google Drive
    destination = 'data/'  # โฟลเดอร์ปลายทางที่ต้องการบันทึก
    os.makedirs(destination, exist_ok=True)
    download_folder_from_google_drive(folder_url, destination)
