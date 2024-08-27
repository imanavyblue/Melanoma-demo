from huggingface_hub import HfApi, HfFolder

api = HfApi()
model_path = "Inception_V3.h5"
repo_id = "Suphawan/Melanoma-3"

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="Inception_V3.h5",
    repo_id=repo_id,
    token=HfFolder.get_token()
)
