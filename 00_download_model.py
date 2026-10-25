"""下載與本次實驗相同 revision 的模型，避免遠端更新後難以重現。"""
from huggingface_hub import snapshot_download
from hw2_utils import MODEL_ID, MODEL_PATH

# Hugging Face revision 是版本識別碼；固定它比只寫模型名稱更精確。
REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"

if __name__ == "__main__":
    snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=MODEL_PATH)
    print(f"模型已下載：{MODEL_PATH}")
