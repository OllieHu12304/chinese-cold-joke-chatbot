"""後續程式共用的路徑、資料驗證與 token 遮罩。

集中管理資料處理、訓練與推論共用的設定及邊界檢查。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"
MODEL_PATH = ROOT / "models" / "Qwen3-1.7B-4bit"
BEST_PATH = ROOT / "output" / "best_adapter"
SYSTEM = "你是冷笑話與諧音梗助手，請使用繁體中文回答。被追問時，請根據前文解釋或改寫。"
GENERATION = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 192, "seed": 42, "enable_thinking": False}


def write_json(path, value):
    """建立父資料夾並保存可閱讀的 UTF-8 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_messages(messages, label="教材"):
    """允許任意完整輪數，但拒絕缺欄位、空內容及錯誤角色。"""
    if not isinstance(messages, list) or len(messages) < 3 or len(messages) % 2 != 1:
        raise ValueError(f"{label} 必須包含 system 與完整 user/assistant 對話")
    for i, message in enumerate(messages):
        expected = "system" if i == 0 else "user" if i % 2 else "assistant"
        if not isinstance(message, dict) or message.get("role") != expected:
            raise ValueError(f"{label} 第 {i+1} 則訊息應為 {expected}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{label} 第 {i+1} 則訊息內容為空或不是文字")


def encode_sample(tokenizer, sample):
    """回傳 (整段 token, 答案起點)，只讓最後一則助手答案計算 loss。

    明確指定非思考模式及 return_dict=False，避免課堂遇到的版本差異。
    先驗證 prompt token 是完整教材的前綴，避免遮罩切錯位置。
    """
    messages = sample["messages"]
    options = dict(tokenize=True, return_dict=False, enable_thinking=False)
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
    prompt_tokens = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **options)
    offset = len(prompt_tokens)
    if tokens[:offset] != prompt_tokens or offset >= len(tokens):
        raise ValueError("對話模板的 prompt 與答案邊界不一致，不能安全地計算 loss")
    return tokens, offset


class AnswerDataset:
    """提供 MLX-LM CacheDataset 需要的介面；不修改安裝套件的原始碼。"""
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def process(self, row):
        return encode_sample(self.tokenizer, row)


def answer_mask(target_positions, lengths):
    """target 位置在 [答案起點, 真實長度)；不能把第一個 padding 算成答案。"""
    return (target_positions >= lengths[:, 0:1]) & (target_positions < lengths[:, 1:2])


def answer_loss(model, batch, lengths):
    """下一 token 交叉熵，只計最後回答，包含結束標記、不包含 padding。

    MLX-LM 0.32.2 的預設右邊界使用 <=，會多計第一個 padding；
    這裡明確使用半開區間，並用單元測試固定這個邊界。
    """
    import mlx.core as mx
    import mlx.nn as nn
    logits = model(batch[:, :-1])
    targets = batch[:, 1:]
    positions = mx.arange(1, batch.shape[1])[None, :]
    mask = answer_mask(positions, lengths)
    n_tokens = mask.sum()
    loss = nn.losses.cross_entropy(logits, targets).astype(mx.float32)
    return (loss * mask).sum() / n_tokens, n_tokens


def model_source():
    """從下載時留下的 metadata 讀 revision；不把本機路徑冒充 revision。"""
    metadata = MODEL_PATH / ".cache/huggingface/download/config.json.metadata"
    revision = metadata.read_text().splitlines()[0] if metadata.exists() else None
    return {"repo_id": MODEL_ID, "revision": revision, "local_path": str(MODEL_PATH.relative_to(ROOT)),
            "config_sha256": hashlib.sha256((MODEL_PATH / "config.json").read_bytes()).hexdigest()}


def load_chat_model(base=False):
    from mlx_lm import load
    if not (MODEL_PATH / "config.json").exists():
        raise FileNotFoundError("找不到基礎模型，請依 README 下載模型")
    if not base and not (BEST_PATH / "adapters.safetensors").exists():
        raise FileNotFoundError("找不到最佳 adapter，請先執行 05_train.py")
    return load(str(MODEL_PATH), adapter_path=None if base else str(BEST_PATH))


def chat_prompt(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
