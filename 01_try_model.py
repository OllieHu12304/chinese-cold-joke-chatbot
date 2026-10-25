from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx
import json
import platform
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

# 從本機資料夾載入模型權重與 tokenizer。
# 這是相對路徑，因此執行時要位於 nlp_hw2 資料夾。
# 此實驗使用 Qwen3-1.7B 的 MLX 4-bit 版本。
model_path = "models/Qwen3-1.7B-4bit"
model, tokenizer = load(model_path)

# 取得這支程式所在的資料夾。
project_dir = Path(__file__).resolve().parent

# 讀取 JSON 檔，轉成 Python 的 list；
# list 裡的每一題會變成一個 dict。
test_cases_path = project_dir / "test_cases.json"
with test_cases_path.open("r", encoding="utf-8") as file:
    test_cases = json.load(file)

# Python 的索引從 0 開始，因此 0 表示第一題。
case_index = 3
test_case = test_cases[case_index]

# 只將問題傳給模型，不傳入人工評分依據 criteria。
messages = [
    {
        "role": "user",
        "content": (
            "已知線索：\n"
            "1. 狗叫聲通常寫成「汪汪」。\n"
            "2.「旺旺」在這個笑話中表示生意興旺。\n"
            "3.「汪」與「旺」聲母、韻母相同，但聲調不同。\n\n"
            "冷笑話：狗狗為什麼適合當店長？因為牠能讓生意旺旺。\n"
            "請根據以上線索，用一到兩句話解釋笑點，不要加入品牌或其他故事。"
        ),
    },
]

print("測試題號：guided_explanation_01")
print(f"問題：{messages[0]['content']}")

settings = {
    "enable_thinking": False,
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
}

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=settings["enable_thinking"],
)

# 字典中的 temperature，對應 make_sampler 的 temp 參數。
sampler = make_sampler(
    temp=settings["temperature"],
    top_p=settings["top_p"],
    top_k=settings["top_k"],
)

mx.random.seed(settings["seed"])

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=settings["max_tokens"],
    sampler=sampler,
    verbose=False,
)

print("模型回答：")
print(response)

# ---------------------------------------------------
# 整理本次實驗：輸入、設定、原始回答和人工評語分開保存。
record = {
    "time": datetime.now().astimezone().isoformat(),
    "environment": {
        "python": platform.python_version(),
        "mlx": version("mlx"),
        "mlx_lm": version("mlx-lm"),
    },
    "model_path": model_path,
    # 保存題號與評分依據，方便之後對照結果。
    "test_case_id": "guided_explanation_01",
    "category": "提供線索後理解笑話",
    "criteria": "能連結狗叫聲汪汪與生意旺旺的近音關係，遵守一到兩句且不加入無關故事。",
    "messages": messages,
    "generation_settings": settings.copy(),
    "response": response,
    "human_notes": "",  # 人工閱讀後填寫，不讓模型替自己評分
}

# 將紀錄放在這支程式旁的 results 資料夾。
# parents=True：需要時一併建立上層資料夾。
# exist_ok=True：資料夾已存在也不報錯。
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)

log_path = results_dir / "experiments.jsonl"

# "a" 表示追加，不覆蓋舊紀錄；UTF-8 用來保存中文。
with log_path.open("a", encoding="utf-8") as file:
    # ensure_ascii=False：讓中文直接以中文字儲存。
    # 每筆紀錄末尾加換行，形成 JSONL。
    file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"實驗已保存：{log_path}")
