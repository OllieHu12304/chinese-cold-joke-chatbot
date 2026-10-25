import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"

# 明確列出教材，避免誤讀其他用途的 JSON 檔。
source_files = [
    "example_conversation.json",
    "ice_conversation.json",
    "keyboard_conversation.json",
]

all_samples = []

for filename in source_files:
    data_path = data_dir / filename

    with data_path.open("r", encoding="utf-8") as file:
        conversation = json.load(file)

    messages = conversation["messages"]

    # 至少需要 system，加上一組 user、assistant。
    if len(messages) < 3:
        raise ValueError(f"{filename} 至少需要一輪完整對話")

    if messages[0]["role"] != "system":
        raise ValueError(f"{filename} 必須以 system 開頭")

    # 跳過第一則 system，檢查後續角色是否交替。
    # 這裡的 index 從 1 開始，對應原始 messages 的索引。
    for index, message in enumerate(messages[1:], start=1):
        # 奇數索引應是 user，偶數索引應是 assistant。
        expected_role = "user" if index % 2 == 1 else "assistant"

        if message["role"] != expected_role:
            raise ValueError(f"{filename} 第 {index + 1} 則訊息應為 {expected_role}")

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"{filename} 必須以 assistant 結尾")

    for number, message in enumerate(messages, start=1):
        if not message["content"].strip():
            raise ValueError(f"{filename} 的第 {number} 則訊息為空")

    sample_count = 0

    # 每則助手回答各產生一筆前綴樣本。
    for index, message in enumerate(messages):
        if message["role"] == "assistant":
            sample = {
                # 記錄原始教材，之後依來源分組切分。
                "source_file": filename,
                "messages": messages[: index + 1],
            }
            all_samples.append(sample)
            sample_count += 1

    print(f"{filename}：檢查通過，產生 {sample_count} 筆樣本")

print(f"\n共讀取 {len(source_files)} 份教材")
print(f"共產生 {len(all_samples)} 筆樣本")


# 先保存整理結果；這還不是正式的訓練集。
output_path = data_dir / "prepared_samples.jsonl"

with output_path.open("w", encoding="utf-8") as file:
    for sample in all_samples:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"整理結果已保存：{output_path}")
