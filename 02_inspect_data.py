import json
from pathlib import Path

# 從程式所在位置尋找資料，不受終端機目前位置影響。
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "example_conversation.json"

# 將 JSON 物件讀成 Python 字典。
with data_path.open("r", encoding="utf-8") as file:
    conversation = json.load(file)

# 取出 messages：它是一個串列，每個元素是一則訊息。
messages = conversation["messages"]

# 檢查這筆教材是否符合預期的兩輪對話順序。
expected_roles = ["system", "user", "assistant", "user", "assistant"]
actual_roles = [message["role"] for message in messages]

if actual_roles != expected_roles:
    raise ValueError(f"角色順序不符合預期：{actual_roles}")

# 逐則檢查並印出；enumerate 的 start=1 讓顯示編號從 1 開始。
for number, message in enumerate(messages, start=1):
    if not message["content"].strip():
        raise ValueError(f"第 {number} 則訊息內容為空")

    print(f"{number}. [{message['role']}]")
    print(message["content"])
    print()

print("檢查通過：這筆資料包含一則 system 訊息與兩輪對話。")


from transformers import AutoTokenizer

# 只載入 tokenizer，不載入模型權重。
# tokenizer 同時提供這個模型使用的對話模板。
tokenizer = AutoTokenizer.from_pretrained(project_dir / "models" / "Qwen3-1.7B-4bit")

# 教材已包含完整的助手答案，因此不需要再加上
# 「接下來輪到助手回答」的生成提示。
formatted_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False,
)

print("\n模型使用的對話格式：")
print(formatted_text)

# 將同一段教材轉成 token 編號，觀察實際長度。
token_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    enable_thinking=False,
    return_dict=False,
)

print(f"\n教材的 token 數：{len(token_ids)}")


# 每遇到一則助手回答，就保存「從開頭到這則回答」的對話。
# 這樣搭配 mask_prompt，便能分別學習每一則助手回答。
training_samples = []

for index, message in enumerate(messages):
    if message["role"] == "assistant":
        sample = {"messages": messages[: index + 1]}
        training_samples.append(sample)

# 印出每筆樣本的角色順序，以及將用來計算 loss 的答案。
print("\n切分後的訓練樣本：")

for number, sample in enumerate(training_samples, start=1):
    sample_messages = sample["messages"]
    roles = [message["role"] for message in sample_messages]

    # 每筆樣本都以助手回答結尾；-1 代表串列最後一個元素。
    target_answer = sample_messages[-1]["content"]

    print(f"\n樣本 {number}")
    print(f"角色順序：{roles}")
    print(f"學習目標：{target_answer}")

print(f"\n共產生 {len(training_samples)} 筆訓練樣本。")


# 將切分結果存成 JSONL：每一行是一筆完整訓練樣本。
output_path = project_dir / "data" / "example_samples.jsonl"

# "w" 表示覆寫：重跑時重新產生，避免重複追加相同教材。
with output_path.open("w", encoding="utf-8") as file:
    for sample in training_samples:
        line = json.dumps(sample, ensure_ascii=False)
        file.write(line + "\n")

print(f"\n訓練樣本已保存：{output_path}")
