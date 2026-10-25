"""建立正式資料：先按整段教材分組，再切成各個助手回答的前綴。"""
import hashlib
import json
import random
from transformers import AutoTokenizer
from hw2_utils import ROOT, MODEL_PATH, SYSTEM, encode_sample, validate_messages, write_json


def build():
    # 三份已經看過的教材固定歸訓練組，不拿來宣稱未見資料的表現。
    examples = []
    for name in ["example_conversation.json", "ice_conversation.json", "keyboard_conversation.json"]:
        row = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        examples.append({"id": name, "kind": "既有教學範例", **row})

    seeds_path = ROOT / "data/joke_seeds.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    if len({r[0] for r in seeds}) != len(seeds):
        raise ValueError("教材主題重複")
    new_examples = []
    for topic, kind, joke, explanation, followup, reply in seeds:
        messages = [{"role": "system", "content": SYSTEM}]
        turns = [(f"講一個關於{topic}的冷笑話或諧音梗。", joke), ("笑點在哪裡？", explanation), (followup, reply)]
        for question, answer in turns:
            messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        new_examples.append({"id": topic, "kind": kind, "messages": messages})

    # 固定亂數，以完整笑話為分組單位；不是先展開樣本再隨機分配。
    random.Random(42).shuffle(new_examples)
    groups = {"train": examples + new_examples[10:], "valid": new_examples[5:10], "test": new_examples[:5]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    limit = json.loads((ROOT / "configs/train.json").read_text())["max_seq_length"]
    summary = {"seed": 42, "source": "由 AI 撰寫並整理的合成對話，無外部抓取語料", "splits": {}}
    split_ids = []
    for split, conversations in groups.items():
        rows, lengths, answer_lengths = [], [], []
        for conversation in conversations:
            validate_messages(conversation["messages"], conversation["id"])
            for i, message in enumerate(conversation["messages"]):
                if message["role"] == "assistant":
                    row = {"source_id": conversation["id"], "messages": conversation["messages"][:i+1]}
                    tokens, offset = encode_sample(tokenizer, row)
                    if len(tokens) > limit:
                        raise ValueError(f"{conversation['id']} 超過 {limit} tokens，請調整資料而非默默截斷")
                    rows.append(row)
                    lengths.append(len(tokens))
                    answer_lengths.append(len(tokens)-offset)
        path = ROOT / "data/splits" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
        ids = {c["id"] for c in conversations}
        split_ids.append(ids)
        summary["splits"][split] = {"conversations": len(conversations), "samples": len(rows), "source_ids": sorted(ids),
            "max_tokens": max(lengths), "min_answer_tokens": min(answer_lengths), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("不同資料組出現相同教材來源")
    write_json(ROOT / "data/corpus.json", [c for group in groups.values() for c in group])
    write_json(ROOT / "results/data_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
