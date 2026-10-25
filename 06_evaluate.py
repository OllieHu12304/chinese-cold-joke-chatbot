"""比較同一設定下的原始模型與最佳 adapter；保存原文，不用 loss 代替品質評分。"""
import argparse
import json
import math
from datetime import datetime

import mlx.core as mx
import numpy as np
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import evaluate
from hw2_utils import (ROOT, SYSTEM, GENERATION, BEST_PATH, AnswerDataset,
                       answer_loss, chat_prompt, load_chat_model, model_source, read_jsonl, write_json)


def run(base):
    mode = "base" if base else "adapted"
    model, tokenizer = load_chat_model(base)
    config = json.loads((ROOT / "configs/train.json").read_text())
    np.random.seed(42)
    dataset = CacheDataset(AnswerDataset(read_jsonl(ROOT / "data/splits/test.jsonl"), tokenizer))
    # test 僅在訓練完成後評估；所有 15 筆都參與，沒有丟掉最後一筆。
    test_loss = float(evaluate(model, dataset, 1, -1, config["max_seq_length"], loss=answer_loss))
    result = {"mode": mode, "time": datetime.now().astimezone().isoformat(), "model": model_source(),
        "adapter": None if base else str(BEST_PATH.relative_to(ROOT)), "generation_settings": GENERATION,
        "test_loss": test_loss, "test_perplexity": math.exp(test_loss), "test_samples": len(dataset),
        "development": [], "heldout": [], "conversations": []}
    sampler = make_sampler(temp=GENERATION["temperature"], top_p=GENERATION["top_p"], top_k=GENERATION["top_k"])

    def answer(messages, seed=42):
        mx.random.seed(seed)
        return generate(model, tokenizer, prompt=chat_prompt(tokenizer, messages),
                        max_tokens=GENERATION["max_tokens"], sampler=sampler, verbose=False)

    # 原有四題僅是開發題；狗狗案例已進訓練，不能宣稱泛化改善。
    development = json.loads((ROOT / "test_cases.json").read_text())
    for case in development:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]
        result["development"].append({**case, "messages": messages, "response": answer(messages),
                                      "human_notes": "待判讀；開發題可能與教材重疊"})
        print(f"{mode}: {case['id']} 完成", flush=True)

    # 每段測試教材取「講笑話」和「解釋」各一筆。
    # 解釋題使用教材前文（teacher forcing），與下方自由聊天分開記錄。
    for row in read_jsonl(ROOT / "data/splits/test.jsonl"):
        if len(row["messages"]) > 5:
            continue
        messages = row["messages"][:-1]
        result["heldout"].append({"source_id": row["source_id"], "messages": messages,
            "reference": row["messages"][-1]["content"], "response": answer(messages),
            "history_type": "教材前文；不是模型自由生成的前文", "human_notes": "待判讀"})
        print(f"{mode}: heldout {row['source_id']} turn={len(messages)//2} 完成", flush=True)

    scenarios = [
        {"id": "seen_dog", "scope": "訓練主題，觀察記憶及追問", "questions":
         ["講一個狗狗的諧音冷笑話。", "笑點在哪裡？", "保留剛才的笑點，改成一句話。"]},
        {"id": "heldout_umbrella", "scope": "未參與訓練的雨傘主題", "questions":
         ["講一個關於雨傘的冷笑話。", "笑點在哪裡？", "保留剛才的笑點，改成一句話。"]},
        {"id": "unseen_bus", "scope": "教材外的新主題", "questions":
         ["講一個關於公車的冷笑話。", "笑點在哪裡？", "保留剛才的笑點，改成一句話。"]},
    ]
    for scenario in scenarios:
        messages = [{"role": "system", "content": SYSTEM}]
        for i, question in enumerate(scenario["questions"]):
            messages.append({"role": "user", "content": question})
            # 後續輪次真正使用模型自己剛才的回答，不偷換成標準答案。
            response = answer(messages, seed=42+i)
            messages.append({"role": "assistant", "content": response})
        result["conversations"].append({**scenario, "messages": messages, "human_notes": "待判讀"})
        print(f"{mode}: 自由三輪 {scenario['id']} 完成", flush=True)
    write_json(ROOT / f"results/evaluation_{mode}.json", result)
    print(f"{mode}: test_loss={test_loss:.4f} perplexity={math.exp(test_loss):.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="不載入 adapter，作為公平比較的基準")
    run(parser.parse_args().base)
