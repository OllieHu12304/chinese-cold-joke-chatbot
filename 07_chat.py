"""終端機多輪聊天。範例：python 07_chat.py；加 --base 可使用原始模型。"""
import argparse
import json
from datetime import datetime

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from project_utils import ROOT, SYSTEM, GENERATION, chat_prompt, load_chat_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="使用未微調模型")
    args = parser.parse_args()
    model, tokenizer = load_chat_model(args.base)
    messages = [{"role": "system", "content": SYSTEM}]
    sampler = make_sampler(temp=GENERATION["temperature"], top_p=GENERATION["top_p"], top_k=GENERATION["top_k"])
    mx.random.seed(GENERATION["seed"])
    print("冷笑話助手｜/reset 清除歷史｜/exit 結束｜回答僅供娛樂")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n聊天結束。")
            break
        if question == "/exit":
            break
        if question == "/reset":
            messages = [{"role": "system", "content": SYSTEM}]
            print("歷史已清除。")
            continue
        if not question:
            continue
        pending = messages + [{"role": "user", "content": question}]
        # 保留 system 與最近完整輪次；避免無限累積歷史吃掉記憶體。
        while len(tokenizer.encode(chat_prompt(tokenizer, pending))) > 4096:
            if len(pending) <= 2:
                print("這則訊息太長，請縮短後再試。")
                pending = None
                break
            del pending[1:3]
        if pending is None:
            continue
        print("助手：", end="", flush=True)
        response = ""
        try:
            for part in stream_generate(model, tokenizer, prompt=chat_prompt(tokenizer, pending),
                                         max_tokens=GENERATION["max_tokens"], sampler=sampler):
                response += part.text
                print(part.text, end="", flush=True)
        except KeyboardInterrupt:
            print("\n生成已中止，本輪不加入歷史。")
            continue
        print()
        messages = pending + [{"role": "assistant", "content": response}]
        # 原始輸出不加工，保存本輪實際傳入的上下文供日後分析。
        path = ROOT / "results/chat_sessions.jsonl"
        path.parent.mkdir(exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"time": datetime.now().astimezone().isoformat(), "base": args.base,
                "settings": GENERATION, "messages": messages}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
