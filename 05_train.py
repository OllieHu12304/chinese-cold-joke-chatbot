"""本機 QLoRA：凍結 4-bit 基礎模型，只更新少量 LoRA 參數。

先執行 python 05_train.py --smoke，確認 GPU 與資料可用；
再執行 python 05_train.py，正式訓練並依完整驗證集選最佳權重。
"""
import argparse
import json
import math
import platform
import time
from importlib.metadata import version

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters
from project_utils import ROOT, MODEL_PATH, AnswerDataset, answer_loss, model_source, read_jsonl, write_json


class ExperimentLog(TrainingCallback):
    """每次回報保存數值；只用 valid 選權重，完全不讀 test。"""
    def __init__(self, model, valid, config, directory, metrics_file):
        self.model, self.valid, self.config = model, valid, config
        self.directory, self.metrics_file = directory, metrics_file
        self.best_loss, self.best_step = float("inf"), None
        self.rows = []

    def on_train_loss_report(self, info):
        row = {"kind": "train", **info}
        if not math.isfinite(info["train_loss"]):
            raise RuntimeError("訓練 loss 非有限值，停止訓練")
        self.rows.append(row)
        print(f"step={info['iteration']} train_loss={info['train_loss']:.4f} peak_GB={info['peak_memory']:.2f}", flush=True)
        if info["iteration"] % self.config["steps_per_eval"] == 0 or info["iteration"] == self.config["iters"]:
            # batch_size=1 確保 15 筆驗證樣本全部使用，不因奇數筆而丟棄一筆。
            val_loss = float(evaluate(self.model, self.valid, batch_size=1, num_batches=-1,
                                      max_seq_length=self.config["max_seq_length"], loss=answer_loss))
            if not math.isfinite(val_loss):
                raise RuntimeError("驗證 loss 非有限值，停止訓練")
            self.rows.append({"kind": "valid", "iteration": info["iteration"], "val_loss": val_loss})
            print(f"step={info['iteration']} valid_loss={val_loss:.4f}", flush=True)
            if val_loss < self.best_loss:
                self.best_loss, self.best_step = val_loss, info["iteration"]
                # 只存可訓練的 adapter，避免複製整個基礎模型。
                mx.save_safetensors(str(self.directory / "adapters.safetensors"),
                                    dict(tree_flatten(self.model.trainable_parameters())))
                write_json(self.directory / "selection.json", {"step": self.best_step, "valid_loss": self.best_loss})
            self.model.train()
        write_json(self.metrics_file, self.rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="只跑 5 steps，輸出至獨立資料夾")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/train.json").read_text())
    if args.smoke:
        config.update(iters=5, steps_per_report=5, steps_per_eval=5, save_every=5)
    name = "smoke" if args.smoke else "training"
    directory = ROOT / "output" / ("smoke/best_adapter" if args.smoke else "best_adapter")
    checkpoints = ROOT / "output" / ("smoke/checkpoints" if args.smoke else "checkpoints")
    # 避免重跑時不小心覆蓋已完成的訓練；需要重跑時先另行備份輸出。
    if (directory / "adapters.safetensors").exists():
        raise SystemExit(f"{directory} 已有權重，請先將它改名備份，再重新訓練")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    mx.random.seed(config["seed"])
    np.random.seed(config["seed"])
    model, tokenizer = load(str(MODEL_PATH))
    datasets = {}
    for split in ("train", "valid"):
        rows = read_jsonl(ROOT / f"data/splits/{split}.jsonl")
        ds = CacheDataset(AnswerDataset(rows, tokenizer))
        # 先編碼全部樣本並檢查長度；不允許截斷後漏掉答案。
        for i in range(len(ds)):
            if len(ds[i][0]) > config["max_seq_length"]:
                raise ValueError("教材過長，請重跑 04_build_corpus.py 檢查資料")
        datasets[split] = ds

    initial_valid = float(evaluate(model, datasets["valid"], 1, -1, config["max_seq_length"], loss=answer_loss))
    print(f"base valid_loss={initial_valid:.4f}", flush=True)
    # freeze 不代表模型不運算，而是原始權重不接受梯度更新。
    model.freeze()
    linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
    print_trainable_parameters(model)
    adapter_config = {"fine_tune_type": "lora", "num_layers": config["num_layers"],
                      "lora_parameters": config["lora_parameters"]}
    write_json(directory / "adapter_config.json", adapter_config)
    write_json(checkpoints / "adapter_config.json", adapter_config)
    logger = ExperimentLog(model, datasets["valid"], config, directory, ROOT / f"results/{name}_metrics.json")
    training_args = TrainingArgs(batch_size=config["batch_size"], iters=config["iters"],
        steps_per_report=config["steps_per_report"], steps_per_save=config["save_every"],
        max_seq_length=config["max_seq_length"], grad_checkpoint=True,
        adapter_file=str(checkpoints / "adapters.safetensors"))
    start = time.perf_counter()
    train(model, optim.Adam(learning_rate=config["learning_rate"]), datasets["train"],
          args=training_args, training_callback=logger, loss=answer_loss)
    elapsed = time.perf_counter() - start
    summary = {"mode": name, "config": config, "model": model_source(), "python": platform.python_version(),
        "versions": {p: version(p) for p in ["mlx", "mlx-lm", "transformers", "numpy"]},
        "seconds": elapsed, "peak_memory_gb": mx.get_peak_memory()/1e9,
        "initial_valid_loss": initial_valid, "best_valid_loss": logger.best_loss, "best_step": logger.best_step,
        "data_manifest": json.loads((ROOT / "results/data_summary.json").read_text())}
    write_json(ROOT / f"results/{name}_summary.json", summary)
    print(f"完成，最佳權重：{directory}，耗時 {elapsed:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()
