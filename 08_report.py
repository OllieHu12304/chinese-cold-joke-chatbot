"""將實際訓練與評估 JSON 轉成可閱讀的報告；不改寫模型回答。"""
import json
from hw2_utils import ROOT


def report():
    training = json.loads((ROOT / "results/training_summary.json").read_text())
    evaluations = {mode: json.loads((ROOT / f"results/evaluation_{mode}.json").read_text())
                   for mode in ["base", "adapted"]}
    lines = ["# 微調結果與原始對話", "", "本檔由 `python 08_report.py` 根據實際結果產生。",
        "所有回答均為原文；內容品質分析與評估方法見 `quality_review.md`。", "",
        "## 訓練", "", f"- 模型：{training['model']['repo_id']}",
        f"- revision：`{training['model']['revision']}`", f"- 步數：{training['config']['iters']}",
        f"- 最佳 step：{training['best_step']}（只依驗證集選擇）",
        f"- 訓練耗時：{training['seconds']:.1f} 秒",
        f"- MLX 尖峰記憶體：{training['peak_memory_gb']:.2f} GB（不是整台電腦的總用量）", "",
        "## Loss", "", "| 指標 | 原始模型 | 最佳 adapter |", "|---|---:|---:|",
        f"| valid loss | {training['initial_valid_loss']:.4f} | {training['best_valid_loss']:.4f} |",
        f"| test loss | {evaluations['base']['test_loss']:.4f} | {evaluations['adapted']['test_loss']:.4f} |",
        f"| test perplexity | {evaluations['base']['test_perplexity']:.2f} | {evaluations['adapted']['test_perplexity']:.2f} |", "",
        "Loss 是已知教材前文下，最後回答 token 的交叉熵；不是笑話正確率或幽默分數。", "",
        "## Loss 歷程", "", "| step | train loss（回報區間平均） | valid loss |", "|---:|---:|---:|"]
    metrics = json.loads((ROOT / "results/training_metrics.json").read_text())
    val = {r["iteration"]: r["val_loss"] for r in metrics if r["kind"] == "valid"}
    for row in metrics:
        if row["kind"] == "train":
            v = f"{val[row['iteration']]:.4f}" if row["iteration"] in val else "—"
            lines.append(f"| {row['iteration']} | {row['train_loss']:.4f} | {v} |")
    for mode, result in evaluations.items():
        label = "原始模型" if mode == "base" else "微調模型"
        lines += ["", f"## {label}：自由多輪展示", "", "這些前文全部使用模型自己生成的回答，不注入教材答案。"]
        for scenario in result["conversations"]:
            lines += ["", f"### {scenario['id']}", "", scenario["scope"], ""]
            for message in scenario["messages"][1:]:
                who = "使用者" if message["role"] == "user" else "助手"
                lines += [f"**{who}**", "", message["content"], ""]
        lines += [f"## {label}：原有開發題", "", "此處使用相同 system prompt，與早期 user-only 紀錄不直接做數值比較。",
                  "狗狗題已是訓練主題，改善不能視為未見題目的泛化。", ""]
        for case in result["development"]:
            lines += [f"### {case['id']}", "", case["prompt"], "", case["response"], ""]
        lines += [f"## {label}：未參與訓練的教材題", "", "解釋題使用教材中的前文，屬 teacher forcing；不是自由多輪對話。", ""]
        for row in result["heldout"]:
            lines += [f"### {row['source_id']} / {len(row['messages'])//2}", "", "問題："+row["messages"][-1]["content"],
                      "", "參考答案："+row["reference"], "", "模型回答：", "", row["response"], ""]
    (ROOT / "results/report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("報告已保存：results/report.md")


if __name__ == "__main__":
    report()
