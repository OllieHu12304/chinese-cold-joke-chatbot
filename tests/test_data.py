"""測容易造成錯誤結論的資料邊界，不只檢查程式能不能執行。"""
import json
import unittest
import numpy as np
from transformers import AutoTokenizer
from project_utils import ROOT, MODEL_PATH, answer_mask, encode_sample, read_jsonl, validate_messages


class DataChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def test_groups_do_not_leak(self):
        groups = [{r["source_id"] for r in read_jsonl(ROOT / f"data/splits/{s}.jsonl")}
                  for s in ("train", "valid", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertIn("example_conversation.json", groups[0])

    def test_prompt_is_prefix_and_answer_not_empty(self):
        for split in ("train", "valid", "test"):
            for row in read_jsonl(ROOT / f"data/splits/{split}.jsonl"):
                validate_messages(row["messages"])
                tokens, offset = encode_sample(self.tokenizer, row)
                self.assertGreater(offset, 0)
                self.assertGreater(len(tokens), offset)
                self.assertLessEqual(len(tokens), 512)
                # 第一個目標字元確實來自助手內容，不是空思考區塊。
                answer_text = self.tokenizer.decode(tokens[offset:])
                self.assertTrue(answer_text.startswith(row["messages"][-1]["content"]))

    def test_padding_and_prompt_not_scored(self):
        # 真實 token 索引為 0..5，答案起點為 3；只能計 3、4、5。
        positions = np.arange(1, 9)[None, :]
        mask = answer_mask(positions, np.array([[3, 6]]))
        self.assertEqual(positions[mask].tolist(), [3, 4, 5])

    def test_invalid_roles_and_empty_content_rejected(self):
        messages = json.loads((ROOT / "data/keyboard_conversation.json").read_text())["messages"]
        broken = [dict(m) for m in messages]
        broken[-1]["role"] = "user"
        with self.assertRaises(ValueError):
            validate_messages(broken)
        broken[-1] = {"role": "assistant", "content": "  "}
        with self.assertRaises(ValueError):
            validate_messages(broken)


if __name__ == "__main__":
    unittest.main()
