# evaluate_our_method.py

import os
import json
import numpy as np
from tqdm import tqdm
import re

from config import COMMON_HUMAN_DIR, OUR_METHOD_OUTPUT_DIR

HUMAN_DIR = COMMON_HUMAN_DIR
MODEL_DIR = OUR_METHOD_OUTPUT_DIR
METHOD_NAME = "Ours"
BAD_CASES_DIR = "./bad_cases/precision"  # ← bad case 保存路径
os.makedirs(BAD_CASES_DIR, exist_ok=True)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_significant_match(str1, str2):
    """
    str1: 人工标注
    str2: 模型输出
    判断是否有显著匹配：str1 中存在长度 ≥ max(10, len(str1)//2) 的子串出现在 str2 中
    """
    len1, len2 = len(str1), len(str2)
    len3 = 10
    if len1 < len3 or len2 < len3:
        return (str1 == str2) or (str1 in str2)
    min_len1 = max(len3, len1 // 2)
    for i in range(len(str1) - min_len1 + 1):
        substr = str1[i:i + min_len1]
        if substr in str2:
            return True
    return False


def evaluate_our_method_performance(human_annotation, model_output, filename=None):
    """适配我们方法的字段结构，并返回 bad cases"""
    human_invention = [preprocess_text(ref) for ref in human_annotation.get("invention_points", [])]
    all_human_refs = human_invention

    model_refs = []
    model_raw_items = []
    for item in model_output.get("文档中潜在发明点", []):
        ref = preprocess_text(item.get("原文Reference", ""))
        if ref:
            model_refs.append(ref)
            model_raw_items.append(item)  # 保留原始 item，便于保存上下文

    if not all_human_refs or not model_refs:
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "human_refs_count": len(all_human_refs),
            "model_refs_count": len(model_refs),
            "matched_human_count": 0,
            "matched_model_count": 0,
            "precision_bad_cases": []  # 无 bad case
        }

    # 召回率：人工是否被模型覆盖
    recalled_human = [False] * len(all_human_refs)
    for i, human_text in enumerate(all_human_refs):
        for model_text in model_refs:
            if is_significant_match(human_text, model_text):
                recalled_human[i] = True
                break

    matched_human_count = sum(recalled_human)
    recall = matched_human_count / len(all_human_refs)

    # 准确率：模型输出是否匹配人工
    correct_model = [False] * len(model_refs)
    for j, model_text in enumerate(model_refs):
        for human_text in all_human_refs:
            if is_significant_match(human_text, model_text):
                correct_model[j] = True
                break

    matched_model_count = sum(correct_model)
    precision = matched_model_count / len(model_refs)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 收集 precision bad cases: 模型输出但未匹配人工
    bad_cases = []
    for j, is_correct in enumerate(correct_model):
        if not is_correct:
            bad_cases.append({
                "model_output_index": j,
                "原文Reference": model_raw_items[j].get("原文Reference", ""),
                "其他字段": {k: v for k, v in model_raw_items[j].items() if k != "原文Reference"}
            })

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "human_refs_count": len(all_human_refs),
        "model_refs_count": len(model_refs),
        "matched_human_count": matched_human_count,
        "matched_model_count": matched_model_count,
        "precision_bad_cases": bad_cases
    }


def run_our_method_evaluation():
    """运行我们的方法评测，并保存 bad cases"""
    if not os.path.exists(HUMAN_DIR):
        raise FileNotFoundError(f"人工标注目录不存在: {HUMAN_DIR}")
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"我们的方法输出目录不存在: {MODEL_DIR}")

    human_files = {f for f in os.listdir(HUMAN_DIR) if f.endswith('.json')}
    model_files = {f for f in os.listdir(MODEL_DIR) if f.endswith('.json')}
    common_files = human_files & model_files

    if not common_files:
        print(f"⚠️ 无匹配文件，跳过 {METHOD_NAME}")
        return {}

    results = {}
    recalls, precisions, f1s = [], [], []

    for filename in tqdm(common_files, desc=f"评估 {METHOD_NAME}"):
        try:
            with open(os.path.join(HUMAN_DIR, filename), 'r', encoding='utf-8') as f:
                human_ann = json.load(f)
            with open(os.path.join(MODEL_DIR, filename), 'r', encoding='utf-8') as f:
                model_out = json.load(f)
            metrics = evaluate_our_method_performance(human_ann, model_out, filename=filename)
            results[filename] = metrics

            # 保存 bad cases（仅当有 bad case 时）
            bad_cases = metrics.get("precision_bad_cases", [])
            if bad_cases:
                bad_case_file = os.path.join(BAD_CASES_DIR, filename)
                with open(bad_case_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "file": filename,
                        "human_invention_points": human_ann.get("invention_points", []),
                        "model_bad_cases": bad_cases
                    }, f, ensure_ascii=False, indent=2)

            recalls.append(metrics["recall"])
            precisions.append(metrics["precision"])
            f1s.append(metrics["f1"])
        except Exception as e:
            print(f"处理 {filename} 出错: {e}")
            error_metrics = {
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "human_refs_count": 0,
                "model_refs_count": 0,
                "matched_human_count": 0,
                "matched_model_count": 0,
                "precision_bad_cases": []
            }
            results[filename] = error_metrics
            recalls.append(0.0)
            precisions.append(0.0)
            f1s.append(0.0)

    # 汇总统计
    total_human_refs = sum(r["human_refs_count"] for r in results.values())
    total_model_refs = sum(r["model_refs_count"] for r in results.values())
    total_matched_human = sum(r["matched_human_count"] for r in results.values())
    total_matched_model = sum(r["matched_model_count"] for r in results.values())

    overall_recall = total_matched_human / total_human_refs if total_human_refs > 0 else 0.0
    overall_precision = total_matched_model / total_model_refs if total_model_refs > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    avg_metrics = {
        "avg_recall": np.mean(recalls) if recalls else 0.0,
        "avg_precision": np.mean(precisions) if precisions else 0.0,
        "avg_f1": np.mean(f1s) if f1s else 0.0,
        "total_documents": len(common_files),
        "total_human_refs": total_human_refs,
        "total_model_refs": total_model_refs,
        "total_matched_human": total_matched_human,
        "total_matched_model": total_matched_model,
    }

    result = {METHOD_NAME: avg_metrics}

    print(f"\n✅ {METHOD_NAME} 评测完成:")
    print(f"  → 评估文档数量: {avg_metrics['total_documents']}")
    print(f"  → 人工标注总数: {avg_metrics['total_human_refs']}")
    print(f"  → 模型抽取总数: {avg_metrics['total_model_refs']}")
    print(f"  → 匹配的人工标注数: {avg_metrics['total_matched_human']}")
    print(f"  → 匹配的模型抽取数: {avg_metrics['total_matched_model']}")
    print(f"  → 平均召回率: {avg_metrics['avg_recall']:.4f}")
    print(f"  → 平均准确率: {avg_metrics['avg_precision']:.4f}")
    print(f"  → 平均F1: {avg_metrics['avg_f1']:.4f}")
    print(f"  → Precision bad cases 已保存至: {BAD_CASES_DIR}")

    return result