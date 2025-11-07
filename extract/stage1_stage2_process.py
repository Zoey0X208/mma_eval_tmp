"""
第一阶段和第二阶段处理脚本
从old_extracted中读取原文Reference，完成第一阶段和第二阶段处理
"""
import os
import sys
import json
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from extract.chat import chat_infer
from extract.prompts import prompt_1_semantic_check, prompt_1_title_clean, prompt_2_experiment_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 禁用HTTP请求相关的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_json_response(response: str) -> Dict[str, Any]:
    """解析LLM返回的JSON响应"""
    try:
        # 移除可能的markdown代码块标记
        response = response.replace("```json", "").replace("```", "").strip()
        # 尝试找到JSON部分
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            logger.warning(f"无法找到JSON格式: {response[:100]}")
            return {}
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {e}, 响应内容: {response[:200]}")
        return {}


def stage1_semantic_check(reference_text: str, llm_type: str = "gpt-4o") -> Dict[str, Any]:
    """
    第一阶段：判断是否包含正常语义信息
    
    Returns:
        {
            "has_semantic_info": "是" or "否",
            "reason": "判断理由",
            "should_delete": bool  # 如果否，则应该删除发明点
        }
    """
    prompt = prompt_1_semantic_check.replace("__REFERENCE_TEXT__", reference_text)
    
    try:
        response = chat_infer(prompt, LLM_Type=llm_type)
        result = parse_json_response(response)
        
        has_semantic = result.get("has_semantic_info", "否")
        should_delete = (has_semantic == "否")
        
        return {
            "has_semantic_info": has_semantic,
            "reason": result.get("reason", ""),
            "should_delete": should_delete
        }
    except Exception as e:
        logger.error(f"第一阶段语义检查失败: {e}")
        return {
            "has_semantic_info": "否",
            "reason": f"处理错误: {e}",
            "should_delete": True
        }


def stage1_title_clean(reference_text: str, llm_type: str = "gpt-4o") -> str:
    """
    第一阶段：标题清洗，返回正文
    
    Returns:
        清洗后的正文内容
    """
    prompt = prompt_1_title_clean.replace("__REFERENCE_TEXT__", reference_text)
    
    try:
        body_text = chat_infer(prompt, LLM_Type=llm_type)
        # 清理可能的markdown格式
        body_text = body_text.strip()
        return body_text
    except Exception as e:
        logger.error(f"第一阶段标题清洗失败: {e}")
        return reference_text  # 失败时返回原文


def stage2_experiment_check(body_text: str, llm_type: str = "gpt-4o") -> Dict[str, Any]:
    """
    第二阶段：判断是否为实验步骤
    
    Returns:
        {
            "is_experiment": "是" or "否",
            "reason": "判断理由",
            "output_type": "实验步骤" or "原文Reference"
        }
    """
    prompt = prompt_2_experiment_check.replace("__BODY_TEXT__", body_text)
    
    try:
        response = chat_infer(prompt, LLM_Type=llm_type)
        result = parse_json_response(response)
        
        is_experiment = result.get("is_experiment", "否")
        output_type = "实验步骤" if (is_experiment == "是") else "原文Reference"
        
        return {
            "is_experiment": is_experiment,
            "reason": result.get("reason", ""),
            "output_type": output_type
        }
    except Exception as e:
        logger.error(f"第二阶段实验步骤检查失败: {e}")
        return {
            "is_experiment": "否",
            "reason": f"处理错误: {e}",
            "output_type": "原文Reference"
        }


def process_single_item(item: Dict[str, Any], llm_type: str = "gpt-4o") -> Dict[str, Any]:
    """
    处理单个知识点/发明点项，完成第一阶段和第二阶段
    
    Args:
        item: 包含"原文Reference"的字典项
        
    Returns:
        简化的处理结果字典，包含：
        - 原文Reference
        - 清洗后的正文
        - 是否为实验步骤
    """
    reference_text = item.get("原文Reference", "")
    
    if not reference_text:
        return {
            "原文Reference": "",
            "清洗后的正文": "",
            "是否为实验步骤": "否",
            "状态": "已删除（原文Reference为空）"
        }
    
    # 第一阶段：语义检查
    stage1_semantic = stage1_semantic_check(reference_text, llm_type)
    
    if stage1_semantic["should_delete"]:
        return {
            "原文Reference": reference_text,
            "清洗后的正文": "",
            "是否为实验步骤": "否",
            "状态": "已删除（不包含正常语义信息）"
        }
    
    # 第一阶段：标题清洗
    body_text = stage1_title_clean(reference_text, llm_type)
    
    # 第二阶段：实验步骤检查
    stage2_experiment = stage2_experiment_check(body_text, llm_type)
    is_experiment = "是" if stage2_experiment["is_experiment"] == "是" else "否"
    
    return {
        "原文Reference": reference_text,
        "清洗后的正文": body_text,
        "是否为实验步骤": is_experiment
    }


def process_old_extracted_file(
    json_path: str,
    output_dir: str = "./tmp/stage_results",
    llm_type: str = "gpt-4o",
    max_workers: int = 6,
) -> Dict[str, Any]:
    """
    处理old_extracted JSON文件，完成第一阶段和第二阶段
    
    Args:
        json_path: old_extracted JSON文件路径
        output_dir: 输出目录
        llm_type: LLM类型
        max_workers: 最大并发数
        
    Returns:
        处理结果
    """
    logger.info(f"读取文件: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 提取所有知识点和潜在发明点
    all_items = []
    
    # 处理"文档中的知识点"
    knowledge_points = data.get("文档中的知识点", [])
    for item in knowledge_points:
        all_items.append({
            "source": "文档中的知识点",
            "item": item
        })
    
    # 处理"文档中潜在发明点"
    invention_points = data.get("文档中潜在发明点", [])
    for item in invention_points:
        all_items.append({
            "source": "文档中潜在发明点",
            "item": item
        })
    
    logger.info(f"总共找到 {len(all_items)} 个条目")
    
    def process_with_index(idx, source_item):
        """带索引的处理函数"""
        result = process_single_item(source_item["item"], llm_type)
        result["index"] = idx
        result["source"] = source_item["source"]
        return result
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_with_index, idx, item): idx
            for idx, item in enumerate(all_items)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理条目", unit="项"):
            result = future.result()
            results.append(result)
    
    # 按索引排序
    results.sort(key=lambda x: x["index"])
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_stage12_result.json")
    
    # 统计信息
    deleted_count = sum(1 for r in results if r.get("状态", "").startswith("已删除"))
    processed_count = len(results) - deleted_count
    experiment_count = sum(1 for r in results if r.get("是否为实验步骤") == "是")
    
    logger.info(f"保存结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": json_path,
            "total_items": len(results),
            "deleted_items": deleted_count,
            "processed_items": processed_count,
            "experiment_items": experiment_count,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"处理完成：总计 {len(results)} 项，已处理 {processed_count} 项，已删除 {deleted_count} 项，实验步骤 {experiment_count} 项")
    return {
        "total": len(results),
        "deleted": deleted_count,
        "processed": processed_count,
        "experiment": experiment_count,
        "results": results
    }


def process_all_old_extracted(
    input_dir: str = "./test_data/old_extracted",
    output_dir: str = "./tmp/stage_results",
    llm_type: str = "gpt-4o",
    max_workers: int = 6,
):
    """
    批量处理old_extracted目录下的所有JSON文件
    """
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    logger.info(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        logger.info(f"\n处理文件: {json_file}")
        try:
            process_old_extracted_file(
                json_path=json_path,
                output_dir=output_dir,
                llm_type=llm_type,
                max_workers=max_workers
            )
        except Exception as e:
            logger.error(f"处理文件 {json_file} 失败: {e}")


if __name__ == "__main__":
    # 示例：处理单个文件
    process_old_extracted_file(
        json_path="./test_data/old_extracted/P2023110253CN1.json",
        output_dir="./tmp/stage_results",
        llm_type="gpt-4o",
        max_workers=6
    )
    
    # 或者批量处理所有文件
    # process_all_old_extracted(
    #     input_dir="./test_data/old_extracted",
    #     output_dir="./tmp/stage_results",
    #     llm_type="gpt-4o",
    #     max_workers=6
    # )

