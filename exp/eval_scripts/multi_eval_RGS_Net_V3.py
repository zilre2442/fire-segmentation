#!/usr/bin/env python3
"""
批量评估 RGS-Net V3 模型的脚本

使用方法:
    # 评估 multi_run 目录下的所有训练结果
    CUDA_VISIBLE_DEVICES=1 python3 exp/eval_scripts/multi_eval_RGS_Net_V3.py
    
    # 评估指定的 experiments_summary.json
    CUDA_VISIBLE_DEVICES=1,2 python3 exp/eval_scripts/multi_eval_RGS_Net_V3.py --summary-file output/RGS_Net_V3/multi_run/experiments_summary.json

功能:
1. 自动读取批量训练的汇总文件
2. 依次评估每个训练好的模型
3. 生成每个模型的评估报告
4. 生成对比表格，便于比较不同超参数的效果
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import argparse


# ============================================================================
# 配置区域
# ============================================================================

# 默认汇总文件路径
DEFAULT_SUMMARY_FILE = "output/RGS_Net_V3/multi_run/experiments_summary.json"

# 评估脚本路径
EVAL_SCRIPT = "exp/eval_scripts/eval_RGS_Net_V3.py"

# 评估配置
BATCH_SIZE = 64
NUM_WORKERS = 4
ALGORITHM = "voting"
BANDS = (7, 6, 5)


# ============================================================================
# 辅助函数
# ============================================================================

def log_message(message: str, level: str = "INFO"):
    """记录日志信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    print(log_line)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="批量评估 RGS-Net V3 模型")
    parser.add_argument(
        "--summary-file",
        type=str,
        default=DEFAULT_SUMMARY_FILE,
        help="训练汇总文件路径 (experiments_summary.json)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="评估批大小"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="数据加载线程数"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=ALGORITHM,
        help="数据集算法标签"
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="跳过可视化生成以加快评估速度"
    )
    return parser.parse_args()


def load_summary(summary_file: str) -> Dict[str, Any]:
    """加载训练汇总文件"""
    if not os.path.exists(summary_file):
        log_message(f"汇总文件不存在: {summary_file}", level="ERROR")
        sys.exit(1)
    
    try:
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)
        log_message(f"成功加载汇总文件: {summary_file}")
        return summary
    except Exception as e:
        log_message(f"加载汇总文件失败: {e}", level="ERROR")
        sys.exit(1)


def check_model_exists(output_dir: str) -> bool:
    """检查模型权重文件是否存在"""
    model_path = os.path.join(output_dir, "weights", "model_best.pth")
    return os.path.exists(model_path)


def run_evaluation(
    output_dir: str,
    exp_name: str,
    batch_size: int,
    num_workers: int,
    algorithm: str
) -> Dict[str, Any]:
    """
    执行单个模型的评估
    
    Args:
        output_dir: 模型输出目录
        exp_name: 实验名称
        batch_size: 批大小
        num_workers: 工作线程数
        algorithm: 算法标签
    
    Returns:
        评估结果字典
    """
    log_message("=" * 80)
    log_message(f"开始评估: {exp_name}")
    log_message(f"模型目录: {output_dir}")
    log_message("=" * 80)
    
    # 检查模型是否存在
    if not check_model_exists(output_dir):
        log_message(f"模型权重不存在，跳过评估", level="WARNING")
        return {
            "exp_name": exp_name,
            "output_dir": output_dir,
            "success": False,
            "error": "模型权重文件不存在",
            "metrics": {}
        }
    
    # 构建评估命令
    # 设置完整的环境变量来避免分布式初始化错误
    env = os.environ.copy()
    env["RANK"] = "0"
    env["WORLD_SIZE"] = "1"
    env["LOCAL_RANK"] = "0"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = "29500"
    
    # 移除可能导致分布式初始化的环境变量
    for key in ["TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS"]:
        env.pop(key, None)
    
    cmd = [
        "python3",
        EVAL_SCRIPT,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--save-dir", output_dir,
        "--algo", algorithm,
    ]
    
    log_message(f"执行命令: {' '.join(cmd)}")
    
    # 执行评估
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        log_message(f"评估完成: {exp_name}")
        
        # 解析评估结果
        metrics = parse_evaluation_results(output_dir)
        
        return {
            "exp_name": exp_name,
            "output_dir": output_dir,
            "success": True,
            "error": None,
            "metrics": metrics
        }
    except subprocess.CalledProcessError as e:
        log_message(f"评估失败: {e}", level="ERROR")
        log_message(f"标准输出: {e.stdout}", level="ERROR")
        log_message(f"标准错误: {e.stderr}", level="ERROR")
        return {
            "exp_name": exp_name,
            "output_dir": output_dir,
            "success": False,
            "error": str(e),
            "metrics": {}
        }
    except Exception as e:
        log_message(f"评估异常: {e}", level="ERROR")
        return {
            "exp_name": exp_name,
            "output_dir": output_dir,
            "success": False,
            "error": str(e),
            "metrics": {}
        }


def parse_evaluation_results(output_dir: str) -> Dict[str, float]:
    """
    从评估报告中解析结果
    
    Args:
        output_dir: 模型输出目录
    
    Returns:
        包含评估指标的字典
    """
    report_path = os.path.join(output_dir, "eval_model_best.txt")
    
    if not os.path.exists(report_path):
        log_message(f"评估报告不存在: {report_path}", level="WARNING")
        return {}
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        metrics = {}
        for line in content.split("\n"):
            if "精确率:" in line or "Precision:" in line:
                metrics["precision"] = float(line.split(":")[-1].strip())
            elif "召回率:" in line or "Recall:" in line:
                metrics["recall"] = float(line.split(":")[-1].strip())
            elif "F1分数:" in line or "F1:" in line:
                metrics["f1"] = float(line.split(":")[-1].strip())
        
        return metrics
    except Exception as e:
        log_message(f"解析评估报告失败: {e}", level="WARNING")
        return {}


def save_comparison_report(
    eval_results: List[Dict[str, Any]],
    training_summary: Dict[str, Any],
    output_file: str
):
    """
    保存对比报告
    
    Args:
        eval_results: 评估结果列表
        training_summary: 训练汇总信息
        output_file: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("RGS-Net V3 批量评估对比报告\n")
            f.write("=" * 100 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总实验数: {len(eval_results)}\n")
            f.write(f"成功评估: {sum(1 for r in eval_results if r['success'])}\n")
            f.write(f"失败评估: {sum(1 for r in eval_results if not r['success'])}\n\n")
            
            f.write("=" * 100 + "\n")
            f.write("评估结果对比表\n")
            f.write("=" * 100 + "\n\n")
            
            # 表头
            f.write(f"{'序号':<6} {'实验名称':<40} {'精确率':<12} {'召回率':<12} {'F1分数':<12} {'状态':<10}\n")
            f.write("-" * 100 + "\n")
            
            # 按 F1 分数排序（降序）
            sorted_results = sorted(
                eval_results,
                key=lambda x: x["metrics"].get("f1", 0.0),
                reverse=True
            )
            
            for idx, result in enumerate(sorted_results, 1):
                exp_name = result["exp_name"][:38]  # 截断过长的名称
                if result["success"]:
                    metrics = result["metrics"]
                    precision = metrics.get("precision", 0.0)
                    recall = metrics.get("recall", 0.0)
                    f1 = metrics.get("f1", 0.0)
                    status = "✓ 成功"
                    f.write(f"{idx:<6} {exp_name:<40} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {status:<10}\n")
                else:
                    f.write(f"{idx:<6} {exp_name:<40} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'✗ 失败':<10}\n")
            
            f.write("\n" + "=" * 100 + "\n")
            f.write("详细信息\n")
            f.write("=" * 100 + "\n\n")
            
            # 详细信息
            for idx, result in enumerate(sorted_results, 1):
                f.write(f"[{idx}] {result['exp_name']}\n")
                f.write(f"    输出目录: {result['output_dir']}\n")
                
                if result["success"]:
                    metrics = result["metrics"]
                    f.write(f"    精确率: {metrics.get('precision', 0.0):.4f}\n")
                    f.write(f"    召回率: {metrics.get('recall', 0.0):.4f}\n")
                    f.write(f"    F1分数: {metrics.get('f1', 0.0):.4f}\n")
                    
                    # 尝试获取训练信息
                    exp_info = next(
                        (e for e in training_summary.get("experiments", []) 
                         if e["name"] == result["exp_name"]),
                        None
                    )
                    if exp_info and "params" in exp_info:
                        f.write(f"    超参数:\n")
                        for key, value in exp_info["params"].items():
                            f.write(f"      - {key}: {value}\n")
                else:
                    f.write(f"    错误: {result['error']}\n")
                
                f.write("\n")
        
        log_message(f"对比报告已保存: {output_file}")
    except Exception as e:
        log_message(f"保存对比报告失败: {e}", level="ERROR")


def save_json_report(
    eval_results: List[Dict[str, Any]],
    training_summary: Dict[str, Any],
    output_file: str
):
    """保存 JSON 格式的评估结果"""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_experiments": len(eval_results),
        "successful_evaluations": sum(1 for r in eval_results if r["success"]),
        "failed_evaluations": sum(1 for r in eval_results if not r["success"]),
        "training_summary": training_summary,
        "evaluation_results": eval_results
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        log_message(f"JSON 报告已保存: {output_file}")
    except Exception as e:
        log_message(f"保存 JSON 报告失败: {e}", level="ERROR")


def main():
    """主函数"""
    args = parse_args()
    
    log_message("=" * 80)
    log_message("RGS-Net V3 批量评估脚本启动")
    log_message("=" * 80)
    log_message(f"汇总文件: {args.summary_file}")
    log_message(f"评估批大小: {args.batch_size}")
    log_message(f"GPU 设备: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    log_message("")
    
    # 检查评估脚本是否存在
    if not os.path.exists(EVAL_SCRIPT):
        log_message(f"评估脚本不存在: {EVAL_SCRIPT}", level="ERROR")
        sys.exit(1)
    
    # 加载训练汇总
    training_summary = load_summary(args.summary_file)
    experiments = training_summary.get("experiments", [])
    
    if not experiments:
        log_message("没有找到任何实验记录", level="ERROR")
        sys.exit(1)
    
    log_message(f"找到 {len(experiments)} 个实验")
    log_message("")
    
    # 过滤出成功的训练
    successful_experiments = [e for e in experiments if e.get("success", False)]
    log_message(f"其中 {len(successful_experiments)} 个训练成功")
    
    if not successful_experiments:
        log_message("没有成功的训练可供评估", level="ERROR")
        sys.exit(1)
    
    # 依次评估每个模型
    eval_results = []
    for i, exp in enumerate(successful_experiments, 1):
        log_message(f"\n进度: [{i}/{len(successful_experiments)}]")
        
        result = run_evaluation(
            output_dir=exp["output_dir"],
            exp_name=exp["name"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            algorithm=args.algorithm
        )
        eval_results.append(result)
    
    # 保存对比报告
    log_message("\n" + "=" * 80)
    log_message("生成评估报告")
    log_message("=" * 80)
    
    summary_dir = os.path.dirname(args.summary_file)
    
    # 文本报告
    txt_report = os.path.join(summary_dir, "evaluation_comparison.txt")
    save_comparison_report(eval_results, training_summary, txt_report)
    
    # JSON 报告
    json_report = os.path.join(summary_dir, "evaluation_results.json")
    save_json_report(eval_results, training_summary, json_report)
    
    # 打印汇总
    log_message("\n" + "=" * 80)
    log_message("批量评估完成")
    log_message("=" * 80)
    log_message(f"总评估数: {len(eval_results)}")
    log_message(f"成功: {sum(1 for r in eval_results if r['success'])}")
    log_message(f"失败: {sum(1 for r in eval_results if not r['success'])}")
    log_message(f"\n报告目录: {summary_dir}")
    log_message(f"  - 文本报告: {txt_report}")
    log_message(f"  - JSON报告: {json_report}")
    
    # 找出最佳模型
    successful_evals = [r for r in eval_results if r["success"] and r["metrics"]]
    if successful_evals:
        best_model = max(successful_evals, key=lambda x: x["metrics"].get("f1", 0.0))
        log_message(f"\n最佳模型 (按 F1 分数):")
        log_message(f"  - 实验名称: {best_model['exp_name']}")
        log_message(f"  - F1 分数: {best_model['metrics'].get('f1', 0.0):.4f}")
        log_message(f"  - 精确率: {best_model['metrics'].get('precision', 0.0):.4f}")
        log_message(f"  - 召回率: {best_model['metrics'].get('recall', 0.0):.4f}")
        log_message(f"  - 输出目录: {best_model['output_dir']}")
    
    log_message("\n" + "=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n评估被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
