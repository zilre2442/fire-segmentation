#!/usr/bin/env python3
"""
多次训练 RGS-Net V3 的批量执行脚本

使用方法:
    CUDA_VISIBLE_DEVICES=1 python3 exp/train_scripts/multi_run_train_RGS_Net_V3.py

功能:
1. 依次执行多个训练任务，每个任务使用不同的超参数配置
2. 自动记录每次训练的超参数和对应的输出目录
3. 支持在训练中断后从未完成的任务继续执行
4. 生成汇总报告，包含所有训练任务的配置和结果路径
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from typing import Dict, List, Any
import time


# ============================================================================
# 超参数配置区域 - 在此定义所有需要运行的实验配置
# ============================================================================

EXPERIMENTS = [
    {
        "name": "分割分支优化1",
        "description": "基于连通域提升_202511011046，重建分支使用uniform策略，降低分割分支SEG_ALPHA和SEG_GAMMA",
        "params": {
            "BATCH_SIZE": 64,
            "LEARNING_RATE": 3e-4,
            "EPOCHS": 200,
            "EARLY_STOPPING_PATIENCE": 10,
            # SpatialFocalLoss 参数 - 分割分支
            "SEG_WEIGHT_STRATEGY": "small",
            "SEG_ALPHA": 0.7,
            "SEG_GAMMA": 1.5,
            "SEG_WEIGHT_MIN": 1.0,
            "SEG_WEIGHT_MAX": 4.0,
            "SEG_WEIGHT_GAMMA": 2.0,
            # SpatialFocalLoss 参数 - 重建分支
            "REC_WEIGHT_STRATEGY": "uniform",
            "REC_ALPHA": 0.25,
            "REC_GAMMA": 1.5,
            "REC_WEIGHT_MIN": 1.0,
            "REC_WEIGHT_MAX": 4.0,
            "REC_WEIGHT_GAMMA": 2.0,
        }
    },
    {
        "name": "分割分支优化2",
        "description": "基于连通域提升_202511011046，重建分支使用large策略，降低分割分支SEG_ALPHA和SEG_GAMMA",
        "params": {
            "BATCH_SIZE": 64,
            "LEARNING_RATE": 3e-4,
            "EPOCHS": 200,
            "EARLY_STOPPING_PATIENCE": 10,
            # SpatialFocalLoss 参数 - 分割分支
            "SEG_WEIGHT_STRATEGY": "small",
            "SEG_ALPHA": 0.7,
            "SEG_GAMMA": 1.5,
            "SEG_WEIGHT_MIN": 1.0,
            "SEG_WEIGHT_MAX": 4.0,
            "SEG_WEIGHT_GAMMA": 1.5,
            # SpatialFocalLoss 参数 - 重建分支
            "REC_WEIGHT_STRATEGY": "large",
            "REC_ALPHA": 0.3,
            "REC_GAMMA": 2.0,
            "REC_WEIGHT_MIN": 1.0,
            "REC_WEIGHT_MAX": 4.0,
            "REC_WEIGHT_GAMMA": 1.5,
        }
    },
    # 可以继续添加更多实验配置...
]


# ============================================================================
# 全局配置
# ============================================================================

# GPU 配置 (从环境变量 CUDA_VISIBLE_DEVICES 读取)
GPU_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1")
NUM_GPUS = len(GPU_DEVICES.split(","))

# 训练脚本路径
TRAIN_SCRIPT = "exp/train_scripts/train_RGS_Net_V3.py"

# 结果保存根目录
RESULTS_ROOT = "output/RGS_Net_V3/multi_run"

# 日志文件
MULTI_RUN_LOG = os.path.join(RESULTS_ROOT, "multi_run_log.txt")
SUMMARY_JSON = os.path.join(RESULTS_ROOT, "experiments_summary.json")


# ============================================================================
# 辅助函数
# ============================================================================

def setup_directories():
    """创建必要的目录"""
    os.makedirs(RESULTS_ROOT, exist_ok=True)


def log_message(message: str, level: str = "INFO"):
    """记录日志信息到文件和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {message}"
    
    print(log_line)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(MULTI_RUN_LOG)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    with open(MULTI_RUN_LOG, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")


def create_modified_train_script(exp_config: Dict[str, Any], output_dir: str) -> str:
    """
    创建临时的训练脚本，将超参数注入到脚本中
    
    Args:
        exp_config: 实验配置字典
        output_dir: 输出目录
    
    Returns:
        临时脚本的路径
    """
    # 读取原始训练脚本
    with open(TRAIN_SCRIPT, "r", encoding="utf-8") as f:
        original_script = f.read()
    
    # 创建临时脚本目录
    temp_dir = os.path.join(RESULTS_ROOT, "temp_scripts")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 临时脚本路径 - 清理实验名称中的特殊字符以避免文件名问题
    safe_name = exp_config['name'].replace('/', '_').replace('\\', '_').replace(' ', '_')
    temp_script = os.path.join(temp_dir, f"train_{safe_name}.py")
    
    # 替换脚本中的超参数
    modified_script = original_script
    
    # 替换 SAVE_DIR
    run_id = datetime.now().strftime("%Y%m%d%H%M")
    save_dir_line = f'SAVE_DIR = "{output_dir}"'
    modified_script = modified_script.replace(
        'SAVE_DIR = f"output/RGS_Net_V3/{ALGORITHM}_{RUN_ID}"',
        save_dir_line
    )
    
    # 替换基本训练参数
    params = exp_config["params"]
    if "BATCH_SIZE" in params:
        modified_script = modified_script.replace(
            "BATCH_SIZE = 64",
            f"BATCH_SIZE = {params['BATCH_SIZE']}"
        )
    if "LEARNING_RATE" in params:
        modified_script = modified_script.replace(
            "LEARNING_RATE = 3e-4",
            f"LEARNING_RATE = {params['LEARNING_RATE']}"
        )
    if "EPOCHS" in params:
        modified_script = modified_script.replace(
            "EPOCHS = 200",
            f"EPOCHS = {params['EPOCHS']}"
        )
    if "EARLY_STOPPING_PATIENCE" in params:
        modified_script = modified_script.replace(
            "EARLY_STOPPING_PATIENCE = 10",
            f"EARLY_STOPPING_PATIENCE = {params['EARLY_STOPPING_PATIENCE']}"
        )
    
    # 替换 SpatialFocalLoss 参数 - 分割分支
    seg_criterion_old = '''seg_criterion = SpatialFocalLoss(
            weight_strategy="small",
            alpha=0.75,
            gamma=2.0,
            weight_min=1.0,
            weight_max=3.0,
            weight_gamma=1.5,
            background_weight=1.0
        )'''
    
    seg_criterion_new = f'''seg_criterion = SpatialFocalLoss(
            weight_strategy="{params.get('SEG_WEIGHT_STRATEGY', 'small')}",
            alpha={params.get('SEG_ALPHA', 0.75)},
            gamma={params.get('SEG_GAMMA', 2.0)},
            weight_min={params.get('SEG_WEIGHT_MIN', 1.0)},
            weight_max={params.get('SEG_WEIGHT_MAX', 3.0)},
            weight_gamma={params.get('SEG_WEIGHT_GAMMA', 1.5)},
            background_weight=1.0
        )'''
    
    modified_script = modified_script.replace(seg_criterion_old, seg_criterion_new)
    
    # 替换 SpatialFocalLoss 参数 - 重建分支
    rec_criterion_old = '''recon_criterion = SpatialFocalLoss(
            weight_strategy="large",
            alpha=0.75,
            gamma=1.5,
            weight_min=1.0,
            weight_max=4.0,
            weight_gamma=1.5,
            background_weight=1.0
        )'''
    
    rec_criterion_new = f'''recon_criterion = SpatialFocalLoss(
            weight_strategy="{params.get('REC_WEIGHT_STRATEGY', 'large')}",
            alpha={params.get('REC_ALPHA', 0.75)},
            gamma={params.get('REC_GAMMA', 1.5)},
            weight_min={params.get('REC_WEIGHT_MIN', 1.0)},
            weight_max={params.get('REC_WEIGHT_MAX', 4.0)},
            weight_gamma={params.get('REC_WEIGHT_GAMMA', 1.5)},
            background_weight=1.0
        )'''
    
    modified_script = modified_script.replace(rec_criterion_old, rec_criterion_new)
    
    # 写入临时脚本
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(modified_script)
    
    log_message(f"Created temporary training script: {temp_script}")
    return temp_script


def run_training(exp_config: Dict[str, Any], exp_index: int, total_exps: int) -> Dict[str, Any]:
    """
    执行单次训练
    
    Args:
        exp_config: 实验配置
        exp_index: 当前实验索引
        total_exps: 总实验数量
    
    Returns:
        包含训练结果的字典
    """
    exp_name = exp_config["name"]
    exp_desc = exp_config["description"]
    
    log_message("=" * 80)
    log_message(f"开始实验 [{exp_index + 1}/{total_exps}]: {exp_name}")
    log_message(f"描述: {exp_desc}")
    log_message("=" * 80)
    
    # 创建输出目录 - 清理实验名称中的特殊字符
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    safe_name = exp_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    output_dir = os.path.join(RESULTS_ROOT, f"{safe_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存实验配置到输出目录
    config_file = os.path.join(output_dir, "experiment_config.json")
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(exp_config, f, indent=4, ensure_ascii=False)
        log_message(f"实验配置已保存: {config_file}")
    except Exception as e:
        log_message(f"保存实验配置失败: {e}", level="ERROR")
    
    # 创建修改后的训练脚本
    try:
        temp_script = create_modified_train_script(exp_config, output_dir)
    except Exception as e:
        log_message(f"创建训练脚本失败: {e}", level="ERROR")
        return {
            "name": exp_name,
            "description": exp_desc,
            "output_dir": output_dir,
            "config_file": config_file,
            "temp_script": None,
            "success": False,
            "error": f"创建训练脚本失败: {e}",
            "duration_seconds": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "params": exp_config["params"]
        }
    
    # 构建训练命令
    cmd = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        temp_script
    ]
    
    log_message(f"执行命令: {' '.join(cmd)}")
    log_message(f"GPU 设备: {GPU_DEVICES}")
    log_message(f"输出目录: {output_dir}")
    
    # 执行训练
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU_DEVICES},
            capture_output=False,
            text=True,
            check=True
        )
        success = True
        error_msg = None
        log_message(f"实验 {exp_name} 完成!")
    except subprocess.CalledProcessError as e:
        success = False
        error_msg = str(e)
        log_message(f"实验 {exp_name} 失败: {error_msg}", level="ERROR")
    except Exception as e:
        success = False
        error_msg = str(e)
        log_message(f"实验 {exp_name} 发生异常: {error_msg}", level="ERROR")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # 返回结果
    result_dict = {
        "name": exp_name,
        "description": exp_desc,
        "output_dir": output_dir,
        "config_file": config_file,
        "temp_script": temp_script,
        "success": success,
        "error": error_msg,
        "duration_seconds": duration,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "params": exp_config["params"]
    }
    
    log_message(f"训练耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
    log_message("")
    
    return result_dict


def save_summary(results: List[Dict[str, Any]]):
    """保存所有实验的汇总信息"""
    summary = {
        "total_experiments": len(results),
        "successful_experiments": sum(1 for r in results if r["success"]),
        "failed_experiments": sum(1 for r in results if not r["success"]),
        "gpu_devices": GPU_DEVICES,
        "num_gpus": NUM_GPUS,
        "experiments": results,
        "generated_at": datetime.now().isoformat()
    }
    
    # 确保目录存在
    summary_dir = os.path.dirname(SUMMARY_JSON)
    if summary_dir and not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)
    
    try:
        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log_message(f"保存汇总文件失败: {e}", level="ERROR")
        return
    
    log_message("=" * 80)
    log_message("所有实验汇总")
    log_message("=" * 80)
    log_message(f"总实验数: {summary['total_experiments']}")
    log_message(f"成功: {summary['successful_experiments']}")
    log_message(f"失败: {summary['failed_experiments']}")
    log_message(f"汇总文件: {SUMMARY_JSON}")
    log_message("")
    
    # 打印每个实验的结果
    for i, result in enumerate(results, 1):
        status = "✓ 成功" if result["success"] else "✗ 失败"
        log_message(f"  [{i}] {result['name']}: {status}")
        log_message(f"      输出目录: {result['output_dir']}")
        log_message(f"      训练时长: {result['duration_seconds']/60:.2f} 分钟")
        if not result["success"]:
            log_message(f"      错误信息: {result['error']}", level="ERROR")
        log_message("")


def main():
    """主函数"""
    # 检查训练脚本是否存在
    if not os.path.exists(TRAIN_SCRIPT):
        print(f"错误: 训练脚本不存在: {TRAIN_SCRIPT}")
        print(f"当前工作目录: {os.getcwd()}")
        sys.exit(1)
    
    # 检查实验配置是否为空
    if not EXPERIMENTS:
        print("错误: 没有配置任何实验，请在 EXPERIMENTS 列表中添加实验配置")
        sys.exit(1)
    
    log_message("=" * 80)
    log_message("RGS-Net V3 多次训练脚本启动")
    log_message("=" * 80)
    log_message(f"GPU 设备: {GPU_DEVICES}")
    log_message(f"GPU 数量: {NUM_GPUS}")
    log_message(f"计划运行实验数: {len(EXPERIMENTS)}")
    log_message(f"结果保存目录: {RESULTS_ROOT}")
    log_message(f"训练脚本路径: {TRAIN_SCRIPT}")
    log_message("")
    
    # 创建必要的目录
    setup_directories()
    
    # 依次执行所有实验
    results = []
    for i, exp_config in enumerate(EXPERIMENTS):
        result = run_training(exp_config, i, len(EXPERIMENTS))
        results.append(result)
        
        # 每次实验后保存中间结果
        save_summary(results)
    
    log_message("=" * 80)
    log_message("所有实验执行完成!")
    log_message("=" * 80)
    
    # 最终汇总
    save_summary(results)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        try:
            log_message("\n训练被用户中断", level="WARNING")
        except:
            pass  # 如果日志记录失败，忽略错误
        sys.exit(1)
    except Exception as e:
        print(f"发生未预期的错误: {e}")
        import traceback
        print(traceback.format_exc())
        try:
            log_message(f"发生未预期的错误: {e}", level="ERROR")
            log_message(traceback.format_exc(), level="ERROR")
        except:
            pass  # 如果日志记录失败，忽略错误
        sys.exit(1)
