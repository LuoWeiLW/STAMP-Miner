import subprocess
import os
import sys
from pathlib import Path
import shutil
import time
from tqdm import tqdm
import threading
import queue
import torch
import logging
from datetime import datetime
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_available():
    """检查 GPU 是否可用"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ GPU 可用: {device_name}")
        logger.info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        return True
    else:
        logger.warning("⚠️ 未检测到 GPU，将使用 CPU 运行，速度会显著降低")
        return False


def convert_txt_to_fasta(input_txt, output_fasta):
    """
    将文本文件转换为 FASTA 格式

    参数:
        input_txt (str): 输入文本文件路径
        output_fasta (str): 输出 FASTA 文件路径

    返回:
        dict: 序列名称到序列的映射
    """
    seq_mapping = {}
    try:
        with open(input_txt, 'r') as f_in, open(output_fasta, 'w') as f_out:
            for i, line in enumerate(f_in):
                line = line.strip()
                if not line:
                    continue

                # 创建序列ID
                seq_id = f"seq{i + 1}"
                # 去除非字母字符并转换为大写
                seq = ''.join(filter(str.isalpha, line)).upper()

                # 写入FASTA格式
                f_out.write(f">{seq_id}\n")
                f_out.write(f"{seq}\n")

                # 保存映射关系
                seq_mapping[seq_id] = seq

        logger.info(f"✅ 成功转换文本文件为 FASTA 格式: {output_fasta}")
        logger.info(f"包含 {len(seq_mapping)} 个肽序列")
        return seq_mapping
    except Exception as e:
        logger.error(f"❌ 转换文件时出错: {str(e)}")
        raise


def run_colabfold(input_file, output_dir):
    """
    使用 ColabFold 批量预测肽结构

    参数:
        input_file (str): 输入文件路径（可以是文本文件或 FASTA 文件）
        output_dir (str): 输出目录路径
    """
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("开始批量预测肽结构")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 检查 GPU 可用性
    gpu_available = check_gpu_available()

    seq_mapping = None

    # 如果输入文件是文本文件（.txt），则转换为 FASTA 格式
    input_path = Path(input_file)
    if input_path.suffix.lower() == ".txt":
        fasta_file = Path(output_dir) / "input.fasta"
        try:
            seq_mapping = convert_txt_to_fasta(input_file, fasta_file)
            input_file = str(fasta_file)
        except Exception as e:
            logger.error(f"❌ 文件转换失败: {str(e)}")
            return
    else:
        # 如果是FASTA文件，直接使用
        logger.info(f"✅ 输入文件已经是 FASTA 格式: {input_file}")

    # 创建临时目录用于处理
    temp_dir = Path(output_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    logger.info(f"✅ 已创建临时目录: {temp_dir}")

    # 构建命令 - 优化参数以最大化速度
    cmd = [
        "colabfold_batch",
        "--model-type", "alphafold2_ptm",  # 使用单体预测模型
        "--num-models", "1",  # 只生成一个模型
        "--num-recycle", "1",  # 设置循环次数为1
        "--max-msa", "1:1",  # 最小化 MSA 大小
        "--msa-mode", "single_sequence",  # 禁用 MSA 搜索
        "--num-seeds", "1",  # 只使用一个种子
        "--num-ensemble", "1",  # 最小化集成次数
        "--overwrite-existing-results",
        input_file,
        str(temp_dir)
    ]

    # 如果 GPU 可用，添加 GPU 加速参数
    if gpu_available:
        cmd.append("--use-gpu-relax")
        logger.info("✅ 已启用 GPU 加速")
    else:
        logger.info("⚠️ 未启用 GPU 加速，使用 CPU 运行")

    logger.info(f"运行 ColabFold 命令: {' '.join(cmd)}")

    # 创建进度队列
    progress_queue = queue.Queue()
    status_queue = queue.Queue()

    # 启动进度监控线程
    progress_thread = threading.Thread(
        target=monitor_progress,
        args=(temp_dir, progress_queue, status_queue),
        daemon=True
    )
    progress_thread.start()

    # 执行命令
    try:
        # 使用 Popen 以便实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并 stdout 和 stderr
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 启动输出处理线程
        output_thread = threading.Thread(
            target=process_output,
            args=(process, status_queue),
            daemon=True
        )
        output_thread.start()

        # 等待进程完成
        returncode = process.wait()
        elapsed_time = time.time() - start_time

        # 通知进度线程停止
        progress_queue.put("STOP")
        progress_thread.join()
        output_thread.join()

        # 处理结果
        if returncode == 0:
            logger.info(f"✅ 预测成功完成! 总耗时: {elapsed_time:.2f} 秒")

            # 移动结果文件到最终输出目录
            move_results(temp_dir, output_dir, seq_mapping)

            # 清理临时目录
            shutil.rmtree(temp_dir)
            logger.info(f"✅ 结果已保存到: {output_dir}")
        else:
            logger.error(f"❌ 预测失败! 返回码: {returncode}")
            logger.error("请检查错误信息")
    except Exception as e:
        logger.error(f"❌ 执行命令时出错: {str(e)}")
        returncode = 1
    finally:
        # 确保线程停止
        progress_queue.put("STOP")
        if 'process' in locals() and process.poll() is None:
            process.terminate()


def process_output(process, status_queue):
    """处理命令输出并发送状态更新"""
    while True:
        output_line = process.stdout.readline()
        if not output_line and process.poll() is not None:
            break
        if output_line:
            # 提取关键信息发送到状态队列
            if "MSA" in output_line or "search" in output_line:
                status_queue.put(("MSA", output_line.strip()))
            elif "model" in output_line or "predict" in output_line:
                status_queue.put(("MODEL", output_line.strip()))
            elif "relax" in output_line:
                status_queue.put(("RELAX", output_line.strip()))
            elif "error" in output_line.lower() or "fail" in output_line.lower():
                status_queue.put(("ERROR", output_line.strip()))
            else:
                status_queue.put(("INFO", output_line.strip()))

    status_queue.put(("DONE", ""))


def monitor_progress(temp_dir, progress_queue, status_queue):
    """监控预测进度和状态"""
    # 初始化进度条
    pbar = tqdm(desc="预测进度", unit="肽", position=0)
    processed_peptides = set()
    last_update = time.time()

    # 状态跟踪
    current_stage = "初始化"
    stage_start = time.time()

    while True:
        # 检查是否收到停止信号
        try:
            if not progress_queue.empty():
                msg = progress_queue.get(timeout=0.1)
                if msg == "STOP":
                    pbar.close()
                    return
        except queue.Empty:
            pass

        # 处理状态更新
        try:
            while not status_queue.empty():
                msg_type, msg_content = status_queue.get(timeout=0.1)

                if msg_type == "MSA":
                    current_stage = "MSA搜索"
                    stage_start = time.time()
                    logger.info(f"🔍 {msg_content}")
                elif msg_type == "MODEL":
                    current_stage = "模型预测"
                    stage_start = time.time()
                    logger.info(f"🧠 {msg_content}")
                elif msg_type == "RELAX":
                    current_stage = "结构优化"
                    stage_start = time.time()
                    logger.info(f"⚙️ {msg_content}")
                elif msg_type == "ERROR":
                    logger.error(f"❌ {msg_content}")
                elif msg_type == "INFO":
                    logger.info(f"ℹ️ {msg_content}")
                elif msg_type == "DONE":
                    logger.info("✅ 所有处理完成")
        except queue.Empty:
            pass

        # 检查目录中的文件
        try:
            files = list(temp_dir.glob("*"))
            new_peptides = set()

            for file in files:
                if file.is_file() and ("_unrelaxed" in file.name or "_relaxed" in file.name):
                    # 提取肽ID
                    parts = file.name.split("_")
                    if len(parts) > 1:
                        peptide_id = parts[0]
                        new_peptides.add(peptide_id)

            # 更新进度条
            new_count = len(new_peptides - processed_peptides)
            if new_count > 0:
                pbar.update(new_count)
                processed_peptides.update(new_peptides)
                logger.info(f"📦 完成肽结构预测: {new_count} 个新肽")

            # 定期更新状态
            if time.time() - last_update > 5:  # 每5秒更新一次状态
                stage_duration = time.time() - stage_start
                logger.info(f"⏳ 当前阶段: {current_stage} | 已运行: {stage_duration:.1f}秒")
                last_update = time.time()

            # 每0.5秒检查一次
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"❌ 监控进度时出错: {str(e)}")
            time.sleep(1)


def move_results(temp_dir, output_dir, seq_mapping):
    """将结果文件移动到最终输出目录并以序列名称命名"""
    logger.info("📦 开始整理结果文件...")

    # 列出所有结果文件
    result_files = list(temp_dir.glob("*"))

    if not result_files:
        logger.warning("⚠️ 未找到结果文件")
        return

    logger.info(f"找到 {len(result_files)} 个结果文件")

    # 创建进度条
    progress = tqdm(result_files, desc="整理结果文件", position=1)

    for file_path in progress:
        # 跳过目录
        if file_path.is_dir():
            continue

        # 获取文件名
        filename = file_path.name

        # 跳过日志文件
        if filename == "log.txt":
            continue

        # 提取肽ID
        if "_unrelaxed" in filename or "_relaxed" in filename:
            # 格式: seq1_unrelaxed_rank_1_model_1.pdb
            parts = filename.split("_")
            if len(parts) > 0:
                peptide_id = parts[0]

                # 获取序列名称
                if seq_mapping and peptide_id in seq_mapping:
                    sequence = seq_mapping[peptide_id]
                    # 使用序列作为文件名
                    new_filename = f"{sequence}.pdb"
                else:
                    new_filename = f"{peptide_id}.pdb"
            else:
                new_filename = filename
        else:
            new_filename = filename

        # 移动文件
        new_path = Path(output_dir) / new_filename
        shutil.move(file_path, new_path)

        progress.set_description(f"移动文件: {filename} -> {new_filename}")
        logger.debug(f"📂 移动文件: {filename} -> {new_filename}")

    logger.info("✅ 结果文件整理完成")


if __name__ == "__main__":
    # 设置输入和输出路径
    input_file = r"D:\fzu\lw\jupyter\pycham2023\B15-screening\ab-result-opt3-5\pic_data\id_enhanced_screened_peptides_finally.txt"
    output_dir = r"D:\fzu\lw\jupyter\pycham2023\B15-screening\ab-result-opt3-5\pic_data\peptides_structures"

    try:
        # 运行预测
        run_colabfold(input_file, output_dir)
        logger.info("✅ 处理完成!")
    except Exception as e:
        logger.error(f"❌ 程序运行失败: {str(e)}")
        sys.exit(1)