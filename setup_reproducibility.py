# -*- coding: utf-8 -*-
"""
STAMP-Miner Reproducibility Setup Tool
Author: Prof. AI Specialist (for Luo Wei)
Function: Automatically refactors absolute paths to local relative paths for seamless peer review.
"""

import os
import re
from pathlib import Path


def patch_paths():
    # 1. 获取当前项目根目录
    root_dir = Path(__file__).resolve().parent
    print(f"[Checking] STAMP-Miner Local Root: {root_dir}")

    # 2. 定义需要扫描的文件夹
    target_folders = ['scripts', 'core', 'structure_prediction', 'step2_prior_knowledge']

    # 3. 定义需要被替换的绝对路径前缀（匹配你的 D 盘路径习惯）
    # 匹配 D:\ 或 1/pythonProject4/ 这种硬编码路径
    abs_path_pattern = re.compile(r'r?[\'"][D1]:[\\/][^\'"]+[\\/]([^\'"]+\.\w+)[\'"]')

    patched_count = 0

    for folder in target_folders:
        folder_path = root_dir / folder
        if not folder_path.exists():
            continue

        for py_file in folder_path.glob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查文件中是否包含绝对路径
            if "D:\\" in content or "D:/" in content or "1/pythonProject4" in content:
                print(f"[Patching] Found hardcoded paths in: {py_file.name}")

                # 智能替换策略：
                # 将路径指向仓库中正确的 data 或 bin 目录
                new_content = content

                # 替换数据文件路径
                new_content = re.sub(r'r?[\'"][D1]:[\\/][^\'"]+[\\/]py_amp_stand_sqe\.csv[\'"]',
                                     f'os.path.join(os.path.dirname(__file__), "../data/py_amp_stand_sqe.csv")',
                                     new_content)

                # 替换模型权重路径
                new_content = re.sub(r'r?[\'"][D1]:[\\/][^\'"]+[\\/](AWLSTM_2\.pth|HWLSTM\.pth)[\'"]',
                                     f'os.path.join(os.path.dirname(__file__), "../bin/\\1")', new_content)

                # 替换字典文件路径
                new_content = re.sub(r'r?[\'"][D1]:[\\/][^\'"]+[\\/]dict_AWLSTM\.csv[\'"]',
                                     f'os.path.join(os.path.dirname(__file__), "../bin/dict_AWLSTM.csv")', new_content)

                # 替换 ColabFold 输入输出
                new_content = re.sub(r'r?[\'"][D1]:[\\/][^\'"]+[\\/]pic_data[\\/][^\'"]+[\'"]',
                                     f'os.path.join(os.path.dirname(__file__), "../data/raw/")', new_content)

                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write("import os\n" + new_content if "import os" not in new_content else new_content)

                patched_count += 1

    print(f"\n[Success] Successfully patched {patched_count} scripts.")
    print("[Action] Now you can run any script in the 'scripts/' directory without path errors.")


if __name__ == "__main__":
    patch_paths()