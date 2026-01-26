# -*- coding: utf-8 -*-
"""
STAMP-Miner Module 3: Targeted Recognition Validation (Inference)
Function: Load trained AWLSTM model and predict specificity against target pathogens.
"""

import os
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchtext.vocab import vocab as torch_vocab
from collections import OrderedDict

# --- 环境配置 ---
def get_args():
    parser = argparse.ArgumentParser(description="STAMP-Miner AWLSTM Inference Pipeline")
    parser.add_argument('--input', type=str, default='results/04_docking_ifp/top100_observed_ifp.csv', 
                        help='Path to the input CSV file from Module 2')
    parser.add_argument('--model_path', type=str, default='bin/AWLSTM_2.pth', 
                        help='Path to the trained .pth model')
    parser.add_argument('--dict_path', type=str, default='bin/dict_AWLSTM.csv', 
                        help='Path to the dictionary file')
    parser.add_argument('--output', type=str, default='results/05_final_leads/P1_P4_candidates.csv', 
                        help='Path to save the prediction results')
    parser.add_argument('--max_len', type=int, default=70, help='Max sequence length for padding')
    return parser.parse_args()


def get_args():
    parser = argparse.ArgumentParser(description="STAMP-Miner AWLSTM Inference Pipeline")
    parser.add_argument('--input_csv', type=str, default='results/04_docking_ifp/top100_observed_ifp.csv', 
                        help='Path to the input CSV file from Module 2')
    
    parser.add_argument('--model_path', type=str, default='bin/HWLSTM.pth', 
                        help='Path to the trained .pth model')
    
    parser.add_argument('--dict_path', type=str, default='bin/dict_AWLSTM.csv', 
                        help='Path to the dictionary file')
    
    parser.add_argument('--output', type=str, default='results/05_final_leads/P1_P4_candidates.csv', 
                        help='Path to save the prediction results')
    parser.add_argument('--max_len', type=int, default=70, help='Max sequence length')
    return parser.parse_args()


# --- 模型定义 (保持与训练代码完全一致) ---
class LSTM_Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_size=70, max_len=70):
        super(LSTM_Net, self).__init__()
        self.max_len = max_len
        self.em = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * max_len, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.em(x)
        x, _ = self.lstm(x)
        x = x.contiguous().view(len(x), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.8)
        x = F.dropout(F.relu(self.fc2(x)), p=0.4)
        x = self.fc3(x)
        return x

# --- 数据处理工具 ---
def reg_text(sequence):
    """提取氨基酸序列中的字母"""
    token = re.compile('[A-Za-z]')
    return token.findall(str(sequence))

def load_custom_vocab(dict_path):
    """从导出的CSV加载词表，确保索引严格一致"""
    df_dict = pd.read_csv(dict_path)
    # 假设CSV第一行是词表字典 {字母: 索引}
    stoi = df_dict.iloc[0].to_dict()
    # 转换索引为整数
    stoi = {k: int(v) for k, v in stoi.items()}
    # 构建 torchtext 兼容的 vocab 对象
    sorted_dict = OrderedDict(sorted(stoi.items(), key=lambda v: v[1]))
    v = torch_vocab(sorted_dict)
    v.set_default_index(v["<unk>"] if "<unk>" in stoi else 0)
    return v

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 加载词表
    if not os.path.exists(args.dict_path):
        raise FileNotFoundError(f"❌ Dictionary not found at {args.dict_path}. Please ensure it exists in bin/")
    vocab = load_custom_vocab(args.dict_path)
    vocab_size = len(vocab)
    print(f"📚 Vocab size loaded: {vocab_size}")

    # 2. 初始化并加载模型
    model = LSTM_Net(vocab_size, max_len=args.max_len).to(device)
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"❌ Model weights not found at {args.model_path}")
    
    # 兼容两种保存方式：state_dict 或 完整模型
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        model = torch.load(args.model_path, map_location=device)
    
    model.eval()
    print("🧠 AWLSTM model loaded successfully.")

    # 3. 载入并预处理待预测数据
    df = pd.read_csv(args.input)
    print(f"📥 Loading {len(df)} candidates from {args.input}")
    
    sequences = df['sequence'].apply(reg_text)
    
    # 4. 序列转为 Tensor 并 Padding
    x_list = []
    for seq in sequences:
        indexed_seq = vocab(seq)
        # 固定长度处理 (Padding & Truncating)
        if len(indexed_seq) < args.max_len:
            indexed_seq = indexed_seq + [0] * (args.max_len - len(indexed_seq))
        else:
            indexed_seq = indexed_seq[:args.max_len]
        x_list.append(torch.tensor(indexed_seq, dtype=torch.int64))

    x_tensor = torch.stack(x_list).to(device)

    # 5. 模型预测
    print("🧪 Running inference...")
    with torch.no_grad():
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        amp_probs = probs[:, 1].cpu().numpy() # 获取属于AMP类别的概率

    # 6. 保存结果
    df['AWLSTM_prediction'] = preds
    df['AMP_probability'] = np.round(amp_probs, 4)
    
    # 筛选预测为正样本的候选肽 (P1-P4 优选)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    pos_count = np.sum(preds)
    print(f"✅ Prediction finished. Found {pos_count} potential STAMPs.")
    print(f"💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
