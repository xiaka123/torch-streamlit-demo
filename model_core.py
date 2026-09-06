# model_core.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------- 全局随机种子 --------------------------
def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    qml.random.seed(seed)


# -------------------------- 数据集加载/仿真生成 --------------------------
def load_or_generate_data(points_per_sample: int,
                          num_samples: int,
                          file_normal=None,
                          file_fault=None):
    """
    file_normal/file_fault: excel文件路径，为None时生成仿真声纹数据
    返回 norm_signals, fault_signals
    """
    if file_normal is not None and file_fault is not None:
        df_norm = pd.read_excel(file_normal)
        df_fault = pd.read_excel(file_fault)
        norm_signals = []
        fault_signals = []
        for col in df_norm.columns:
            sig = df_norm[col].values[:points_per_sample]
            norm_signals.append(sig)
        for col in df_fault.columns:
            sig = df_fault[col].values[:points_per_sample]
            fault_signals.append(sig)
        return norm_signals, fault_signals
    else:
        # 仿真生成正常/故障声纹
        fs = 50000
        t = np.linspace(0, 0.6, points_per_sample)
        norm_signals = []
        fault_signals = []
        for _ in range(num_samples):
            # 正常信号
            sig_n = 0.4 * np.sin(2 * np.pi * 1200 * t) + 0.2 * np.sin(2 * np.pi * 2400 * t)
            sig_n += 0.08 * np.random.randn(points_per_sample)
            norm_signals.append(sig_n)
            # 故障信号，增加高频分量
            sig_f = 0.4 * np.sin(2 * np.pi * 1200 * t) + 0.35 * np.sin(2 * np.pi * 4800 * t)
            sig_f += 0.12 * np.random.randn(points_per_sample)
            fault_signals.append(sig_f)
        return norm_signals, fault_signals


# -------------------------- 信号分段切片函数 --------------------------
def slice_partition(signal_list, label_array, slice_len=2048, stride=512):
    """
    对长信号滑动切片，返回切片样本与对应标签
    """
    X_out = []
    y_out = []
    for sig, lab in zip(signal_list, label_array):
        n = len(sig)
        start = 0
        while start + slice_len <= n:
            piece = sig[start:start + slice_len]
            X_out.append(piece)
            y_out.append(lab)
            start += stride
    X_arr = np.array(X_out, dtype=np.float32)
    y_arr = np.array(y_out, dtype=np.int64)
    return X_arr, y_arr


# -------------------------- VQC量子电路层 --------------------------
class RigorousVQCLayer(torch.nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        n_params = n_qubits * 6
        self.theta = nn.Parameter(torch.randn(n_params) * 0.01)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            idx = 0
            for _ in range(3):
                for q in range(n_qubits):
                    qml.RX(weights[idx], wires=q)
                    qml.RY(weights[idx+1], wires=q)
                    qml.RZ(weights[idx+2], wires=q)
                    idx +=3
                for q in range(n_qubits-1):
                    qml.CNOT(wires=[q, q+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        self.circuit = circuit

    def forward(self, x):
        bsz = x.shape[0]
        out_list = []
        for b in range(bsz):
            res = self.circuit(x[b], self.theta)
            out_list.append(torch.stack(res))
        return torch.stack(out_list)


# -------------------------- VQC‑Q网络 DQN网络 --------------------------
class VQC_QNetwork(torch.nn.Module):
    def __init__(self, input_channels=1, num_actions=2):
        super().__init__()
        self.n_qubits = 4
        self.feat_extract = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=64, stride=16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=32, stride=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(self.n_qubits)
        )
        self.vqc = RigorousVQCLayer(n_qubits=self.n_qubits)
        self.head = nn.Linear(self.n_qubits, num_actions)

    def forward(self, x):
        # x: [B, slice_len]
        x = x.unsqueeze(1)
        feat = self.feat_extract(x)
        feat = feat.squeeze(-1)
        vqc_out = self.vqc(feat)
        q_val = self.head(vqc_out)
        return q_val


# -------------------------- DQN经验回放缓冲区 --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        data = (state.copy(), action, reward, next_state.copy(), done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        s_t = torch.tensor(np.array(s), dtype=torch.float32)
        a_t = torch.tensor(np.array(a), dtype=torch.int64)
        r_t = torch.tensor(np.array(r), dtype=torch.float32)
        ns_t = torch.tensor(np.array(ns), dtype=torch.float32)
        d_t = torch.tensor(np.array(d), dtype=torch.float32)
        return s_t, a_t, r_t, ns_t, d_t

    def __len__(self):
        return len(self.buffer)
