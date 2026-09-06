import os
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 设置Matplotlib支持中文与负号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (18, 10)


def load_or_generate_data(df_normal=None, df_fault=None, points_per_sample=3000, num_samples=70):
    """生成模拟故障数据集，二分类：0正常，1传动机构连杆受阻"""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    X_list = []
    y_list = []
    for _ in range(num_samples // 2):
        sig = np.sin(np.linspace(0, 8 * np.pi, points_per_sample)) + 0.2 * np.random.randn(points_per_sample)
        X_list.append(sig)
        y_list.append(0)
    for _ in range(num_samples // 2):
        t = np.linspace(0, 8 * np.pi, points_per_sample)
        sig = np.sin(t) + 0.4 * np.sin(2.8 * t) + 0.3 * np.random.randn(points_per_sample)
        X_list.append(sig)
        y_list.append(1)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logit = self.fc(last_hidden)
        return logit


def main():
    st.title("🔧 高压断路器故障诊断 LSTM 模型演示")
    st.markdown("模拟数据集：正常 / 传动机构连杆受阻")

    # 加载数据
    X_train, X_test, y_train, y_test = load_or_generate_data()
    seq_len = X_train.shape[1]
    # 增加维度 (batch, seq_len, feature)
    X_train_tensor = torch.from_numpy(X_train).unsqueeze(-1)
    y_train_tensor = torch.from_numpy(y_train).unsqueeze(-1)
    X_test_tensor = torch.from_numpy(X_test).unsqueeze(-1)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(-1)

    # 超参
    epochs = 60
    lr = 1e-3
    model = LSTMClassifier(input_dim=1, hidden_dim=64, num_layers=2)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = loss_fn(logits, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_loss_history.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_tensor)
            val_loss = loss_fn(val_logits, y_test_tensor)
            val_loss_history.append(val_loss.item())

        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"训练轮次 {epoch+1}/{epochs} | loss:{loss.item():.4f} | val_loss:{val_loss.item():.4f}")

    status_text.text("✅训练完成！")

    # 推理预测
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_tensor)
        y_pred_prob = torch.sigmoid(test_logits).numpy().ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)
        y_true = y_test

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    st.subheader("📊模型测试集评估指标")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("准确率", f"{acc*100:.2f}%")
    with col2:
        st.metric("精确率", f"{prec*100:.2f}%")
    with col3:
        st.metric("召回率", f"{rec*100:.2f}%")
    with col4:
        st.metric("F1‑Score", f"{f1:.4f}")
    with col5:
        st.metric("特异度", f"{specificity*100:.2f}%")

    # 分类报告转为DataFrame表格（修复无表格问题）
    report_dict = classification_report(
        y_true, y_pred,
        target_names=["正常", "传动机构连杆受阻"],
        output_dict=True
    )
    df_report = pd.DataFrame(report_dict).T
    st.subheader("📋分类报告表格")
    st.dataframe(df_report, use_container_width=True)

    # 绘图：训练loss、混淆矩阵、ROC曲线；全部使用英文短横线 '-'，修复ValueError报错
    epochs_range = list(range(1, epochs + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # loss曲线
    axes[0,0].plot(epochs_range, train_loss_history, 'o-', color="#2ca02c", markersize=3, label="Train Loss")
    axes[0,0].plot(epochs_range, val_loss_history, 's-', color="#1f77b4", markersize=3, label="Val Loss")
    axes[0,0].set_title("训练 & 验证损失曲线")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0,1],
                xticklabels=["正常","传动机构连杆受阻"],
                yticklabels=["正常","传动机构连杆受阻"])
    axes[0,1].set_title("混淆矩阵")
    axes[0,1].set_xlabel("预测标签")
    axes[0,1].set_ylabel("真实标签")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[0,2].plot(fpr, tpr, 'o-', lw=2, label=f"AUC={roc_auc:.4f}")
    axes[0,2].plot([0,1],[0,1],'--',color="gray")
    axes[0,2].set_title("ROC曲线")
    axes[0,2].set_xlabel("FPR")
    axes[0,2].set_ylabel("TPR")
    axes[0,2].legend()
    axes[0,2].grid(True,alpha=0.3)

    # 样本示例
    axes[1,0].plot(X_train[0],color="#1f77b4")
    axes[1,0].set_title("正常样本信号示例")
    axes[1,1].plot(X_train[-1],color="#d62728")
    axes[1,1].set_title("故障样本信号示例")

    axes[1,2].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
