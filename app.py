import streamlit as st
import io
import os
import matplotlib.pyplot as plt
from model_core import (
    seed_everything,
    load_or_generate_data,
    slice_partition,
    VQC_QNetwork,
    ReplayBuffer
)
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,roc_curve,auc

st.set_page_config(page_title="VQC‑RL断路器故障诊断",layout="wide")
seed_everything(42)

def main():
    st.title("ZN63高压真空断路器｜VQC‑RL声纹量子强化学习故障诊断系统")
    st.markdown("SCI仿真实验平台｜支持上传Excel声纹数据 / 内置物理仿真数据集")

    tab1,tab2,tab3 = st.tabs(["数据集设置","训练执行","结果可视化"])

    with tab1:
        st.subheader("数据源")
        use_sim = st.checkbox("使用内置物理仿真生成数据(无Excel文件时勾选)",value=True)
        up_normal = st.file_uploader("上传正常状态Excel",type=["xlsx"],disabled=use_sim)
        up_fault = st.file_uploader("上传故障状态Excel",type=["xlsx"],disabled=use_sim)
        points_per_sample = st.number_input("单样本采样点数",value=30000,step=1000)
        num_samples_sim = st.number_input("仿真样本数量",value=30,min_value=5)

        norm_signals,fault_signals = None,None
        if st.button("加载/生成数据集"):
            with st.spinner("生成/加载声纹数据..."):
                if use_sim:
                    norm_signals,fault_signals = load_or_generate_data(
                        points_per_sample=int(points_per_sample),
                        num_samples=int(num_samples_sim)
                    )
                else:
                    # streamlit上传文件是内存io，临时写入本地
                    with open("tmp_normal.xlsx","wb") as f:
                        f.write(up_normal.getvalue())
                    with open("tmp_fault.xlsx","wb") as f:
                        f.write(up_fault.getvalue())
                    norm_signals,fault_signals = load_or_generate_data(
                        file_normal="tmp_normal.xlsx",
                        file_fault="tmp_fault.xlsx",
                        points_per_sample=int(points_per_sample),
                        num_samples=int(num_samples_sim)
                    )
                st.session_state["norm"] = norm_signals
                st.session_state["fault"] = fault_signals
                st.success(f"完成！正常样本:{len(norm_signals)} 故障样本:{len(fault_signals)}")

    with tab2:
        st.subheader("训练超参数")
        epochs = st.slider("训练轮数",10,100,60)
        batch_size = st.slider("Batch size",4,32,16)
        lr = st.number_input("学习率",value=0.00015,format="%.6f")
        run_btn = st.button("启动VQC‑RL训练",type="primary")

        if run_btn:
            if "norm" not in st.session_state:
                st.warning("请先加载数据集！")
                return
            norm_signals = st.session_state["norm"]
            fault_signals = st.session_state["fault"]
            n_norm = len(norm_signals)
            n_fault = len(fault_signals)

            # 数据集划分逻辑，复制原代码
            if n_norm >=5 and n_fault >=5:
                idx_norm = np.arange(n_norm)
                idx_fault = np.arange(n_fault)
                tr_n,temp_n = train_test_split(idx_norm,test_size=0.4,random_state=42)
                val_n,te_n = train_test_split(temp_n,test_size=0.5,random_state=42)
                tr_f,temp_f = train_test_split(idx_fault,test_size=0.4,random_state=42)
                val_f,te_f = train_test_split(temp_f,test_size=0.5,random_state=42)

                X_train,y_train = slice_partition(np.vstack([norm_signals[tr_n],fault_signals[tr_f]]),
                                                  np.array([0]*len(tr_n)+[1]*len(tr_f)))
                X_val,y_val = slice_partition(np.vstack([norm_signals[val_n],fault_signals[val_f]]),
                                              np.array([0]*len(val_n)+[1]*len(val_f)))
                X_test,y_test = slice_partition(np.vstack([norm_signals[te_n],fault_signals[te_f]]),
                                               np.array([0]*len(te_n)+[1]*len(te_f)))
            else:
                def split_signal_blocks(sigs):
                    tr_blocks,val_blocks,te_blocks = [],[],[]
                    for s in sigs:
                        l=len(s)
                        p1=int(0.6*l)
                        p2=int(0.8*l)
                        tr_blocks.append(s[:p1])
                        val_blocks.append(s[p1:p2])
                        te_blocks.append(s[p2:])
                    return tr_blocks,val_blocks,te_blocks
                tr_norm_b,val_norm_b,te_norm_b = split_signal_blocks(norm_signals)
                tr_fault_b,val_fault_b,te_fault_b = split_signal_blocks(fault_signals)
                X_train,y_train = slice_partition(tr_norm_b+tr_fault_b,np.array([0]*len(tr_norm_b)+[1]*len(tr_fault_b)))
                X_val,y_val = slice_partition(val_norm_b+val_fault_b,np.array([0]*len(val_norm_b)+[1]*len(val_fault_b)))
                X_test,y_test = slice_partition(te_norm_b+te_fault_b,np.array([0]*len(te_norm_b)+[1]*len(te_fault_b)))

            st.info(f"训练:{len(X_train)} 验证:{len(X_val)} 测试:{len(X_test)}")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"运行设备: {device}")

            q_net = VQC_QNetwork(input_channels=1,num_actions=2).to(device)
            target_net = VQC_QNetwork(input_channels=1,num_actions=2).to(device)
            target_net.load_state_dict(q_net.state_dict())
            target_net.eval()
            optimizer = optim.AdamW(q_net.parameters(),lr=lr,weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5)
            criterion = nn.SmoothL1Loss()
            replay_buffer = ReplayBuffer(capacity=8000)

            gamma=0.95
            epsilon=0.9
            epsilon_min=0.05
            epsilon_decay=(epsilon-epsilon_min)/(epochs*0.7)

            train_loss_history=[]
            val_loss_history=[]
            train_acc_history=[]
            val_acc_history=[]

            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(1,epochs+1):
                q_net.train()
                epoch_losses=[]
                correct_train_steps=0
                total_train_steps=0
                indices = np.arange(len(X_train))
                np.random.shuffle(indices)
                for start_idx in range(0,len(indices),batch_size):
                    b_idx = indices[start_idx:start_idx+batch_size]
                    b_states = X_train[b_idx]
                    b_labels = y_train[b_idx]
                    states_t = torch.tensor(b_states,dtype=torch.float32).to(device)
                    with torch.no_grad():
                        q_vals = q_net(states_t)
                        greedy_actions = q_vals.argmax(dim=1).cpu().numpy()
                    actions = greedy_actions.copy()
                    for i in range(len(actions)):
                        if np.random.random() < epsilon:
                            actions[i]=np.random.randint(0,1)
                    rewards = np.where(actions==b_labels,1.0,-1.5)
                    for s,a,r in zip(b_states,actions,rewards):
                        replay_buffer.push(s,a,r,s,True)
                    correct_train_steps += np.sum(greedy_actions==b_labels)
                    total_train_steps += len(b_labels)
                    if len(replay_buffer)>=batch_size:
                        s_b,a_b,r_b,ns_b,d_b = replay_buffer.sample(batch_size)
                        s_b,a_b,r_b,ns_b,d_b = s_b.to(device),a_b.to(device),r_b.to(device),ns_b.to(device),d_b.to(device)
                        curr_q = q_net(s_b).gather(1,a_b.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            max_next_q = target_net(ns_b).max(dim=1)[0]
                            target_q = r_b + gamma*max_next_q*(1.0-d_b)
                        loss = criterion(curr_q,target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_net.parameters(),max_norm=2.0)
                        optimizer.step()
                        epoch_losses.append(loss.item())
                epsilon = max(epsilon_min,epsilon-epsilon_decay)
                scheduler.step()
                with torch.no_grad():
                    for param,target_param in zip(q_net.parameters(),target_net.parameters()):
                        target_param.data.copy_(0.15*param.data+0.85*target_param.data)

                q_net.eval()
                train_epoch_loss = np.mean(epoch_losses) if len(epoch_losses)>0 else 0
                train_epoch_acc = correct_train_steps/total_train_steps

                val_losses=[]
                val_correct=0
                v_indices = np.arange(len(X_val))
                for v_start in range(0,len(v_indices),batch_size):
                    vb_idx = v_indices[v_start:v_start+batch_size]
                    vb_x = torch.tensor(X_val[vb_idx],dtype=torch.float32).to(device)
                    vb_y = torch.tensor(y_val[vb_idx],dtype=torch.long).to(device)
                    v_q = q_net(vb_x)
                    v_preds = v_q.argmax(dim=1)
                    val_correct += (v_preds==vb_y).sum().item()
                    v_rewards = torch.where(v_preds==vb_y,1.0,-1.5).float()
                    v_curr_q = v_q.gather(1,v_preds.unsqueeze(1)).squeeze(1)
                    val_losses.append(criterion(v_curr_q,v_rewards).item())
                val_epoch_loss = np.mean(val_losses)
                val_epoch_acc = val_correct / len(X_val)

                train_loss_history.append(train_epoch_loss)
                val_loss_history.append(val_epoch_loss)
                train_acc_history.append(train_epoch_acc)
                val_acc_history.append(val_epoch_acc)

                progress_bar.progress(int(epoch/epochs*100))
                status_text.text(f"Epoch {epoch}/{epochs} | TrainLoss:{train_epoch_loss:.4f} TrainAcc:{train_epoch_acc:.3f} ValLoss:{val_epoch_loss:.4f} ValAcc:{val_epoch_acc:.3f}")

            # --------测试集评估----------
            all_preds=[]
            all_probs=[]
            q_net.eval()
            t_indices=np.arange(len(X_test))
            with torch.no_grad():
                for t_start in range(0,len(t_indices),batch_size):
                    tb_idx = t_indices[t_start:t_start+batch_size]
                    tb_x = torch.tensor(X_test[tb_idx],dtype=torch.float32).to(device)
                    tb_q = q_net(tb_x)
                    tb_probs = torch.softmax(tb_q,dim=1)[:,1]
                    tb_preds = tb_q.argmax(dim=1)
                    all_preds.extend(tb_preds.cpu().numpy())
                    all_probs.extend(tb_probs.cpu().numpy())
            all_preds=np.array(all_preds)
            all_probs=np.array(all_probs)
            all_targets=y_test

            acc=accuracy_score(all_targets,all_preds)
            prec=precision_score(all_targets,all_preds,zero_division=0)
            rec=recall_score(all_targets,all_preds,zero_division=0)
            f1=f1_score(all_targets,all_preds,zero_division=0)
            cm=confusion_matrix(all_targets,all_preds)
            tn,fp,fn,tp = cm.ravel()
            specificity = tn/(tn+fp) if (tn+fp)>0 else 0
            fpr,tpr,_ = roc_curve(all_targets,all_probs)
            roc_auc = auc(fpr,tpr)

            st.session_state["results"] = {
                "norm_signals":norm_signals,
                "fault_signals":fault_signals,
                "points_per_sample":points_per_sample,
                "train_loss_history":train_loss_history,
                "val_loss_history":val_loss_history,
                "train_acc_history":train_acc_history,
                "val_acc_history":val_acc_history,
                "acc":acc,"prec":prec,"rec":rec,"f1":f1,"specificity":specificity,"roc_auc":roc_auc,
                "cm":cm,"fpr":fpr,"tpr":tpr,"all_targets":all_targets,"all_preds":all_preds
            }
            st.success("训练完成！切换到【结果可视化】查看图表与指标。")

    with tab3:
        if "results" not in st.session_state:
            st.info("请先完成数据集加载与训练！")
            return
        res = st.session_state["results"]
        st.subheader("测试集诊断指标")
        col1,col2,col3,col4,col5,col6 = st.columns(6)
        col1.metric("Accuracy",f"{res['acc']*100:.2f}%")
        col2.metric("Precision",f"{res['prec']*100:.2f}%")
        col3.metric("Recall",f"{res['rec']*100:.2f}%")
        col4.metric("F1‑Score",f"{res['f1']:.4f}")
        col5.metric("Specificity",f"{res['specificity']*100:.2f}%")
        col6.metric("AUC",f"{res['roc_auc']:.4f}")

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans','Arial']
        plt.rcParams['axes.unicode_minus']=False
        fig,axes = plt.subplots(2,3,figsize=(18,10),dpi=150)
        fig.suptitle('ZN63 Vacuum Circuit Breaker MAX9814 Acoustic Diagnosis (VQC‑RL System)',fontsize=11,y=0.98)

        time_axis = np.linspace(0,0.6,res["points_per_sample"])
        axes[0,0].plot(time_axis,res["norm_signals"][0],label="Normal State",color="#1f77b4",lw=0.8)
        axes[0,0].plot(time_axis,res["fault_signals"][0],label="Linkage Blocked",color="#d62728",lw=0.8)
        axes[0,0].set_title("(a) Raw Acoustic Waveforms")
        axes[0,0].set_xlabel("Time(s)")
        axes[0,0].legend()

        freqs = np.fft.rfftfreq(res["points_per_sample"],1.0/50000)
        fft_norm = np.abs(np.fft.rfft(res["norm_signals"][0]))/res["points_per_sample"]
        fft_fault = np.abs(np.fft.rfft(res["fault_signals"][0]))/res["points_per_sample"]
        axes[0,1].plot(freqs,fft_norm,label="Normal Spectrum",color="#1f77b4",lw=0.8)
        axes[0,1].plot(freqs,fft_fault,label="Fault Spectrum",color="#d62728",lw=0.8)
        axes[0,1].set_xlim(0,10000)
        axes[0,1].set_title("(b) Frequency Spectra(FFT)")
        axes[0,1].legend()

        ep_range = list(range(1,len(res["train_loss_history"])+1))
        axes[0,2].plot(ep_range,res["train_loss_history"],label="Train Loss",color="#2ca02c")
        axes[0,2].plot(ep_range,res["val_loss_history"],label="Val Loss",color="#ff7f0e",linestyle="--")
        axes[0,2].set_title("(c) Huber Loss Convergence")
        axes[0,2].legend()

        axes[1,0].plot(ep_range,[x*100 for x in res["train_acc_history"]],label="Train Acc",color="#2ca02c")
        axes[1,0].plot(ep_range,[x*100 for x in res["val_acc_history"]],label="Val Acc",color="#ff7f0e",linestyle="--")
        axes[1,0].set_ylim(40,102)
        axes[1,0].set_title("(d) Accuracy vs Epochs")
        axes[1,0].legend()

        sns.heatmap(res["cm"],annot=True,fmt="d",cmap="Blues",ax=axes[1,1],cbar=False,
                    xticklabels=["Normal","Fault"],yticklabels=["Normal","Fault"])
        axes[1,1].set_title(f"(e) Confusion Matrix, Test Acc:{res['acc']*100:.2f}%")

        axes[1,2].plot(res["fpr"],res["tpr"],color="#d62728",label=f"AUC={res['roc_auc']:.4f}")
        axes[1,2].plot([0,1],[0,1],"k:")
        axes[1,2].set_title("(f) ROC Curve")
        axes[1,2].legend()

        plt.tight_layout(rect=[0,0,1,0.96])
        st.pyplot(fig)

if __name__=="__main__":
    main()
