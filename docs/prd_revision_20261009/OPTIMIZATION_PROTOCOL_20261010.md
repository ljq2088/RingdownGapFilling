# DGF 继续优化协议（2026-10-10）

保持旧分支 `codex/mask-aware-ringdown-optimization` 和原 DGF 双分支主干。上一轮白化模型物理测试 PSD mismatch 均值为0.01504264；旧合成基准0.00536877尚未达到，两者不同分布不能作公平性能排序。保持G*h+n，不考虑G(h+n)。保留全部样本。

验证集诊断：SNR<10的44个源均值0.15109；源质量1e5–3e5的421个源均值0.03611；最后四分之一片段占真实能量0.37%、重建误差能量16.00%。这些是验证诊断，不是删除样本的依据。增加训练覆盖，并测试跨通道补偿和尾段约束。

两组同预算、同60000源训练集、同固定1500源验证集，从上轮white最佳checkpoint初始化。continuation保留原结构和原物理损失；joint_hann增加45410参数的A/E联合残差模块（零输出初始化），并附加权重0.15的Hann直接mismatch及平滑幅值损失。两者共同引入EMA（decay0.995），每epoch评估在线/EMA，按全部验证源直接PSD mismatch选权重。Hann是副指标，不能改变主指标来制造改善。

batch128；AdamW base2e-6、原adapter1e-5、新joint1e-4；weight_decay1e-4；Plateau factor0.5/patience15；max800/min80/patience120/min_delta2e-6。最佳模型、在线模型、EMA、优化器、调度器、CPU/CUDA/NumPy随机状态及独立shuffle generator均保存。没有把短探针训练过的权重用于正式初始化。

新增6000源测试使用2030200000..2030205999种子，在两组完成选模并冻结首选后才生成。所有候选和上轮white在同一新测试集评估直接PSD、时域、Hann、L2及峰值落gap子群，源级配对bootstrap5000次。旧测试只作已报告结果，不进入本轮调参；新测试不得用于后续选超参数而仍声称独立。

源质量1e5–5e6 Msun、q=m_light/m_heavy0.3–1、z1–6；观测528s、mask33s；理想等臂有限臂长LISA TDI2 A/E fractional-frequency；独立instrument noise和原冻结noise-only PSD。幅值/相位仍为合成处方，不能声明NR完整波形、真实丢失遥测、参数估计或无毛检验。

两组并行；共享约9.44GiB可回收文件映射，不复制两套完整训练缓存。每worker GPU cap30%，准入后缓存8192条train及1500条val。保留4GiB全局空余和3GiB激活预算；低VRAM释放输入缓存；低RAM/VRAM等待或记录失败，保留可恢复checkpoint。运行通过WSL后台进程，与会话结束独立；系统注销、WSL重启会中断进程，应由保存的last恢复。

预检查：新缓存所有文件已哈希；6个训练样本（含新增边界）input/target/mask/CWT逐位重现，固定db4 GPU批形状误差<8.4e-7；原模型与joint零head输出逐位相同；HannGPU计算与独立物理计算吻合；同样128源混合GPU/CPU缓存批、预测、offload后的值逐位一致且CUDA RNG不变。24步固定样本学习检查中，两组确定性eval-mode训练损失下降；带dropout的批损失有波动，不作为短训练未学习的证据。短检查不是最终性能结论。

代码入口：`python -m prd_optimize.run`。产物：`FillingGap/results_prd_optimization_20261010/`。创建产物目录内STOP，在epoch结束后暂停；删STOP并重启同入口可恢复。运行冻结源SHA不一致时拒绝恢复。正式训练的早期实测速率决定ETA，不能用短探针声称完成时间或性能。
