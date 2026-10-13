# 6%–10% 变长缺口优化，2026-10-13

状态：长跑启动；最终结果由新的独立测试集给出。

## 实验范围

沿用原 DGF 双分支结构与 1056 点、2 Hz 的 528 秒输入。缺口宽度在 64、66、…、104 点的 21 个偶数宽度中均匀取样，对应 32–52 秒、窗口长度的 6.0606%–9.8485%。100 点对应 50 秒、9.4697%，覆盖旧 revise 的典型宽度。每个源的缺口中心保持不变。观测为 G h+n，缺口内保留噪声，模型输入包含已知缺口掩码。

60,000 个训练源，每源两组观测，共 120,000 次观测/轮。第一组使用原噪声与变长缺口；第二组使用新的独立物理噪声、另一组变长缺口与随机整体符号。训练输入的 CWT、物理频段 CWT 和 db4 特征全部重新计算。基底沿用仅由训练目标拟合的 128 维联合 A/E 基底。

模型选择使用 1500 个验证源的随机变长缺口直接 PSD mismatch 全样本算术平均，逐轮比较在线权重和 EMA。固定 33 秒与 50 秒验证同时记录。全样本统计保留所有源。SNR≥10 作为事先规定的模拟条件分组，逐组报告样本数和同组基线。

联合微调 DGF 主干、解码器、缺口残差、A/E 残差、物理频段支路和低秩修正头。批量128，最多500轮，至少100轮；100轮无超过1e-6的改善后早停。平台15轮后学习率减半，EMA=.995。检查点包含原始/EMA权重、优化器、学习率调度器、CPU/CUDA随机状态及打乱顺序状态。

模型选择完成后冻结检查点并生成新的6000源测试集，种子2036200000–2036205999。主测试为均匀变长缺口，固定33秒和50秒为配对的次级测试。所有比较使用相同源、噪声、缺口和物理指标，无时间/相位最大化。

## 表述计划

背景英文候选：

> Interruptions occupying a sizeable fraction of a short ringdown analysis window, for example 20%, motivate reconstruction methods that exploit temporal and spectral correlations.

实验方法英文候选（训练和测试完成后同步）：

> We consider single signal-suppression intervals with variable widths of 64–104 samples, corresponding to 32–52 s and approximately 6%–10% of the 528-s analysis window. The widths are drawn uniformly from 21 even integer values. Instrument noise remains present during the suppressed-signal interval, and the gap mask is supplied to the reconstruction model.

论文摘要、结果和结论的适用范围与该实验保持一致。20%保留为背景中的示例性较长缺口情形。变长缺口训练正在进行，最终性能数字和性能结论待独立测试后填写。

## 启动前验证

冻结权重复现误差：每源直接 mismatch 最大差1.21e-8。白化工作空间和物理空间的单精度指标差小于1.54e-7。真实变长缺口128源批量的前向、反向和优化器更新有限；覆盖全部21个缺口宽度。训练目标逐值复现已冻结缓存。检查点的权重、优化器状态和随机状态读写逐值一致。

预热后的中位训练步耗时0.188秒，120000观测/轮约176秒纯训练计算。数据传输、双权重验证、检查点保存计入正式轮耗时。预计完整轮约3.5–5分钟；100轮约6–8.5小时，500轮约29–42小时。训练时间将按实际轮耗时更新。

Overleaf副本当前显示离线，云端表述修改和编译验证待恢复连接。原Overleaf项目与arXiv版本保持可追溯。
