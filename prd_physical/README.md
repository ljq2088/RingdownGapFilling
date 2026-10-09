# 原 DGF 的物理数据迁移实验（旧分支）

本目录延续 arXiv 2511.00834 的 DGF 双分支主干，使用校正物理约定的数据做迁移训练；不是此前新分支的 U-Net 实验。

用户最新指示：不考虑 G(h+n)。本实验严格沿用 `observation = (1-mask)*truth + noise`。mask=1 只抑制信号，保留噪声；文稿不得把此结果描述为真实丢失的 spacecraft samples。

## 冻结实验

- 数据来源：相邻主仓库已生成的物理 A/E TDI2 数据，只读复用。源总质量 log-uniform 1e5–5e6 Msun，q=m_light/m_heavy uniform 0.3–1，z uniform 1–6；无自旋 progenitors，各向同性取向。LAL 残余质量/自旋，探测器系 remnant mass = source remnant mass ×(1+z)，FlatLCDM luminosity distance，PyCBC 220/210/330/440 fundamental QNMs，Kamaretsos 1107.0854 Eqs31–34 的非自旋振幅拟合。
- LISA fastlisaresponse 有限臂响应，等臂理想化二代 TDI；A=(Z-X)/sqrt2，E=(X-2Y+Z)/sqrt6；信号和 scirdv1 仪器噪声均为 fractional frequency。无银河前景、无非平稳噪声/非等臂噪声相关建模。模式相位是简化的 m*orbital_phase；短 ring-up 是数值延拓，不能称作 IMR。
- 复核六个 train 记录（含原来重生成的前缀边界）与冻结 simulator 逐位一致。早期 train HDF 配置缺 numerical_ringup 键，记录这一元数据缺陷，不能据此声称内容不同；没有声称逐位复核全部 60000 条。
- 选择预声明 train 前12000条、validation 前1500条，test全6000条；种子不相交。没有按预测表现筛选。父波形有32768点，DGF 每段仍为1056点/528 s，gap仍66点/33 s。通过已提供的 gap 起点及固定随机数选取 segment，局部 gap 起点132–263。父 gap 分布在注入峰值附近；模型推理只接收 observation、已知 mask、noise-only PSD，不接收干净 peak 或参数。此有限段试验不声称恢复完整最长 ringdown。
- 512个独立纯噪声有限 crop 估计固定 PSD，训练/验证/测试共享；直接 overlap 不最大化时间/相位，只排除 DC、Nyquist，逐 source 联合 A/E 后做总体均值及 bootstrap，保留全部条目。有限窗口 PSD 的对角权重是约定的指标，不等于完整相关噪声似然。
- raw 与 white 两版本：同一 DGF+25489参数小 mask adapter、相同初始化/seed/训练集/预算/物理空间损失。raw 仅 noise-only 常数单位归一化；white 额外做可逆 FFT PSD 白化并恢复为物理单位。二者 CWT 振幅/相位均从 observation 在 working units 下除以其自身 RMS 得到，解决原物理量幅值过小使 CWT 幅值输入近于零的问题；这属于明确的预处理更新。
- 总参数9209603。base AdamW lr2e-5，adapter lr1e-4；验证均值 Plateau 调度 factor0.5/patience20。batch128个源（每源A/E共享权重）。每个版本 max600/min100 epoch，连续100 epoch验证改善小于1e-5后早停，保存最佳权重及可恢复 last checkpoint。
- 验证仅按全部1500条的 joint A/E 直接 PSD mismatch 选 epoch 和 raw/white 首选；不以测试选模型。两个版本选完后才首次读取 test truth/noise。
- 测试同条件对照：两个物理迁移版本、冻结原 DGF、此前最新 augmented checkpoint、线性插值、观测本身；附直接 time mismatch、Hann PSD mismatch、相对幅值/波形误差与峰值落 gap 子群。bootstrap 按源配对，不能冒称 training-seed 不确定性。
- 历史旧合成数据均值0.00536877只作背景；物理数据均值与其不同分布，不能直接当成公平的百分比改进。新物理测试的公平进步要求相对于同组数据的旧 checkpoint 直接 mismatch 均值降低且配对 CI 支持。

## 已通过的前置验证

已有 augmented 预测8条的复现 relative L2 = 1.59e-7。物理 gates 含 LAL/PyCBC QNM、distance线性缩放、红移、padding/interpolation收敛、噪声单边 PSD/方差、A/E独立噪声统计。

白化往返 relative L2 <1.9e-7；物理/working直接 overlap差 <5.6e-10。相位、正振幅尺度、符号范围、white PSD Parseval、整体PSD缩放、低PSD错误频点加大误差等约定测试通过。raw/white 同物理损失差5.8e-8。

16源25步学习探针有有限梯度并降低训练损失；不当作性能结果。验证人群的 finite-crop diagonal-PSD SNR中位55.8，5%分位12.4，minimum3.3；时域信号/噪声能量幅度比中位0.052。未剔除低SNR条目。

## 运行

在旧 worktree `/home/ljq/code/RingdownGapFilling_PRD` 的 `gw_ml_env` 下：

```
PYTHONDONTWRITEBYTECODE=1 python -m prd_physical.run
```

输入父 HDF 位于只读使用的 `/home/ljq/code/RingdownGapFilling/FillingGap/results_physical_retraining_20261009`。原权重已复制到旧 worktree ignored artifact 路径；代码不修改权重。运行产物为 `FillingGap/results_prd_physical_transfer_20261009/`；`status.json`、`raw/history.json`、`white/history.json`、`final_results.json` 给出实际状态。创建 `STOP` 可在 epoch末暂停，不读取 test；删除STOP后原命令恢复。源 hash 不匹配时拒绝恢复。

本进程限制GPU allocator为20%（约3.2GiB），不停止其他已有训练任务。128源 throughput probe峰值2.41GiB；该短探针吞吐随其他GPU任务变化，只作batch sizing。训练 ETA须按完整 epoch实测更新。

## 文稿边界

Overleaf修订项目仍为 https://cn.overleaf.com/project/6ac83816750c2cce452054d2 。目前其中经核验的结果来自旧基准。本轮物理训练完成且独立测试通过之前，不更新该项目的数值结果或宣称性能已改善；原cloud/arXiv保持。后续把物理迁移作为原方法的扩展，保留标题、作者、DGF主干和文章身份。

振幅拟合原文已核对：https://arxiv.org/pdf/1107.0854 ，Eqs31–34确为0.25 exp(-Q/7.5)及21/33/44相对振幅；Q=m_heavy/m_light与代码q倒数对应，参考时刻10M after22 luminosity peak，非任意辐射能量直接作为 strain。

## 用户要求的资源调整（2026-10-09，raw epoch37后）

系统检查：WSL可用RAM约27GiB、GPU空余约10GiB。现将每版本约1.12GiB训练/验证特征常驻GPU；每worker allocator上限从20%提升至40%，GPU缓存前预留3GiB激活预算及4GiB全局空余。每epoch检查CUDA和nvidia-smi全局空余；不足4GiB则退回CPU缓存。缓存等价、预测逐位一致、CUDA RNG不变、保存checkpoint未修改、主动提高门槛触发offload均通过实测。单worker GPU训练峰值约3.46GiB。

raw从epoch37的last checkpoint继续，white同时独立训练；每版本batch128、数据、目标、seed、优化器/学习率、选模及早停预算保持一致。per-variant文件锁防止两个进程写同一版本；white worker有独立status/log/checkpoint，永不读取test。主进程只有在两个版本选模结束后才统一读取test。并行仅改变执行顺序和资源使用，原“之后自动训练white”的串行说明由本段覆盖。

原执行snapshot与新snapshot及checkpoint hash保留在结果目录resource_adjustment_20261009中。资源变更前后loss/metric/eval/test函数AST一致，模型和数据来源文件hash不变。
