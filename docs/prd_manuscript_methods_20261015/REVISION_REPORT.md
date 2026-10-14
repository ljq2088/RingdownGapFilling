# PRD 稿件方法与流程图修订记录

日期：2026-10-15。在线稿：[Overleaf 修订项目](https://cn.overleaf.com/project/6ac83816750c2cce452054d2)。

## 已完成

- 新增两幅独立绘制的矢量图。图 1（在线 PDF 第 4 页）展示白化、DGF 双分支、融合与解码、mask/A/E 残差模块、低秩基和噪声条件系数头，以及逆白化后的物理输出。图 2（第 6 页）展示源级数据划分、增广、固定校准、三阶段训练、验证决策、早停及冻结后的新独立测试。
- 正文补充了输入与特征尺寸、实际注意力运算、连续相位特征、参数数量、优化目标、原始权重与 EMA 的选择规则和可恢复检查点。
- 数据描述采用当前代码的源质量、红移、Kerr 模式、有限臂 LISA 响应、二代 TDI A/E 分数频率和仪器噪声约定。固定纯噪声校准包含 512 条裁剪序列；其前 128 条用于预期噪声 RMS。
- 主实验缺口为 64、66、…、104 个样本，采样率 2 Hz，窗口 1056 点/528 s。20% 保留为引言中的背景示例。
- 观测采用 Gh+n。主指标采用固定时间和相位的有符号联合 A/E PSD mismatch，逐源等权平均。
- 摘要、动机、方法、讨论与实现附录同步修订；保留标题、作者、DGF 双分支和原稿的主要研究任务。

## 结果更新约定

本次修改保留原有结果表的数值和原有结果图文件；正文及附录明确说明这些条目的早期实验范围。稿件首页含可见的修订状态说明。当前物理 A/E 训练及冻结后的独立测试完成后，统一替换结果表、结果图、摘要结论和讨论中的定量表述。

当前训练的配置预算写入方法和流程图；最终执行轮数及最终选定配置随独立测试结果补齐。本次写稿时微调进入第 33 轮。训练运行文件保持原有 SHA-256 校验值。

## 文献与图解参考

通过 Consensus 检索 TSMixer 和 TimeMixer，并核对原论文。图解采用多尺度分支、模块尺寸和分阶段流程的组织方式；模块及连接关系由本项目代码确定。

- [TSMixer 原论文](https://arxiv.org/abs/2303.06053)，对应 [Consensus 条目](https://consensus.app/papers/tsmixer-an-allmlp-architecture-for-time-series-chen-li/6f0ea0c3da0c51388b04be5db402ece3/)。
- [TimeMixer 原论文](https://arxiv.org/abs/2405.14616)，对应 [Consensus 条目](https://consensus.app/papers/timemixer-decomposable-multiscale-mixing-for-time-series-wang-wu/f9e07ccf044f5da198b61876f1e0f261/)。
- [Kamaretsos 等模式激发拟合](https://arxiv.org/abs/1107.0854)。
- [有限臂 LISA 响应实现论文](https://arxiv.org/abs/2204.06633)。
- [PyCBC 波形与 ringdown 文档](https://pycbc.org/pycbc/latest/html/pycbc.waveform.html)。
- [LISA Analysis Tools 实现](https://github.com/lisa-analysis-tools/lisa-analysis-tools)。

## 校验与归档

- 本地 TeX Live 编译成功；Overleaf 最终在线编译得到 13 页。
- 已从 Overleaf 下载最终项目源码及 PDF，正文、参考文献与两张图逐字节匹配本地稿。校验记录在 `server_verification.json`。
- 两张图所在的最终在线 PDF 页面已渲染并检查，模块标签和图注完整，图中文字及箭头无相互遮挡。
- 修改前完整云端项目备份为 `overleaf_before.zip`；修改后云端源码备份为 `overleaf_after.zip`。原 Overleaf 项目 69bb9d12f9a917663fb9b9c0 保持原有内容。
- `source_edited` 保存 TeX、BibTeX 和两幅图的 PDF/SVG/PNG；`make_method_figures.py` 保存作图源代码。独立图 PDF 使用嵌入字体，可用于论文排版与导出。
- 稿件数值更新完成后，再整理最终结果章节和浮动图布局，并移除首页的修订状态说明。

## 图解风格补充修订

按用户的进一步要求，参考旧稿 `Img/DGF_architecture.png` 和 `Img/DGF_pipeline.png` 的输入分叉、编码器与解码器分区、波形与频谱小图和切片示意。新版由 `make_illustrated_figures.py` 生成，使用可编辑 SVG、嵌入字体 PDF 和 PNG。

结构图加入两通道带缺口观测、Morlet 时频图、db4 子带与重叠切片、注意力层堆叠、网络节点、基矩阵及系数、噪声谱和相位示意。训练图加入三种训练视图、白化前后噪声谱、训练模块、验证平台期和冻结模型小图，并补全微调到验证的主路径。

这些图形是方法示意。图注明确标记波形、频谱、子带、网络、优化曲线和比较图标的示意性质，定量结果仍由独立实验报告。当前图中保留实际代码的 CWT、逐通道注意力和学习式波形合成命名。训练与评估协议、结果数值保持上一轮修订内容。
