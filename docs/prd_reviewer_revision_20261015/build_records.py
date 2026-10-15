from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
a=json.loads((P/'implementation_audit.json').read_text())
a['software_versions']['lisa-analysis-tools']='1.2.8'
a['software_versions']['pytorch_wavelets']='1.3.0'
b=json.loads((P/'basis_prior_corrected/basis_results.json').read_text())
d=json.loads((P/'additional_diagnostics.json').read_text())
f=json.loads((P.parent/'figure4_noise_overlay_20261015/figure_audit.json').read_text())
manifest=dict(schema_version=1,computational_source_commit='ae47cce8a5c0048951d63d4040bfc4a37d0a712d',software=a['software_versions'],assets=a['assets'],source_seeds=a['source_seeds'],test_seeds=dict(first=2038200000,last=2038205999,n=6000),observation='d=G*h+n; suppression at synthesized TDI-channel level before whitening',metric=dict(name='direct signed joint A/E PSD-weighted mismatch',alignment='fixed arrival time and phase',excluded_bins=['DC','Nyquist'],population='all test sources; SNR>=10 subset identified separately'),noise_rms=dict(independent_realizations=128,seeds=[2027100000,2027100127],crop_start_sample=6000,n=1056,aggregation='square root of mean per-segment mean square after whitening and excluding DC/Nyquist',rms=[.9993543067089424,.9990880986927643]),transforms=dict(morlet=a['morlet'],temporal_wavelet=dict(package='pytorch_wavelets',wavelet='db4',levels=8,boundary='zero',channels=['A8','D8','D7','D6','D5','D4','D3','D2']),phase_display=a['phase_plot'],periodogram=a['periodogram']),figure_examples=f,linear_benchmark=b,additional_diagnostics=d,availability='Checkpoints, calibration arrays and evaluation records available from the authors on request. Source and asset identifiers are recorded here; a public versioned checkpoint release remains to be prepared.')
(P/'source/reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

items=[
(1,'完成','删除无明确来源的 20% 数字，直接定义 32-52 s / 528 s。'),
(2,'完成','删除 consequently 引出的概念罗列，直接列评价量。'),
(3,'完成','明确固定信号基及依赖观测的系数修正。'),
(4,'完成','摘要注明每个通道各自的干净解析包络峰值。'),
(5,'完成','补明自旋权重 -2 球谐函数、探测器系质量/距离因子和复极化约定。'),
(6,'完成','写出 ring-up 的指数、tau/10 时间尺度、开始时刻和截断。'),
(7,'完成','明确指标在 528-s 窗口上计算。'),
(8,'完成并修正事实','RMS 来自另 128 个独立噪声实现，明确汇总均方后开方；不属于 PSD 样本前缀。'),
(9,'完成','定义父序列实通道峰值、离散均匀偏移、裁剪和变长缺口起点。'),
(10,'完成','补 Morlet mu、1-16 尺度、观测 RMS 归一化、L1 约定及 reflection padding。'),
(11,'完成','给出细化预测 v、适配系数 beta 和 b 的运算关系。'),
(12,'完成','新增表 II，列出 262+16+192 输入、分项维数、归一化、频率组及截断。'),
(13,'完成','用 amplitude-weighted circular phase representation 替代 continuous phase。'),
(14,'完成','补出 P0 b 在 span(B) 内的条件；图注限定零初始化时刻。'),
(15,'完成','改为 PSD-weighted network SNR，注明使用完整目标。'),
(16,'完成','E_S 称 normalized noise-weighted waveform error。'),
(17,'完成','定义三视图的共同中心、独立噪声、固定宽度/符号抽样和跨 epoch 复用。'),
(18,'部分完成','补当前物理数据适配阶段、目标、源集合、选择方式与检查点链。旧初始化按固定权重提供；原始历史随机划分索引与完整早期运行记录仍待补齐。'),
(19,'完成','将条件统计解释为组内源、噪声与缺口位置联合分布。'),
(20,'完成','正文定义 6000 训练源上三类特征组合及验证选择；流程图保留主要阶段。'),
(21,'按现有展示要求处理','保留两例清楚的波形对比；在图注指向实际例图条件及伴随 manifest，避免典型性与总体代表性暗示。'),
(22,'完成','相位示例表述限定一个源的 A 通道。'),
(23,'完成','图 3 图注明确为经验累积分布函数，原曲线及数据保留。'),
(24,'完成','norm bias 定义为两波形范数之差的相对量。'),
(25,'完成','缺口误差分母写为缺口内目标的欧氏范数。'),
(26,'完成','给出两通道共用 H_TDI=max|h| 的明确归一化。'),
(27,'完成','按真实绘图代码写 unwrap(arg z)+Arg(zhat z*)，采用 pi 跳变判据。'),
(28,'完成','删除图 5 末尾重复的全体样本数声明。'),
(29,'完成','删除未经定义的 tail-relative residuals grow，限定对两例绝对值图的观察。'),
(30,'完成','尾部差异只对应所展示案例；各指标的贡献依赖具体权重。'),
(31,'完成','参数推断改为后续工作，似然与现有观测模型对应。'),
(32,'完成','未来工作聚焦响应误差、非平稳噪声及训练模态关系之外的信号。'),
(33,'完成','图 7 明确为 absolute values，保持与解析包络的术语区分。'),
(34,'完成','定义 boxcar、constant detrend、单边密度及逐源平均；图 8 加目标 PSD。'),
(35,'完成','删除论文中 unused attention tensors 的工程说明；存储参数统计保留其原含义。'),
(36,'完成','删除底层存储 clone 叙述，保留两个映射为独立参数。'),
(37,'部分完成','加入实际源版本、检查点/资产哈希、软件版本及取得方式，云端附科学复现 manifest。公开版本化完整资产包仍待发布。'),
(38,'完成','每源每通道一个相位值，删除 preserves source counts 式解释。'),
(39,'完成','参考模型以 a1e9959a7a95 检查点及模块组成标识。'),
(40,'完成','定义 DGF-reference 差值方向、源级配对 bootstrap、5000 次及 percentile 构造。'),
(41,'完成','软件文献增加真实版本和已核对的版本定位链接。'),
(42,'完成','统一活跃引文的期刊缩写，保护 Fourier、Bayesian 等专名。'),
]
lines=['评审意见处理记录 | 2026-10-15',
'来源：https://chatgpt.com/s/cx_6ad0128d5e6c8191b5ad5b110fab2d12',
'本轮保持最终 DGF 检查点、6000 测试源及所有既有核心统计。新增计算沿用相同观测、PSD、时间和相位。',
'', '逐页意见（编号与分享页面第二轮 1-42 条对应）：']
lines += [f'{i:02d} | {status} | {text}' for i,status,text in items]
lines += ['', '新增验证：',
'同一训练信号基的线性正则拟合，另含 4 个端点方向；使用训练方向系数方差作为先验。lambda 由完整验证集选择。SNR>=10 线性均值 0.04131，DGF 0.00532；全体线性均值 0.05193，DGF 0.00852。',
'基的干净目标投影：rank 32/64/128 的全体均值 0.03690/0.00410/0.00016450；属于表示误差诊断。',
'峰值位于缺口内、SNR>=10 的 1885 源：缺口内包络加权相位 RMS 均值 4.92 度、中位数 3.89 度、95% 分位 11.61 度；完整峰值入缺口总体均值 5.51 度。',
'新增 gap 信号范数比例、Tgap/tau220、E_S 及目标 PSD；独立 Cholesky 求解与物理单位 mismatch 复算通过。',
'', '进一步实证工作：',
'相同振铃族的物理参数拟合基准、匹配预算/多种子的神经网络消融、连续 IMR 历史、独立模态扰动及后验覆盖率检验。当前文字明确这些工作属于后续检验。',
'现有 G*h+n 观测模型与图例选择遵循本会话已确定范围；低 SNR 子总体保留于总体统计，详细分布与失败案例不单独展开。',
'', '计算核验说明：',
'采用 basis_prior_corrected/ 结果入稿。先期端点拟合探索中的评价口径经独立复算发现需剔除 DC/Nyquist，已修正并重选验证参数；先期 basis_prior/、basis_endpoints/ 结果作为调试记录保留，排除于正文、表格和最终 manifest。',
'冻结训练运行时与检查点保持原状；新增脚本单独存放在本次文稿目录，计算使用 CPU。']
(P/'review_actions.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(P/'review_actions.json').write_text(json.dumps([dict(index=i,status=st,text=text) for i,st,text in items],ensure_ascii=False,indent=2),encoding='utf-8')
(P/'implementation_notes.txt').write_text('''Implementation notes retained outside manuscript prose

The stored 9,589,685-parameter count includes allocated multihead-attention tensors not consumed by the forward pass. The actual attention is specified by the manuscript equation softmax(ZZ^T/sqrt(64)) Z, with no learned QKV projection in that operation.

PhysicalDGF.__init__ clones the decoder inverse_linear storage before loading the original state dictionary. The encoder embedding and decoder patch maps hold independent trained values. The frozen loading implementation remains unchanged.

The 128 noise-RMS realizations are independent of the PSD calibration realizations. For each channel, RMS is the square root of the mean of segment mean squares after whitening and removing DC/Nyquist. The temporal transform returns A8,D8,...,D2 from an eight-level zero-padded db4 decomposition.

Reproduction starts from provided legacy weights. The original historical split indices and complete historical run configuration were not recovered in this audit. The current physical A/E adaptation source sets, objectives, validation rules and fixed initialization assets are recorded in the manuscript and accompanying manifest.
''',encoding='utf-8')
print('Scientific manifest and 42-item response record written.')
