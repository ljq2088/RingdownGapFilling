# 全量低秩残差训练启动报告

北京时间 2026-10-13T09:48:30，后台协调器PID 19246，代码提交 `9b1c6bb`。工作树和分支保持 `/home/ljq/code/RingdownGapFilling_PRD`、`codex/mask-aware-ringdown-optimization`。

当前状态为三组完整观测的特征准备，view=0，已完成12064 / 60000源，实测吞吐约79.6源/秒。两条既有GPU长跑继续运行；本轮使用CPU，Python运行源码哈希保持冻结值。

本轮覆盖全部60000个训练源，每源每轮三次呈现：增广组为原始观测加两份独立LISA噪声观测；对照组重复原始观测三次。每轮180000次源呈现，batch128，两组每轮更新数一致。冻结DGF主干，训练66688参数的128维系数残差头。所有1500例验证样本参与直接PSD mismatch的算术均值评价，时间和相位固定。

两组各最多800轮、至少80轮、patience120、min_delta1e-6；保留EMA、最佳权重与逐轮完整恢复状态。两组完成验证选择并冻结后，自动生成新的6000例物理测试，对同条件基线、固定投影及训练后模型进行评价。正式测试样本当前保持封存。

按已经实测的特征准备速度与两个并行CPU计算基准，特征准备预计于10月13日10:25–10:35完成。完整训练的每轮耗时估计为25–40秒；当前估计来源于实际batch吞吐外推，包含缓存访问、验证和checkpoint开销的余量。

若最佳验证checkpoint落在第20–80轮，随后连续120轮达到早停条件，预计训练及独立测试于10月13日11:45–13:15完成。若持续改善并用满800轮，预计10月13日16:30–20:00完成。验证改善的持续时间决定早停点；正式全量epoch日志提供后续时间估计依据。

资源检查时WSL可用RAM约23GiB，GPU空余约5GiB。本流程的CUDA参与为零，CPU worker各2线程，共享只读memmap；可用RAM低于6GiB时等待。后台进程独立于当前会话执行。

自动流程入口 `python -m prd_lowrank.launch`。状态：`FillingGap/results_prd_lowrank_20261013/status.json`；模型各自的history.json、best.pth、last.pth位于同目录下的repeated_control和noise_augmented子目录。最终结果由final_results.json和final_metric_audit.json记录。前置检查和完整数据合同见PLAN_AND_PREFLIGHT.md。
