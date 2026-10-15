评审修订记录，2026-10-15

最终云端稿件：overleaf_final.pdf；完整云端源码：overleaf_after.zip。
review_actions.txt 逐项记录评审意见、已完成更改和待补实验。
source/ 为同步后的项目；verification.json 记录文件哈希与云端构建标识。
basis_benchmark_prior.py 与 diagnostics.py 为本轮新增 CPU 计算。有效基线结果位于 basis_prior_corrected/；正则参数由 1500 验证源选择，测试为冻结的 6000 源。
主模型、原统计、训练运行时代码和活跃训练任务保持原状。
本目录的新增计算仅覆盖信号基拟合、表示误差及缺口内相位诊断。物理参数拟合、神经网络消融、连续 IMR、模态扰动与后验覆盖率留待后续实验。
