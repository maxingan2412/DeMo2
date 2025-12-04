# SDTPS 整合审阅

目标：只保留 SDTPS（基于 SEPS 的 TokenSparse），用 RGB/NIR/TIR 三模态的 score 组合驱动 Gumbel 选择，在 HDM 之前完成，并替换掉 HDM/ATM 流程。以下对现有实现进行对照审阅。

## 已核对的文件
- `seps_modules_reviewed_v2_enhanced.py`：原 SEPS 参考实现。
- `modeling/sdtps.py`：当前多模态 SDTPS 版本（TokenSparse + MultiModalSDTPS）。
- `modeling/make_model.py`：SDTPS 在 DeMo 主干中的接入位置。

## 现状与偏差
1) **Score 计算来源**  
   - 当前 `TokenSparse.forward` 使用 `(1-2β)·s_pred + β·(s_m2 + s_m3 + 2·s_im)`，其中 s_im 来自 `_compute_self_attention`（patch 与本模态全局均值的余弦），s_m2/s_m3 来自其他模态全局均值与目标模态 patch 的余弦，已符合“四路 score” 思路。  
   - 但自注意力/交叉注意力未加 `torch.no_grad()`，会产生梯度与显存开销，和 SEPS 注释版存在差异。

2) **Gumbel-Softmax 可微性**  
   - `use_gumbel=True` 时仍然用 Top-K 索引 `keep_policy` 进行 `torch.gather`，软 mask 只作用在 `score_mask`，梯度不会穿过 token 选择（与 SEPS 注释的已知问题一致）。如果需要可微的 token 选择，需要用 mask 加权方式而不是硬 gather。

3) **HDM/ATM 替换路径**  
   - `make_model.py` 中 `USE_SDTPS` 只是在原有 HDM/ATM 旁边新增分支；若配置仍打开 HDM/ATM，会同时走 MOE 和 SDTPS，而不是“替换”。需要一个明确的互斥/优先级开关来禁用 HDM 和 ATM。

4) **输入特征位置**  
   - SDTPS 使用 `RGB_global = rgb_reduce(cat(global, local))` 等作为全局引导，这与需求“用 reduce 后的全局特征做 cross-attn”一致。patch 特征使用的是 backbone 的 token (`*_cash`)；未见在 HDM 之前共享 attn map 的其他依赖。

5) **聚合与输出形式**  
   - 当前 SDTPS 只做 Top-K 选择 + extra token，随后简单平均得到分类特征；没有 SEPS 的 TokenAggregation/HRPA 等后续对齐。若只需要“token selection + 增强”可接受，但与“尽量保留原设计思路”相比，聚合与对齐能力缺失。

6) **推理/返回值**  
   - 推理时 `return_pattern`=2 仅返回 SDTPS 拼接特征，=3 返回原始全局 + SDTPS 拼接；未暴露单模态增强 token，用于下游可解释性或进一步融合可能不足。

## 建议/待确认
- 是否需要按 SEPS 习惯将自/跨注意力计算包在 `torch.no_grad()` 内以减轻梯度干扰。
- 是否需要修正 Gumbel 分支为可微的 token 选择（mask 加权而非硬索引）。
- 确认在 `USE_SDTPS=True` 时默认关闭/跳过 HDM 与 ATM，避免双分支并行的歧义。
- 若想更贴近 SEPS 设计，可考虑补充轻量聚合（如 TokenAggregation）而不只是均值池化。

---

## 第二轮审阅（当前使用 `modeling/sdtps_complete.py`）

新增核对文件：
- `modeling/sdtps_complete.py`（新的 SDTPS 完整版，包含 TokenAggregation、no_grad 修复）
- `modeling/make_model.py`（改为引入 `sdtps_complete.MultiModalSDTPS`）

改进点（相对上一版）：
- 自/跨注意力计算已包裹 `torch.no_grad()`，显存和梯度干扰风险降低。
- 增加了 TokenAggregation 阶段（N → N_s → N_c），更贴近 SEPS 的两段压缩；K/N_c 计算按 stride 推导的 patch 数量。

仍存在 / 新发现的问题：
1) **Gumbel 仍是硬 Top-K**  
   - `keep_policy` 始终由原始 score Top-K 决定，Gumbel 噪声只参与 `score_mask`，不影响被 gather 的 tokens；选中索引固定，选择路径不可微。想要可微，需要软选择（mask 加权）或将噪声纳入排序。

2) **HDM/ATM 并行问题未解**  
   - `USE_SDTPS` 仍与 `HDM/ATM` 并行，配置若未手动关闭 HDM/ATM 会同时跑 MOE 和 SDTPS，违背“替换”预期。

3) **输出与原 SEPS 的对齐缺失**  
   - 仅做 TokenAggregation + extra token 后的均值池化用于分类，未包含 HRPA / 对齐损失；如果需要原论文的对齐增益，需要后续 HRPA 或对比损失。

4) **Gumbel 梯度覆盖面有限**  
   - 即便开启 Gumbel，梯度只流向 Top-K 的 score（因为 gather 先裁剪）；未选中的 patch 得分不参与梯度。若想让未选中 patch 也受训练信号，需要软选择或额外正则。

5) **patch 数量推导假设**  
   - `num_patches = (h//stride_h)*(w//stride_w)` 写死在 __init__；若 backbone 的实际 patch 数与 stride 不符（自定义 patch size、重采样等），`keeped_patches` 将与真实比例不一致，可考虑用 `RGB_cash.shape[1]` 计算。

6) **初始化日志噪声**  
   - `MultiModalSDTPS` 在 __init__ 打印参数信息，每次构建模型都会输出，在分布式场景可能产生大量重复日志。
