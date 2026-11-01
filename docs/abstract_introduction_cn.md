# Toward Causally Valid Medical World Models: Learning Intervention-Sufficient Temporal States from Factual Trajectories

## 摘要

Medical World Model 旨在从患者纵向历史中学习可复用的动态模拟器，以生成不同治疗策略下的长期患者轨迹，而不仅是预测患者在实际治疗路径下的未来。然而，因果效应在统计意义上可识别，并不意味着有限容量的 World Model 能够准确实现这些效应：将完整患者历史压缩为潜状态时，模型可能合并对事实预测相似、但对干预响应不同的历史；即使治疗相关信息已进入潜状态，动作条件化的递归转移也可能无法在多步模拟中正确保持和更新这些信息。本文将这一问题形式化为纵向因果识别与潜在动态实现之间的缺口，并提出两个必要条件：潜状态对所研究的干预查询充分，以及该状态在动作条件化转移下保持闭合。基于此，我们提出动作条件化的 Temporal Joint-Embedding Predictive Architecture（Temporal JEPA），利用当前患者历史、实际治疗序列及其已观测未来历史，为多时间跨度潜状态转移提供自监督，而无需反事实结果、治疗效应标签或策略价值监督。我们在满足 consistency、sequential exchangeability 和 positivity 的受控医疗动态系统中，系统考察潜状态容量、多步误差传播及未见治疗时间安排对纵向治疗效应轨迹的影响。实验表明，在保持事实预测性能的候选模型中，Temporal JEPA 显著降低潜状态闭合误差，并在多数测试治疗安排下改善长期效应轨迹估计，但其收益仍受到潜状态充分性和策略支持范围的限制。本文由此将可靠的 Medical World Model 学习表述为一个因果动态的模型实现问题：因果假设决定干预效应能否从数据中识别，而潜状态及其递归转移决定这些可识别效应能否被同一个模拟器可靠地表示、传播和复用。

## 1. 引言

患者的疾病进展由当前生理状态、既往治疗、治疗依从性及临床决策随时间共同塑造。Medical World Model 希望将这些纵向信息整合为患者状态，并在输入候选治疗计划后，模拟患者未来可能经历的完整轨迹。这一目标不同于在既有临床路径下预测下一次化验结果或终点风险：一个可用于临床推演的 World Model 应当能够从同一患者历史出发，比较连续治疗、间歇治疗、延迟开始或提前停止等不同时间安排，并描述治疗作用何时产生、如何累积以及何时消退。换言之，它试图成为一个能够支持多个治疗策略、多个时间跨度和多个后续临床查询的可复用动态模拟器。

纵向因果推断为这些查询何时能够从观测数据中恢复提供了理论基础。在 consistency、sequential exchangeability 和 positivity 成立时，不同治疗策略下的潜在结果可以通过 longitudinal g-formula 识别。对于单一且预先指定的估计量，专用的 g-computation、加权或双重稳健方法可能是更直接的选择。本文研究 Medical World Model，是因为其目标更广：同一个学习到的状态转移应当被反复用于不同治疗序列、预测时长、结局变量和下游决策标准。因此，World Model 可以被理解为 longitudinal g-computation 的一种潜在状态实现，其中高维患者历史被压缩为

\[
Z_t=E_\phi(H_t),
\]

并通过动作条件化转移递归演化：

\[
\widehat Z_{t+1}=F_\theta(\widehat Z_t,A_t),
\qquad
\widehat O_{t+1}=G_\psi(\widehat Z_{t+1}).
\]

这一实现引入了一个独立于因果识别的新问题。即使完整历史满足

\[
Y_{t+1:T}^{\bar a}\perp A_t\mid H_t,
\]

也不能自动推出压缩后的表示满足

\[
Y_{t+1:T}^{\bar a}\perp A_t\mid Z_t.
\]

Encoder 可能丢弃病情严重程度、累积治疗暴露或治疗效应修饰变量，使原本在完整历史中已被测量的混杂和响应信息，在 latent space 中变得不可用。更进一步，即使初始潜状态包含了这些信息，实际部署仍要求模型反复使用自身生成的状态：

\[
Z_t\xrightarrow{A_t}\widehat Z_{t+1}
\xrightarrow{A_{t+1}}\cdots
\xrightarrow{A_{t+H-1}}\widehat Z_{t+H}.
\]

如果 transition 没有按照真实规律更新治疗记忆、累积剂量或效应修饰状态，局部误差就会在多步模拟中传播。因此，因果效应由观测历史可识别，并不等价于有限 World Model 已经学习到能够实现该效应的潜在动态。

本文将一个可靠的潜在因果模拟器概括为两个要求。第一，表示必须具有干预充分性：被映射到相同潜状态的患者历史，应当在所有受支持的治疗计划下具有相同的未来响应分布。第二，转移必须具有动态闭合性：当前历史的编码经过实际动作演化后，应当到达与下一时刻真实历史编码一致的状态，即

\[
F_\theta^{(h)}
\left(E_\phi(H_t),A_{t:t+h-1}\right)
\approx E_\phi(H_{t+h}).
\]

前者决定模型是否保留了回答干预问题所需的信息，后者决定这些信息能否在递归 rollout 中持续存在。我们证明，任意历史压缩并不保持 exchangeability；相反，当潜状态对支持内干预充分、潜在转移正确闭合且 outcome decoder 正确时，递归 latent rollout 可以实现由完整历史识别的 longitudinal g-formula。对于近似模型，我们进一步将长期治疗效应误差上界分解为两条干预分支上的潜状态传播误差与 decoder residual，从而为潜状态闭合提供可检验的理论角色。

现有的一步或多步预测目标主要约束模型在事实治疗路径上的 observable trajectory。闭环 multi-step training 能够缓解 teacher forcing 与部署之间的差距，但低 observable error 并不唯一确定潜状态应如何演化：多个不同的 latent trajectory 可能由 decoder 映射为相似的事实结局，却在改变治疗时间安排后产生不同响应。与此同时，现实数据通常无法提供同一患者在多种未来治疗安排下的长期反事实标签。因此，核心方法问题是：能否仅利用已观测的事实纵向轨迹，为治疗条件化的潜状态演化提供更直接的时间监督？

为此，我们提出动作条件化的 Temporal JEPA。给定当前患者历史表示和随后实际发生的治疗序列，模型递归预测多个未来时间点的潜状态，并将其与目标编码器对相应未来事实历史产生的表示对齐：

\[
\widehat Z_{t+h}
=F_\theta^{(h)}
\left(E_\phi(H_t),A_{t:t+h-1}^{\mathrm{obs}}\right),
\]

\[
Z_{t+h}^{*}=E_{\bar\phi}(H_{t+h}),
\qquad
\mathcal L_{\mathrm{TJEPA}}
=\sum_{h\in\{1,2,4,8\}}
d_{\mathcal Z}
\left(
\widehat Z_{t+h},
\operatorname{sg}(Z_{t+h}^{*})
\right).
\]

该目标不使用模拟器隐藏状态、替代治疗下的结果、显式 treatment-effect label 或 policy-value target。它也不从观察数据中创造新的因果信息；其作用是在识别假设和数据支持给定时，利用事实未来历史约束可识别动态在有限潜状态中的多步实现。

我们在受控医疗动态系统中检验这一观点。模拟器包含状态依赖治疗效应、延迟治疗作用、累积剂量和状态依赖的医生治疗策略，并按构造满足因果识别条件。训练阶段所有方法只使用事实患者轨迹；测试阶段则从相同模拟器状态分叉不同治疗计划，以获得精确的纵向因果效应轨迹。实验分别考察潜状态压缩、多步 rollout 和 held-out treatment schedules，并比较一步预测、多步 observable rollout 与 Temporal JEPA。为避免使用反事实结果选择模型，我们先保留事实 rollout 性能达到基线要求的候选模型，再依据事实验证数据上的 latent closure 选择模型，最后才在未参与选择的反事实效应轨迹上评价。

我们的具体贡献如下：

1. **从因果识别到模型实现。** 我们区分了“纵向效应能否由完整历史识别”与“有限 World Model 能否通过共享潜状态和递归转移实现这些效应”，并将 Medical World Model 解释为 latent g-computation。

2. **潜在因果动态的理论条件。** 我们证明 arbitrary compression 不保持 exchangeability，定义 intervention sufficiency 与 causal transition closure，并说明二者在什么条件下足以支持有效的 latent g-computation。

3. **无需反事实标签的时间自监督。** 我们提出动作条件化的 Temporal JEPA，利用事实未来历史监督多时间跨度潜状态传播，而不依赖 treatment-effect 或 policy-value labels。

4. **面向干预的行为评价。** 我们不只评价事实预测误差，而是在多个 latent capacities、rollout horizons 和 held-out schedules 上评价完整治疗效应轨迹，检验潜状态闭合能否为可迁移的治疗时间动态提供增量证据。

本文不声称解决未测量混杂或完全缺乏治疗支持的问题。相反，我们控制这些识别因素，以隔离一个更具体的学习问题：当因果效应原则上可识别时，怎样使一个有限、可复用的 Medical World Model 更可靠地保存和传播实现这些效应所需的时间状态。

