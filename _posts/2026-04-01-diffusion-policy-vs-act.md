---
layout: post
title: "Diffusion Policy vs ACT：机器人学习算法对比分析"
date: 2026-04-01
tags: [AI algorithm]
toc: true
comments: true
author: yisheng
---

## 引言

Diffusion Policy（扩散策略）和ACT（Action Chunking with Transformers，基于Transformer的动作分块）是当前机器人学习和视觉运动控制领域的两种主流方法。本文将从算法原理、部署特性、优缺点等多个维度对这两种方法进行深入对比分析。

在机器人操作任务中，如何从视觉观测生成精确的动作序列一直是核心挑战。传统的行为克隆方法往往难以处理多模态行为和长时序依赖。Diffusion Policy和ACT分别从生成模型和序列建模的角度提供了创新性的解决方案。

<!-- more -->

## 算法原理详解

### Diffusion Policy（扩散策略）

![Diffusion Policy框架图](/images/posts/diffusion-policy-vs-act/diffusion_policy_framework.png)
*图1: Diffusion Policy整体框架*

Diffusion Policy将动作生成问题建模为条件去噪扩散过程，这一思想源自图像生成领域的DDPM（Denoising Diffusion Probabilistic Models）：

#### 数学推导

**1. 前向扩散过程（Forward Process）**

给定专家演示的动作序列 $\mathbf{a}_0$，前向过程逐步添加高斯噪声：

$$q(\mathbf{a}_t | \mathbf{a}_{t-1}) = \mathcal{N}(\mathbf{a}_t; \sqrt{1-\beta_t}\mathbf{a}_{t-1}, \beta_t\mathbf{I})$$

其中 $\beta_t$ 是噪声调度参数。利用重参数化技巧，可以直接从 $\mathbf{a}_0$ 采样任意时刻 $t$ 的状态：

$$q(\mathbf{a}_t | \mathbf{a}_0) = \mathcal{N}(\mathbf{a}_t; \sqrt{\bar{\alpha}_t}\mathbf{a}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$

**2. 反向去噪过程（Reverse Process）**

学习反向过程来从噪声恢复动作，条件为观测 $\mathbf{o}$：

$$p_\theta(\mathbf{a}_{t-1} | \mathbf{a}_t, \mathbf{o}) = \mathcal{N}(\mathbf{a}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{a}_t, \mathbf{o}, t), \boldsymbol{\Sigma}_\theta(\mathbf{a}_t, \mathbf{o}, t))$$

**3. 训练目标（Training Objective）**

简化的训练目标是预测添加的噪声：

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{a}_0, \boldsymbol{\epsilon}} \left[ \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{a}_t, \mathbf{o}, t)\|^2 \right]$$

其中 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$ 是添加的噪声，$\boldsymbol{\epsilon}_\theta$ 是神经网络预测的噪声。

![扩散过程示意图](/images/posts/diffusion-policy-vs-act/diffusion_process.png)
*图2: 前向扩散与反向去噪过程*

**核心机制：**
- **前向扩散过程**：在训练阶段，将专家演示的动作序列逐步添加高斯噪声，直到变成纯噪声
- **反向去噪过程**：学习一个神经网络来逆转这个过程，从噪声中恢复出有意义的动作序列
- **条件生成**：去噪过程以当前和历史的观测（图像、状态等）为条件，确保生成的动作与环境状态相匹配

**4. 采样算法（DDPM Sampling）**

推理时从纯噪声开始迭代去噪：

```
输入: 观测 o, 扩散步数 T
初始化: a_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    ε_pred = ε_θ(a_t, o, t)
    a_{t-1} = 1/√α_t * (a_t - (1-α_t)/√(1-ᾱ_t) * ε_pred) + √β_t * z
输出: a_0
```

**5. DDIM加速采样**

DDIM（Denoising Diffusion Implicit Models）通过确定性采样减少步数：

$$\mathbf{a}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\underbrace{\left(\frac{\mathbf{a}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, \mathbf{o}, t)}{\sqrt{\bar{\alpha}_t}}\right)}_{\text{预测的}x_0} + \underbrace{\sqrt{1-\bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_\theta(\mathbf{a}_t, \mathbf{o}, t)}_{\text{方向指向}a_t}$$

**推理过程：**
1. 从标准高斯分布采样初始噪声
2. 迭代执行去噪步骤（通常10-100步）
3. 每步根据当前观测条件预测并去除部分噪声
4. 最终得到清晰的动作序列

**动作表示：**
- 生成动作块（action chunks）：一次预测未来多个时间步的动作
- 通过时间集成（temporal ensembling）提高稳定性

![网络架构图](/images/posts/diffusion-policy-vs-act/diffusion_network.png)
*图3: Diffusion Policy网络架构（U-Net/Transformer）*

### ACT (Action Chunking with Transformers)

![ACT框架图](/images/posts/diffusion-policy-vs-act/act_framework.png)
*图4: ACT整体框架*

ACT采用条件变分自编码器（CVAE）结合Transformer架构来实现动作序列预测：

#### 数学推导

**1. CVAE基本框架**

给定观测 $\mathbf{o}$ 和动作序列 $\mathbf{a}$，CVAE学习条件分布：

$$p_\theta(\mathbf{a}|\mathbf{o}) = \int p_\theta(\mathbf{a}|\mathbf{z}, \mathbf{o})p(\mathbf{z})d\mathbf{z}$$

其中 $\mathbf{z}$ 是潜在变量，通常假设先验 $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$

**2. 变分下界（ELBO）**

由于后验 $p_\theta(\mathbf{z}|\mathbf{a}, \mathbf{o})$ 难以计算，引入变分后验 $q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o})$：

$$\log p_\theta(\mathbf{a}|\mathbf{o}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o})}[\log p_\theta(\mathbf{a}|\mathbf{z}, \mathbf{o})] - D_{KL}(q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o}) \| p(\mathbf{z}))$$

**3. 训练目标**

最大化ELBO等价于最小化：

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi}[\|\mathbf{a} - \mathbf{a}_\theta(\mathbf{z}, \mathbf{o})\|^2]}_{\text{重建损失}} + \underbrace{\beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o}) \| \mathcal{N}(0, \mathbf{I}))}_{\text{KL正则化}}$$

其中 $\beta$ 是平衡系数（β-VAE）

**4. KL散度计算**

假设 $q_\phi(\mathbf{z}|\mathbf{a}, \mathbf{o}) = \mathcal{N}(\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi^2\mathbf{I})$，KL散度有闭式解：

$$D_{KL} = \frac{1}{2}\sum_{i=1}^{d}\left(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\right)$$

**5. 网络结构**

```
视觉编码器: CNN/ResNet提取图像特征 f_v
状态编码器: MLP编码本体感受状态 f_s
潜在编码器: Transformer编码动作 → μ_φ, σ_φ
解码器: Transformer([f_v, f_s, z]) → 动作序列
```

![ACT网络架构](/images/posts/diffusion-policy-vs-act/act_network.png)
*图5: ACT Transformer编码器-解码器架构*

**推理过程：**
1. 从先验分布p(z)采样潜在变量（通常是标准高斯）
2. 将潜在变量与观测特征输入解码器
3. 单次前向传播生成完整的动作块
4. 执行动作块中的第一个动作，滑动窗口更新

**关键技术：**
- **时间位置编码**：为动作序列添加时间信息
- **交叉注意力机制**：解码器关注编码的观测特征
- **KL退火**：训练初期降低KL权重，避免后验坍塌

**6. 训练算法**

```
for each batch (o, a):
    # 编码观测
    f = Encoder(o)
    
    # 编码动作到潜在空间
    μ, log_σ = LatentEncoder(a, o)
    z = μ + σ * ε, where ε ~ N(0, I)
    
    # 解码生成动作
    a_pred = Decoder(z, f)
    
    # 计算损失
    L_recon = ||a - a_pred||²
    L_KL = KL(N(μ, σ²) || N(0, I))
    L = L_recon + β * L_KL
    
    # 更新参数
    optimize(L)
```

![CVAE潜在空间](/images/posts/diffusion-policy-vs-act/cvae_latent_space.png)
*图6: CVAE潜在空间可视化*

## 核心算法差异对比

![算法对比图](/images/posts/diffusion-policy-vs-act/comparison_table.png)
*图7: Diffusion Policy vs ACT核心差异*

| 维度 | Diffusion Policy | ACT |
|------|------------------|-----|
| **生成模型类型** | 去噪扩散模型 | 条件变分自编码器 |
| **网络架构** | U-Net / Transformer | Transformer编码器-解码器 |
| **推理步数** | 多步迭代（10-100步） | 单次前向传播 |
| **多模态建模** | 隐式（通过随机采样） | 显式（潜在变量） |
| **训练目标** | 去噪分数匹配 | ELBO（重建+KL） |
| **采样策略** | DDPM/DDIM采样器 | 从先验分布采样 |
| **时序建模** | 通过U-Net或注意力机制 | Transformer自注意力 |

## 实现细节与技巧

### Diffusion Policy实现要点

**1. 噪声调度策略**

常用的噪声调度方案：

- **线性调度**: $\beta_t = \beta_{\min} + (\beta_{\max} - \beta_{\min})\frac{t}{T}$
- **余弦调度**: $\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)^2$

![噪声调度对比](/images/posts/diffusion-policy-vs-act/noise_schedule.png)
*图8: 不同噪声调度策略的影响*

**2. 动作分块与时间集成**

```python
# 伪代码
def predict_action(obs_history, model, T=50):
    # 初始化噪声
    action_chunk = torch.randn(chunk_size, action_dim)
    
    # 迭代去噪
    for t in reversed(range(T)):
        noise_pred = model(action_chunk, obs_history, t)
        action_chunk = denoise_step(action_chunk, noise_pred, t)
    
    return action_chunk[0]  # 返回第一个动作
```

**3. 条件编码策略**

- 使用ResNet/ViT编码视觉观测
- 通过FiLM层或交叉注意力注入条件
- 历史观测的时序编码

### ACT实现要点

**1. 避免后验坍塌**

关键技巧：

- **KL退火**: $\beta_t = \min(1.0, t / T_{\text{warmup}})$
- **自由位（Free bits）**: $\mathcal{L}_{KL} = \max(\lambda, D_{KL})$
- **增强潜在变量**: 确保解码器依赖z

```python
# KL退火示例
def compute_loss(a_pred, a_true, mu, log_var, step, warmup_steps=10000):
    recon_loss = F.mse_loss(a_pred, a_true)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # KL退火
    beta = min(1.0, step / warmup_steps)
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss
```

**2. Transformer配置**

- 层数: 4-8层
- 注意力头: 8-16个
- 隐藏维度: 256-512
- 动作块长度: 10-100步

![Transformer注意力图](/images/posts/diffusion-policy-vs-act/attention_map.png)
*图9: ACT中的注意力权重可视化*

## 部署实施考量

### Diffusion Policy部署特性

**计算资源需求：**
- **推理成本较高**：迭代去噪需要10-100次神经网络前向传播
- **灵活的质量-速度权衡**：可通过调整去噪步数在精度和速度间平衡
  - 少步数（10-20步）：更快但可能牺牲精度
  - 多步数（50-100步）：更高质量但延迟增加
- **硬件加速**：GPU加速显著提升性能，但CPU也可运行
- **内存占用**：需存储中间去噪状态，内存需求中等

**实时性能表现：**
- **典型延迟**：50-200ms（取决于步数和硬件）
- **动作分块优势**：一次生成多步动作，分摊计算成本
- **并行化潜力**：可使用DDIM等快速采样器减少步数
- **适用场景**：适合10-20Hz控制频率的操作任务

**实现复杂度：**
- 需要实现扩散训练框架（噪声调度、采样器等）
- 噪声调度策略需要针对任务调优
- 相对标准的训练流程，收敛稳定
- 开源实现丰富（如diffusion_policy库）

### ACT部署特性

**计算资源需求：**
- **推理高效**：单次前向传播即可生成动作序列
- **GPU友好**：Transformer架构充分利用GPU并行计算
- **内存密集**：长序列的注意力机制内存占用较大
- **模型规模**：参数量取决于Transformer层数和隐藏维度

**实时性能表现：**
- **低延迟**：典型延迟20-50ms
- **高频控制**：适合50Hz以上的控制频率
- **确定性推理**：给定潜在变量，输出确定
- **适用场景**：适合需要快速响应的精细操作

**实现复杂度：**
- 标准Transformer训练流程
- **CVAE训练挑战**：
  - KL坍塌问题：潜在变量被忽略
  - 需要β-VAE平衡重建和正则化
  - 可能需要KL退火等训练技巧
- 潜在空间需要仔细调优
- 相对成熟的实现框架（基于PyTorch/TensorFlow）

## 优势与劣势深度分析

### Diffusion Policy

**优势：**

1. **卓越的多模态建模能力**
   - 自然处理一个观测对应多个有效动作的情况
   - 例如：抓取物体时可以从不同角度接近
   - 通过随机采样探索不同的行为模式

2. **训练稳定性高**
   - 去噪目标函数简单直观，梯度稳定
   - 不存在模式坍塌或后验坍塌问题
   - 对超参数相对鲁棒

3. **表达能力强**
   - 可以建模复杂的高维动作分布
   - 捕获动作序列中的细微模式
   - 在复杂操作任务上表现优异

4. **最先进的性能**
   - 在多个机器人操作基准测试中达到SOTA
   - 特别是在需要精细控制的任务上表现突出
   - 泛化能力较强

**劣势：**

1. **推理速度慢**
   - 迭代去噪过程计算开销大
   - 不适合超高频控制（>50Hz）
   - 实时性受限于硬件性能

2. **超参数敏感**
   - 噪声调度需要针对任务调优
   - 去噪步数影响质量和速度
   - 不同任务可能需要不同配置

3. **内存开销**
   - 需要存储中间去噪状态
   - 批处理时内存占用较大

4. **可解释性较差**
   - 扩散过程不如显式潜在变量直观
   - 难以分析和调试生成的动作
   - 黑盒特性较强

### ACT

**优势：**

1. **推理速度快**
   - 单次前向传播完成预测
   - 适合实时控制应用
   - 延迟低，响应迅速

2. **计算效率高**
   - 部署成本低
   - 适合资源受限的硬件平台
   - 能耗相对较低

3. **显式多模态建模**
   - 潜在变量提供可解释的行为模式
   - 可以通过潜在空间插值探索行为
   - 便于分析和可视化

4. **时序建模能力强**
   - Transformer擅长捕获长程时序依赖
   - 注意力机制提供可解释性
   - 适合复杂的时序任务

**劣势：**

1. **训练不稳定**
   - CVAE容易出现后验坍塌
   - 潜在变量可能被忽略
   - 需要精心设计训练策略

2. **模式覆盖不足**
   - 可能遗漏训练数据中的罕见但有效行为
   - 潜在空间维度限制表达能力
   - 多模态建模能力弱于扩散模型

3. **潜在空间调优困难**
   - β-VAE的β参数需要仔细平衡
   - KL权重影响重建质量和多样性
   - 不同任务需要不同配置

4. **数据效率较低**
   - 通常需要比扩散方法更多的演示数据
   - 对数据质量要求较高
   - 小样本场景下性能下降明显

## 应用场景推荐

### 选择Diffusion Policy的场景：

1. **多模态行为任务**
   - 存在多种有效解决方案的任务（如多角度抓取）
   - 需要探索不同策略的场景
   - 行为多样性比速度更重要

2. **精度优先任务**
   - 精细操作任务（如装配、插入）
   - 对动作精度要求极高
   - 可以容忍一定的推理延迟

3. **计算资源充足**
   - 有GPU加速支持
   - 控制频率要求不高（10-30Hz）
   - 可以接受50-200ms延迟

4. **复杂操作任务**
   - 高维动作空间
   - 长时序依赖
   - 需要处理复杂的视觉输入

### 选择ACT的场景：

1. **实时控制需求**
   - 高频控制任务（>50Hz）
   - 对延迟敏感的应用
   - 需要快速响应的场景

2. **资源受限环境**
   - 边缘设备部署
   - 嵌入式系统
   - 计算资源有限的机器人平台

3. **可解释性需求**
   - 需要理解行为模式
   - 通过潜在空间分析行为
   - 调试和优化策略

4. **明确时序结构**
   - 任务具有清晰的时序模式
   - 需要建模长程依赖
   - 序列预测任务

## 实际性能对比

### 基准测试结果

在多个机器人操作基准上的表现对比：

**Push-T任务（2D推动）：**
- Diffusion Policy: 成功率 ~90%
- ACT: 成功率 ~85%
- Diffusion Policy在处理多模态推动路径时表现更优

**ALOHA双臂操作：**
- ACT: 在双臂协调任务上表现出色，低延迟优势明显
- Diffusion Policy: 在精细操作上精度更高

**RLBench操作任务：**
- Diffusion Policy: 在复杂长时序任务上成功率更高
- ACT: 在简单快速任务上效率更高

### 训练效率对比

- **收敛速度**：ACT通常需要更多训练轮次
- **数据需求**：Diffusion Policy对数据量要求相对较低
- **训练稳定性**：Diffusion Policy训练过程更稳定

## 未来发展方向

### Diffusion Policy的改进方向

1. **加速推理**
   - 一致性模型（Consistency Models）：单步生成
   - 知识蒸馏：将多步模型蒸馏为少步模型
   - 潜在扩散：在低维潜在空间进行扩散

2. **架构优化**
   - 更高效的去噪网络设计
   - 自适应步数调整
   - 混合精度推理

### ACT的改进方向

1. **训练稳定性**
   - 改进的CVAE训练方法
   - 更好的正则化策略
   - 自适应KL权重调整

2. **表达能力增强**
   - 更大的潜在空间
   - 层次化潜在变量
   - 结合扩散模型的优势

### 混合方法探索

1. **扩散+Transformer**：结合两者优势
2. **分层策略**：高层用ACT规划，低层用Diffusion执行
3. **自适应选择**：根据任务特性动态选择方法

## 总结

Diffusion Policy和ACT代表了机器人视觉运动控制的两种强大范式。Diffusion Policy在表达能力和多模态行为建模上表现卓越，但推理速度较慢；ACT提供了快速部署和显式潜在结构，但训练需要更多技巧。

**选择建议：**
- 追求最高精度和多模态能力 → Diffusion Policy
- 需要实时响应和快速部署 → ACT
- 资源充足的研究场景 → Diffusion Policy
- 工业部署和边缘计算 → ACT

最终选择取决于具体应用需求、计算约束和任务特性。随着两种方法的持续发展，未来可能会出现融合两者优势的混合方法。

---

**参考文献：**
- Diffusion Policy: [Chi et al., 2023 - Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://diffusion-policy.cs.columbia.edu/)
- ACT: [Zhao et al., 2023 - Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://tonyzhaozh.github.io/aloha/)
- DDPM: [Ho et al., 2020 - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

**END**

