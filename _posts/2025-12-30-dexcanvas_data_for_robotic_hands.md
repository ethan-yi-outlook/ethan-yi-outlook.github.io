---
layout: post
title: "看不见的操控数据=没用？DexCanvas用强化学习'复活'百万轨迹的物理灵魂"
date: 2025-12-30
tags: [AI Data Robotic]
toc: true
comments: true
author: yisheng
---

## 前言 图片阅读有问题？参考微信链接 [dexcanvas](https://mp.weixin.qq.com/s/AEm9BJEuisq0yPeS58dVnw)

国内外各大机器人公司都在发布海量数据集——百万级轨迹、千小时演示，数字一个比一个惊人。

但在灵巧手领域，有个所有人都回避的尖锐问题：**这些20+自由度的复杂系统，数据里竟然**没有一条包含完整的力信息**。

<!-- more -->

为什么强调灵巧手？因为只有高自由度的灵巧手，才能真正覆盖人类操作的全部场景：

- 夹爪能抓取，但无法做手内操作
- 三指手能简单操控，但无法使用工具
- **只有五指灵巧手，才能像人手一样万能**

但恰恰是这些最有潜力的灵巧手，面临着最严峻的数据困境：你教灵巧手抓鸡蛋，它学会了20个关节的轨迹，却不知道5个手指该用多大力；你演示拧瓶盖，它模仿了复杂的手指协调，却不懂每个接触点的扭矩；你展示精密装配，它复制了毫米级路径，却感受不到微妙的接触反馈。

这就是灵巧手操作数据的现状——**自由度越高，力的缺失越致命**。

## 行业困境：灵巧手数据的"不可能三角"

现有灵巧手数据集陷入了**"不可能三角"**：**规模、真实、力觉——你只能选两个**。

- **选规模+真实？** 真实灵巧手演示最多几万条，且没有力信息
- **选规模+力觉？** 数字孪生真数据，但sim2real鸿沟难以跨越
- **选真实+力觉？** 实验室级小规模数据，不足以支撑大规模预训练

![图 1 人手与灵巧手抓鸡蛋对比](https://mmbiz.qpic.cn/mmbiz_jpg/ONuxLzOgbUkcfDIAMqUvT3u6Zq7boSsYBaDVoon5vILhXfkhvJdBoEShggtcFvQ7icDKo9sznIhia3icKXGDYQJWw/0?wx_fmt=jpeg)

![图 2 灵巧手数据的"不可能三角"]()

### 01 为什么破不了这个三角？

**根本原因：灵巧手的力太贵、太难。**

一只五指灵巧手有20+个关节，要全面采集力信息需要在每个关节和接触面都装传感器。这不仅改变了手的动力学特性，让动作不再自然，更带来了巨额成本和工程复杂度。多个手指间的力如何分配、内力外力如何区分，这些都是传统方法无法解决的难题。

所以大家选择了逃避——**"反正视觉模仿学习也能work，为什么要为灵巧手死磕力？"**

### 02 传统方法的局限：数据源决定了上限

不同方法采集的是不同的数据源：

- **遥操作**：采集机器人数据，但成本高、效率低、动作受限于操作界面
- **视频学习**：采集人类数据，但遮挡导致深度不准，公开数据不够系统，更别提力信息
- **纯仿真**：大多是仿真机器人手，与特定硬件绑定，容易利仿真漏洞

![图 3 数据获取方式与困境](https://mmbiz.qpic.cn/mmbiz_png/ONuxLzOgbUkcfDIAMqUvT3u6Zq7boSsYzia7Sq7PQjWYvqYZDmiad0NmKveD6EsQVNnC0QQb55xEjY6CQsp2CY3Q/640?wx_fmt=png)

**DexCanvas的选择：人类操作是通用接口。**

我们相信人类操作包含了最自然、最通用的操作智慧。通过大规模真实人类演示（1000小时）作为基础，在仿真中恢复物理信息，既避免了纯仿真的漏洞利用，又不局限于特定硬件限制。这不是在三角形里选两个，而是找到了新的突破口。

![图 4 灵巧操作示范数据采集与合成](https://mmbiz.qpic.cn/mmbiz_png/ONuxLzOgbUkcfDIAMqUvT3u6Zq7boSsYrIT9FgwPdfqHhusAZrxgfyJyUEcsEvHKWnOkxJC4ZIuOWvJeRTwmCw/640?wx_fmt=png)

## DexCanvas的"邪道"：用强化学习反推物理

### 核心洞察：轨迹里藏着力的线索

一个简单的物理事实：**如果你知道物体怎么动的，又知道它的质量和几何，理论上可以反推出作用力。**

但传统方法行不通：

- 逆动力学？接触点未知，方程无解
- 优化方法？非凸问题，局部最优遍地

DexCanvas的方法：**让强化学习智能体在仿真中"重演"真实轨迹。**

### 三步走的"复活"流程

**Step 1：精确采集人类演示**

- 20个动捕相机，亚毫米精度（特别感谢青瞳视觉的设备及技术支持）
- 22类抓取分类（基于Cutkosky体系）
- 不是随便抓，是系统化覆盖操作空间

![图 5 22类抓取分类（基于Cutkosky体系）](https://mmbiz.qpic.cn/mmbiz_png/ONuxLzOgbUkcfDIAMqUvT3u6Zq7boSsYwfqVM2MwCjYRg6gaJlZYojEaOUcs7WejZR7kNicXRo8s1GshVoI5UtQ/640?wx_fmt=png)

[视频 1 系统化覆盖操作空间](https://mpvideo.qpic.cn/0bc3jycv6aafmeahm76pt5ufktwdl5hakxya.f10002.mp4?dis_k=8c258fedce4ce73ae6bc8a82553e844b&dis_t=1767058755&play_scene=10120&auth_info=CqDcpXMfPVTviPC3XHVqPz9oSzdgQ2VDAGQ9SgdTJX00TGMbEyU2ATR0ERY+NlMwKz4=&auth_key=0847d3668cb6dabd7ab2da26bcd02a27&vid=wxv_4168920205789151242&format_id=10002&support_redirect=0&mmversion=false)

**Step 2：仿真重演+物理反推**

我们将真实轨迹放进物理仿真环境，让强化学习智能体尝试复现这些动作。这里的关键是：智能体必须施加正确的力、选择正确的接触点，才能在仿真中成功重现人类的演示。通过这个过程，我们自动获得了每个时刻的接触点位置和接触力大小——这些正是原始数据中缺失的关键信息。

[视频 2 仿真重演+物理反推](https://mpvideo.qpic.cn/0bc3xecmeaaehyao7ygov5ufjoodyk4qjqqa.f10002.mp4?dis_k=eb9961a727773ab8c4e5ab502df1e967&dis_t=1767058755&play_scene=10120&auth_info=Drt.yog+BwS29Xv4ylvF4kOWg9Mh8/YBI2QQBlP0kACX92N0VkSUVxZAJkcEQdPGcAZylk&auth_key=9a14f7149602889d821f0451132857aa&vid=wxv_4168938872908431373&format_id=10002&support_redirect=0&mmversion=false)

**Step 3：数据增强x100**

- 改变物体质量→生成新的力分布
- 调整摩擦系数→适应不同材质
- 调整姿态位置→提高鲁棒性
- 1000小时人类演示 → 10万小时物理增强数据

[视频 3 数据增强x100](https://mpvideo.qpic.cn/0bc3m4bg6aacniaf2egidnufez6dn5tqe3ya.f10002.mp4?dis_k=cdcc04b3b101a8c89a2bb38961ecfd15&dis_t=1767058755&play_scene=10120&auth_info=HNTp/e0jTWhR7dWkvARxbj8+YxI2ZEc1EwM4OEsGDyUsMUBnTRV3YwQ2KUUdZjJXMCo1&auth_key=0886eda95b0c17e89757fdda1d5d3747&vid=wxv_4168958305370325016&format_id=10002&support_redirect=0&mmversion=false)

### 03 为什么这不是"作弊"？

有人会问："你这不就是仿真数据吗？"

**关键区别：**

- 纯仿真：从零开始生成机器人动作，容易利用仿真漏洞产生不真实的行为
- DexCanvas：基于1000小时真实人类演示，仿真只是"显影"隐藏的物理信息

更重要的是，我们仿真的是人手而非机器人手——这让数据不局限于特定硬件限制，成为真正通用的操作数据。

就像考古学家通过骨骼复原恐龙的肌肉——骨骼（轨迹）是真的，肌肉（力）是科学推断的。大规模仿真真实数据确保了我们不会偏离物理现实。

### 04 看看效果：从数据到真机灵巧手

我们用DexCanvas训练的策略，已经在真实的五指灵巧手上跑通了。

[视频 4 DexCanvas数据集真机使用](https://mpvideo.qpic.cn/0bc3dmczgaaf5eanl26pxbufkg6dsmnqleya.f10002.mp4?dis_k=dbf4297da865dfe5ad25a5f60caf95c5&dis_t=1767058755&play_scene=10120&auth_info=S5ujhIQkRG5V6IuisVx0bmg9ZUw5YhUxFlxpaksBCSB7M00yGhZ+ZQAzd0MQPjdXZykz&auth_key=9ed40cf49d2332c183bb351f61684837&vid=wxv_4169048038544949250&format_id=10002&support_redirect=0&mmversion=false)

## 开源计划：不卷规模，卷质量

### 01 我们要开源什么？

**不是又一个"百万轨迹"数据集**，而是专门为灵巧手设计的物理完备数据：

- **物理增强的灵巧手操作数据** - 每条轨迹都有完整的多指力/接触标签，专门针对20+自由度系统优化
- **从人手到灵巧手的完整pipeline** - 动捕→MANO拟合→物理重演→五指灵巧手执 行，开源所有处理代码
- **HuggingFace即插即用** - 不是让你下载10TB然后不知道怎么用，而是提供预处理的、可直接训练的版本

### 02 为什么现在开源？

整个领域正在经历数据收集的爆发期，但大部分努力都集中在扩大规模上。现有的大规模数据集推动了视觉和模仿学习的进步，但始终缺少物理交互这个关键维度。与此同时，研究社区仍热衷于比较规模有限的经典数据集进行基准测试。

**DexCanvas不想参与"谁的数据集更大"的军备竞赛。**

我们想证明：**100小时高质量的物理数据，胜过100万小时的纯视觉轨迹。**

## 写在最后

灵巧手是具备智能的终极极战——20+自由度、复杂接触、精 细控制。

数据不缺，缺的是"对的"数据。特别是对于灵巧手而言，**没有力信息的轨迹，便是没有灵魂的躯壳。**

当所有人都在比谁的数据更大时，**DexCanvas**选择了另一条路：**专注于灵巧手，让每一条轨迹都包含完整的物理信息**。

因为我们相信，只有解决了最难的灵巧操作问题，具备智能才能真正实现。

![DexCanvas](https://mmbiz.qpic.cn/mmbiz_jpg/ONuxLzOgbUkcfDIAMqUvT3u6Zq7boSsYjM0D9POcBTsrwYk1ZXYAewBT3uan3kHIwEWxianpNIc2EYMmCYEtH4w/640?wx_fmt=jpeg)

**END**