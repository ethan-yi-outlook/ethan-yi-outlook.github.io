# 图片说明文档

本目录包含博客文章《Diffusion Policy vs ACT：机器人学习算法对比分析》所需的图片。

## 需要的图片列表

### 1. diffusion_policy_framework.png
**描述**: Diffusion Policy整体框架图
**内容要求**:
- 展示从观测到动作生成的完整流程
- 包含：观测编码器 → 条件扩散模型 → 动作序列输出
- 标注前向扩散和反向去噪过程

### 2. diffusion_process.png
**描述**: 前向扩散与反向去噪过程示意图
**内容要求**:
- 左侧：前向过程（动作 → 噪声）
- 右侧：反向过程（噪声 → 动作）
- 显示多个时间步的中间状态
- 标注噪声水平变化

### 3. diffusion_network.png
**描述**: Diffusion Policy网络架构（U-Net/Transformer）
**内容要求**:
- U-Net或Transformer的详细结构
- 显示条件注入位置
- 标注输入输出维度

### 4. act_framework.png
**描述**: ACT整体框架图
**内容要求**:
- CVAE架构：编码器-潜在空间-解码器
- 显示观测编码、潜在变量采样、动作解码
- 标注训练和推理的不同路径

### 5. act_network.png
**描述**: ACT Transformer编码器-解码器架构
**内容要求**:
- Transformer的encoder-decoder结构
- 显示自注意力和交叉注意力层
- 标注潜在变量注入位置

### 6. cvae_latent_space.png
**描述**: CVAE潜在空间可视化
**内容要求**:
- 2D/3D潜在空间的散点图
- 不同行为模式的聚类
- 可选：潜在空间插值示例

### 7. comparison_table.png
**描述**: Diffusion Policy vs ACT核心差异对比图
**内容要求**:
- 并排对比两种方法的关键特性
- 可视化推理流程差异
- 突出多模态建模方式的不同

### 8. noise_schedule.png
**描述**: 不同噪声调度策略的影响
**内容要求**:
- 线性调度 vs 余弦调度的曲线对比
- 显示α_t和β_t随时间的变化
- 对比不同调度对性能的影响

### 9. attention_map.png
**描述**: ACT中的注意力权重可视化
**内容要求**:
- 热力图显示注意力分布
- 展示解码器如何关注观测特征
- 时序注意力模式

## 图片规格要求

- 格式: PNG（支持透明背景）
- 分辨率: 至少 1200x800 像素
- 风格: 学术论文风格，清晰简洁
- 配色: 专业配色方案，避免过于鲜艳
- 文字: 中英文标注，字体清晰可读

## 创建建议

可以使用以下工具创建图片：
- **框架图**: draw.io, Figma, PowerPoint
- **网络架构**: NN-SVG, PlotNeuralNet
- **数学可视化**: Python (matplotlib, seaborn)
- **注意力图**: Python (matplotlib, seaborn)
