"""
生成Diffusion Policy和ACT对比图片的Python脚本
需要安装: pip install matplotlib numpy seaborn
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_noise_schedule():
    """生成噪声调度对比图"""
    T = 1000
    t = np.linspace(0, T, T)

    # 线性调度
    beta_min, beta_max = 0.0001, 0.02
    beta_linear = beta_min + (beta_max - beta_min) * t / T
    alpha_linear = 1 - beta_linear
    alpha_bar_linear = np.cumprod(alpha_linear)

    # 余弦调度
    s = 0.008
    f_t = np.cos((t/T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar_cosine = f_t / f_t[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制alpha_bar
    axes[0].plot(t, alpha_bar_linear, label='线性调度', linewidth=2)
    axes[0].plot(t, alpha_bar_cosine, label='余弦调度', linewidth=2)
    axes[0].set_xlabel('时间步 t', fontsize=12)
    axes[0].set_ylabel(r'$\bar{\alpha}_t$', fontsize=12)
    axes[0].set_title('累积噪声系数对比', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 绘制beta
    beta_cosine = 1 - alpha_bar_cosine[1:] / alpha_bar_cosine[:-1]
    axes[1].plot(t, beta_linear, label='线性调度', linewidth=2)
    axes[1].plot(t[1:], beta_cosine, label='余弦调度', linewidth=2)
    axes[1].set_xlabel('时间步 t', fontsize=12)
    axes[1].set_ylabel(r'$\beta_t$', fontsize=12)
    axes[1].set_title('噪声强度对比', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noise_schedule.png', dpi=300, bbox_inches='tight')
    print("Generated: noise_schedule.png")

if __name__ == "__main__":
    create_noise_schedule()
