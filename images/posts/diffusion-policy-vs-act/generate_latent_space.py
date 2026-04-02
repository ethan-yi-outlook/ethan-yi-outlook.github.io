"""
生成CVAE潜在空间可视化
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_latent_space():
    """生成CVAE潜在空间可视化"""
    np.random.seed(42)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 模拟不同行为模式的潜在变量
    n_samples = 200

    # 模式1: 左侧抓取
    z1 = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, 2])
    # 模式2: 右侧抓取
    z2 = np.random.randn(n_samples, 2) * 0.5 + np.array([2, 2])
    # 模式3: 上方抓取
    z3 = np.random.randn(n_samples, 2) * 0.5 + np.array([0, -2])

    # 绘制潜在空间分布
    axes[0].scatter(z1[:, 0], z1[:, 1], alpha=0.6, s=30, label='左侧抓取', c='blue')
    axes[0].scatter(z2[:, 0], z2[:, 1], alpha=0.6, s=30, label='右侧抓取', c='red')
    axes[0].scatter(z3[:, 0], z3[:, 1], alpha=0.6, s=30, label='上方抓取', c='green')
    axes[0].set_xlabel('潜在维度 z₁', fontsize=12)
    axes[0].set_ylabel('潜在维度 z₂', fontsize=12)
    axes[0].set_title('CVAE潜在空间聚类', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 绘制潜在空间插值
    t = np.linspace(0, 1, 10)
    interp = np.outer(1-t, z1[0]) + np.outer(t, z2[0])

    axes[1].scatter(z1[:, 0], z1[:, 1], alpha=0.3, s=20, c='blue')
    axes[1].scatter(z2[:, 0], z2[:, 1], alpha=0.3, s=20, c='red')
    axes[1].plot(interp[:, 0], interp[:, 1], 'ko-', linewidth=2, markersize=8, label='插值路径')
    axes[1].set_xlabel('潜在维度 z₁', fontsize=12)
    axes[1].set_ylabel('潜在维度 z₂', fontsize=12)
    axes[1].set_title('潜在空间插值', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cvae_latent_space.png', dpi=300, bbox_inches='tight')
    print("Generated: cvae_latent_space.png")

if __name__ == "__main__":
    create_latent_space()
