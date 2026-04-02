"""
生成注意力权重热力图
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_attention_map():
    """生成Transformer注意力权重可视化"""
    np.random.seed(42)

    # 模拟注意力权重矩阵
    seq_len = 16
    attention = np.random.rand(seq_len, seq_len)

    # 添加对角线模式（自注意力）
    for i in range(seq_len):
        attention[i, max(0, i-3):min(seq_len, i+4)] += 0.5

    # 归一化
    attention = attention / attention.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attention, cmap='YlOrRd', ax=ax, cbar_kws={'label': '注意力权重'})
    ax.set_xlabel('键（Key）位置', fontsize=12)
    ax.set_ylabel('查询（Query）位置', fontsize=12)
    ax.set_title('ACT Transformer注意力权重可视化', fontsize=14)

    plt.tight_layout()
    plt.savefig('attention_map.png', dpi=300, bbox_inches='tight')
    print("Generated: attention_map.png")

if __name__ == "__main__":
    create_attention_map()
