"""
生成扩散过程示意图
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def create_diffusion_process():
    """生成前向扩散和反向去噪过程示意图"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # 时间步
    steps = [0, 250, 500, 750, 1000]
    x_pos = np.linspace(0, 10, len(steps))

    # 绘制前向过程（上半部分）
    for i, (step, x) in enumerate(zip(steps, x_pos)):
        noise_level = step / 1000
        # 模拟动作序列的噪声化
        y = np.linspace(2, 3, 50)
        if i == 0:
            signal = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.3 + 2.5
        else:
            signal = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.3 * (1-noise_level) + 2.5
            signal += np.random.randn(50) * noise_level * 0.3

        ax.plot(np.ones(50)*x, signal, 'b-', linewidth=2, alpha=0.7)
        ax.text(x, 3.5, f't={step}', ha='center', fontsize=10)

    # 箭头：前向过程
    for i in range(len(x_pos)-1):
        ax.annotate('', xy=(x_pos[i+1]-0.3, 2.5), xytext=(x_pos[i]+0.3, 2.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.text(5, 4, '前向扩散过程（添加噪声）', ha='center', fontsize=14,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 绘制反向过程（下半部分）
    for i, (step, x) in enumerate(zip(reversed(steps), x_pos)):
        noise_level = step / 1000
        y = np.linspace(0.5, 1.5, 50)
        if i == len(steps)-1:
            signal = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.3 + 1
        else:
            signal = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.3 * (1-noise_level) + 1
            signal += np.random.randn(50) * noise_level * 0.3

        ax.plot(np.ones(50)*x, signal, 'g-', linewidth=2, alpha=0.7)
        ax.text(x, 0, f't={step}', ha='center', fontsize=10)

    # 箭头：反向过程
    for i in range(len(x_pos)-1):
        ax.annotate('', xy=(x_pos[i+1]-0.3, 1), xytext=(x_pos[i]+0.3, 1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))

    ax.text(5, -0.5, '反向去噪过程（恢复动作）', ha='center', fontsize=14,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1, 4.5)
    ax.axis('off')
    ax.set_title('Diffusion Policy: 前向扩散与反向去噪', fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig('diffusion_process.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Generated: diffusion_process.png")

if __name__ == "__main__":
    create_diffusion_process()
