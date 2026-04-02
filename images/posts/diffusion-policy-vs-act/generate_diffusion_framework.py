"""生成Diffusion Policy框架图"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 6))

# 观测输入
obs_box = FancyBboxPatch((0.5, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                         edgecolor='blue', facecolor='lightblue', linewidth=2)
ax.add_patch(obs_box)
ax.text(1.25, 3, 'Observation\n(Image + State)', ha='center', va='center', fontsize=11, weight='bold')

# 编码器
encoder_box = FancyBboxPatch((3, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                            edgecolor='green', facecolor='lightgreen', linewidth=2)
ax.add_patch(encoder_box)
ax.text(3.75, 3, 'Vision\nEncoder', ha='center', va='center', fontsize=11, weight='bold')

# 噪声
noise_box = FancyBboxPatch((5.5, 4), 1.2, 0.8, boxstyle="round,pad=0.1",
                          edgecolor='gray', facecolor='lightgray', linewidth=2)
ax.add_patch(noise_box)
ax.text(6.1, 4.4, 'Noise\nN(0,I)', ha='center', va='center', fontsize=10)

# 扩散模型
diff_box = FancyBboxPatch((5.5, 2.3), 2.5, 1.4, boxstyle="round,pad=0.1",
                         edgecolor='red', facecolor='lightyellow', linewidth=3)
ax.add_patch(diff_box)
ax.text(6.75, 3, 'Diffusion Model\n(U-Net/Transformer)', ha='center', va='center', fontsize=12, weight='bold')

# 动作输出
action_box = FancyBboxPatch((9, 2.5), 1.5, 1, boxstyle="round,pad=0.1",
                           edgecolor='purple', facecolor='plum', linewidth=2)
ax.add_patch(action_box)
ax.text(9.75, 3, 'Action\nChunk', ha='center', va='center', fontsize=11, weight='bold')

# 箭头
arrows = [
    ((2, 3), (3, 3)),
    ((4.5, 3), (5.5, 3)),
    ((6.1, 4), (6.5, 3.7)),
    ((8, 3), (9, 3))
]

for start, end in arrows:
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)

# 迭代去噪标注
ax.text(6.75, 1.5, 'Iterative Denoising\n(T steps)', ha='center', fontsize=10,
       style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_xlim(0, 11)
ax.set_ylim(1, 5)
ax.axis('off')
ax.set_title('Diffusion Policy Framework', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('diffusion_policy_framework.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Generated: diffusion_policy_framework.png")
